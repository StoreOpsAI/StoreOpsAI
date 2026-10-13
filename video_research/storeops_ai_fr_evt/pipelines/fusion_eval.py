"""B안(후기 결합) OOF 평가: S3D OOF 로짓(4클래스) + 사람-물건 관계 요약 특징을 합쳐 유기·절도 점수를 만든다. 이벤트 단위 평가는 eval_finetune과 같다.

- 결합 학습 데이터 = 다른 fold들의 모든 창(S3D 점수는 그 창을 학습하지 않은 모델이 낸 OOF 로짓이라 누수가 없다), 보류 fold = 시험. 날짜 fold 그대로.
- 라벨: 그 행동 클립의 1 = 양성, 그 밖(다른 행동 사건 포함)은 0, 애매(-1)는 학습·평가 제외.
- 후보: s3d(S3D 4로짓만) / rel(관계 요약만) / s3d+rel. 분류기는 표준화 + 클래스 균형 로지스틱 하나(설정 고정).
- 사용: S3D OOF 파일(output/finetune_s3d/oof_fold*.npz)이 있는 fold만, 관계 특징(relwin)이 있는 영상만 쓴다. 공식 validation은 읽지 않는다.
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.eval_finetune import CLASSES, evaluate
from pipelines.global_split import SPLIT_PATH
from pipelines.oof_search import candidates_from_values

TARGET_CLASSES = {2: "유기", 3: "절도"}
RECALL_TARGETS = (0.90, 0.95)


def load() -> dict[str, np.ndarray]:
    """창 단위 표: clip, start, fold, behavior, label, cls, logits(4), rel(56)."""
    folds = {r["clip"]: int(r["fold"]) for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))}
    parts = [np.load(f) for f in sorted(glob.glob(str(APP_ROOT / "output" / "finetune_s3d" / "oof_fold*.npz")))]
    rows = {k: np.concatenate([p[k] for p in parts]) for k in ("clip", "start", "label", "cls", "behavior", "logits")}
    rel_dir, cache, rel = Path(BEHAVIOR_FEATURE_ROOT) / "relwin", {}, np.full((len(rows["clip"]), 56), np.nan, np.float32)
    for i, (clip, start) in enumerate(zip(rows["clip"], rows["start"])):
        if clip not in cache:
            f = rel_dir / f"{clip}.npz"
            cache[clip] = (dict(zip(np.load(f)["start"].tolist(), np.load(f)["summary"])) if f.is_file() else {})
        if int(start) in cache[clip]:
            rel[i] = cache[clip][int(start)]
    rows["rel"], rows["fold"] = rel, np.array([folds[c] for c in rows["clip"]])
    keep = ~np.isnan(rel).any(1)
    return {k: v[keep] for k, v in rows.items()}


def design(d: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "s3d":
        return d["logits"]
    if mode == "rel":
        return d["rel"]
    return np.hstack([d["logits"], d["rel"]])


def fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    p = _fit_linear(x_tr, y_tr)(x_te)
    return np.log(p + 1e-6) - np.log(1 - p + 1e-6)  # 확률 -> 로짓(순위·기준 탐색용)


def clips_for(d: dict[str, np.ndarray], z: np.ndarray, cls: int) -> list[dict]:
    """eval_finetune.evaluate가 받는 형식: [{z, y}]. y: 1 = 이 행동 양성, 0 = 정상·다른 행동, -1 = 애매."""
    out = []
    for clip in np.unique(d["clip"]):
        m = np.flatnonzero(d["clip"] == clip)
        m = m[np.argsort(d["start"][m])]
        own = d["behavior"][m][0] == CLASSES[cls]
        lab = d["label"][m]
        other = -1 if os.getenv("IGNORE_OTHER_EVENTS", "0") == "1" else 0  # 1이면 다른 행동 사건 구간은 판정 제외
        out.append({"clip": clip, "z": z[m], "y": np.where(lab == 1, 1 if own else other, lab)})
    return out


def main() -> None:
    d = load()
    folds = sorted(set(d["fold"].tolist()))
    print(f"창 {len(d['clip'])}개(S3D OOF + 관계 특징 모두 있음), fold {folds}")
    if len(folds) < 2:
        print("결합 학습에 fold가 2개 이상 필요합니다."); return
    for cls, name in TARGET_CLASSES.items():
        y_all = np.where(d["cls"] == cls, 1, 0)
        valid = d["label"] != -1
        print(f"\n[{name}] 양성 창 {int((y_all[valid] == 1).sum())}개")
        for mode in ("s3d", "rel", "s3d+rel"):
            z = np.zeros(len(y_all))
            for k in folds:
                tr, te = valid & (d["fold"] != k), d["fold"] == k
                x = design(d, mode).astype(np.float32)
                z[te] = fit_predict(x[tr], y_all[tr].astype(np.int64), x[te])
            clips = clips_for(d, z, cls)
            cand = candidates_from_values(z[valid], 120)
            row = []
            for target in RECALL_TARGETS:
                ok = [(t, r) for t, r in ((t, evaluate(clips, t)) for t in cand) if r["recall"] >= target]
                if ok:
                    t, r = min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"]))
                    row.append(f"재현율>={target}: {r['recall']:.3f} 오경보 {r['fa_per_hour']:.2f}/시간(음성 창 {r['fa_windows_per_hour']:.1f}/시간)")
                else:
                    row.append(f"재현율>={target}: 만족 없음")
            print(f"  {mode:8s} | " + " | ".join(row))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
