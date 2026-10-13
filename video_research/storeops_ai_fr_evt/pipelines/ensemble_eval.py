"""A안(화면 전체) + C1(사람 크롭) S3D OOF 로짓 (+ 관계 특징)을 학습형으로 결합해 이벤트 단위로 평가한다(단순 평균 아님).

- 두 모델이 모두 OOF 점수를 가진 fold만 쓴다(같은 창, (영상, 시작 위치)로 맞춤). 결합은 날짜 fold CV로 다른 fold의 OOF 점수에서 학습한다.
- 후보: A만 / C1만 / A+C1 / A+C1+관계. 분류기는 표준화 + 클래스 균형 로지스틱 하나. 평가기와 라벨 규칙은 eval_finetune과 같다. 공식 validation 미사용.
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.eval_finetune import CLASSES, evaluate
from pipelines.global_split import SPLIT_PATH
from pipelines.fusion_eval import clips_for
from pipelines.oof_search import candidates_from_values

DIRS = {"A": "finetune_s3d", "C1": "finetune_s3d_c1", "H": "finetune_s3d_h", "P": "pose_head"}
RECALL_TARGETS = (0.90, 0.95)


def load_source(name: str) -> dict[int, dict[str, np.ndarray]]:
    out = {}
    for f in sorted(glob.glob(str(APP_ROOT / "output" / DIRS[name] / "oof_fold*.npz"))):
        out[int(Path(f).stem.replace("oof_fold", ""))] = dict(np.load(f))
    return out


def merged(folds_use: list[int] | None = None, rel: bool = False) -> dict[str, np.ndarray]:
    a, c = load_source("A"), load_source("C1")
    folds = sorted(set(a) & set(c) & (set(folds_use) if folds_use else set(a)))
    rows: dict[str, list] = {k: [] for k in ("clip", "start", "label", "cls", "behavior", "fold", "la", "lc", "rel")}
    fold_of = {r["clip"]: int(r["fold"]) for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))}
    rel_dir = Path(BEHAVIOR_FEATURE_ROOT) / ("relwin" + __import__("os").environ.get("OBJDET_SUFFIX", ""))   # OBJDET_SUFFIX=30이면 30에폭 탐지기 관계 특징
    for k in folds:
        key_c = {(cl, int(s)): i for i, (cl, s) in enumerate(zip(c[k]["clip"], c[k]["start"]))}
        cache: dict[str, dict[int, np.ndarray]] = {}
        for i, (cl, s) in enumerate(zip(a[k]["clip"], a[k]["start"])):
            j = key_c.get((cl, int(s)))
            if j is None:
                continue
            if rel:
                if cl not in cache:
                    f = rel_dir / f"{cl}.npz"
                    cache[cl] = dict(zip(np.load(f)["start"].tolist(), np.load(f)["summary"])) if f.is_file() else {}
                if int(s) not in cache[cl]:
                    continue
                rows["rel"].append(cache[cl][int(s)])
            for key in ("clip", "start", "label", "cls", "behavior"):
                rows[key].append(a[k][key][i])
            rows["fold"].append(k), rows["la"].append(a[k]["logits"][i]), rows["lc"].append(c[k]["logits"][j])
    return {k: np.array(v) for k, v in rows.items() if v}


def run(d: dict[str, np.ndarray], mode: str, cls: int) -> np.ndarray:
    feats = {"A": d["la"], "C1": d["lc"], "A+C1": np.hstack([d["la"], d["lc"]])}
    if "rel" in d:
        feats["A+C1+rel"] = np.hstack([d["la"], d["lc"], d["rel"]])
    x = feats[mode].astype(np.float32)
    y = np.where(d["cls"] == cls, 1, 0).astype(np.int64)
    valid = d["label"] != -1
    z = np.zeros(len(y))
    for k in sorted(set(d["fold"].tolist())):
        tr, te = valid & (d["fold"] != k), d["fold"] == k
        p = _fit_linear(x[tr], y[tr])(x[te])
        z[te] = np.log(p + 1e-6) - np.log(1 - p + 1e-6)
    return z


def main() -> None:
    use_rel = "--rel" in sys.argv
    d = merged(rel=use_rel)
    folds = sorted(set(d["fold"].tolist()))
    print(f"결합 평가: 창 {len(d['clip'])}개, fold {folds} (A안·C1 모두 있는 fold{', 관계 특징 포함' if use_rel else ''})")
    modes = ["A", "C1", "A+C1"] + (["A+C1+rel"] if use_rel else [])
    for cls, name in CLASSES.items():
        valid = d["label"] != -1
        print(f"\n[{name}]")
        for mode in modes:
            z = run(d, mode, cls)
            clips = clips_for(d, z, cls)
            cand = candidates_from_values(z[valid], 120)
            row = []
            for target in RECALL_TARGETS:
                ok = [(t, r) for t, r in ((t, evaluate(clips, t)) for t in cand) if r["recall"] >= target]
                if ok:
                    t, r = min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"]))
                    row.append(f"재현율>={target}: {r['recall']:.3f} 오경보 {r['fa_per_hour']:.2f}/시간")
                else:
                    row.append(f"재현율>={target}: 만족 없음")
            print(f"  {mode:9s} | " + " | ".join(row))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
