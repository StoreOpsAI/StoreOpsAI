"""화면 전체 S3D 창 점수(OOF)에 '연속 K개 창' 유지 규칙을 얹어 이벤트 단위로 평가한다(점검용, 미세조정 전).

- OOF: 날짜 fold마다 학습 fold의 라벨 있는(1/0) 창으로 선형 분류기를 학습하고, 보류 fold의 모든 창(애매한 -1 포함)을 채점한다.
- 경보: 같은 영상에서 점수 >= thr인 창이 K개 이상 연속된 구간 하나가 경보 하나(창 간격 2초).
- 적중: 구간 안에 양성(1) 창이 있음. 오경보: 구간 안에 양성도 애매(-1)도 없음. 애매한 창만 있는 구간은 판정하지 않는다.
- 오경보/시간 = 오경보 수 / 정상 창 시간(정상 창 수 x 2초). K와 thr은 OOF 점수 분포의 분위수 후보에서 재현율 제약을 만족하는 최소 오경보 조합을 고른다.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.global_split import SPLIT_PATH
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.oof_search import candidates_from_values

STRIDE_SEC = 2.0
KS = (1, 2, 3)
RECALL_TARGETS = (0.90, 0.95)


def load() -> tuple[list[dict], dict]:
    clips = []
    for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")):
        p = Path(BEHAVIOR_FEATURE_ROOT) / "fullscene" / f"{r['clip']}.npz"
        if r["official_split"] != "train" or r["q_fall"] == "True" or not p.is_file():
            continue
        z = np.load(p)
        y = z["label"].astype(int) if r["behavior"] == "전도" else np.zeros(len(z["start"]), int)
        clips.append({"clip": r["clip"], "behavior": r["behavior"], "fold": int(r["fold"]), "x": z["s3d"].astype(np.float32), "y": y})
    return clips, {}


def oof(clips: list[dict]) -> None:
    folds = sorted({c["fold"] for c in clips})
    for k in folds:
        tr = [c for c in clips if c["fold"] != k]
        x = np.concatenate([c["x"][c["y"] >= 0] for c in tr]); y = np.concatenate([c["y"][c["y"] >= 0] for c in tr]).astype(np.int64)
        model = _fit_linear(x, y)
        for c in clips:
            if c["fold"] == k:
                c["s"] = model(c["x"])


def alarm_runs(scores: np.ndarray, thr: float, k: int) -> list[tuple[int, int]]:
    """점수 >= thr가 k개 이상 연속된 구간 [시작 창, 끝 창]. 구간 하나가 경보 하나다(끊길 때까지 같은 경보)."""
    hit = np.concatenate([[False], scores >= thr, [False]])
    edges = np.flatnonzero(hit[1:] != hit[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2]) if b - a >= k]


def evaluate(clips: list[dict], thr: float, k: int) -> dict:
    """구간에 양성(1) 창이 있으면 적중, 양성도 애매(-1)도 없으면 오경보, 애매만 있으면 판정하지 않는다."""
    hits = fall_clips = false = 0
    for c in clips:
        runs = alarm_runs(c["s"], thr, k)
        if (c["y"] == 1).any():
            fall_clips += 1
            hits += any((c["y"][a:b + 1] == 1).any() for a, b in runs)
        false += sum(1 for a, b in runs if not (c["y"][a:b + 1] != 0).any())
    hours = sum(int((c["y"] == 0).sum()) for c in clips) * STRIDE_SEC / 3600
    return {"recall": hits / fall_clips, "false_alarms": int(false), "fa_per_hour": false / hours, "hours": hours}


def main() -> None:
    clips, _ = load()
    oof(clips)
    allscores = np.concatenate([c["s"] for c in clips])
    cand = candidates_from_values(allscores[np.concatenate([c["y"] for c in clips]) >= 0], 80)
    print(f"클립 {len(clips)}개, 전도 클립 {sum((c['y'] == 1).any() for c in clips)}개, 정상 창 시간 {evaluate(clips, 0.5, 1)['hours']:.1f}시간")
    for target in RECALL_TARGETS:
        print(f"\n[클립 재현율 >= {target}]  K | 점수 기준 | 재현율 | 오경보 | 오경보/시간")
        for k in KS:
            rows = [(t, evaluate(clips, t, k)) for t in cand]
            ok = [(t, r) for t, r in rows if r["recall"] >= target]
            if not ok:
                print(f"  K={k} | 만족 조합 없음"); continue
            t, r = min(ok, key=lambda tr: (tr[1]["false_alarms"], -tr[1]["recall"]))
            print(f"  K={k} | {t:.3f} | {r['recall']:.3f} | {r['false_alarms']} | {r['fa_per_hour']:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
