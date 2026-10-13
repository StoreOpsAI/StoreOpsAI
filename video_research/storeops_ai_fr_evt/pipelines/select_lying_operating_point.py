"""전도 운영점(점수 경보 기준, 같은 자리 유지 시간)을 데이터로 정한다. 사람이 정한 0.5·1초를 쓰지 않는다. 공식 validation·quarantine 미사용.

1) dev 5-fold: 각 영상을 자기 fold를 안 본 모델로 채점한다(out-of-fold 점수).
2) (기준 × 유지 시간) 격자마다 클립 적중 수와 분당 오경보를 센다.
3) 목표 오경보율(카메라당 하루 N건 -> N / MINUTES_PER_DAY 건/분) 이하인 조합 중 재현율이 가장 높은 것을 고르고,
   같으면 유지 시간이 짧은(경보가 빠른) 것, 그래도 같으면 기준이 낮은 것을 고른다.
고른 조합을 동결해 새 시험 집합에 한 번 적용한다(이 스크립트는 dev 선택까지만 한다). 출력: output/lying_eval/operating_point.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from config.config import LYING_FA_BUDGETS_PER_DAY, LYING_GRID_HOLDS, LYING_GRID_THRESHOLDS, MINUTES_PER_DAY
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_lying import FPS, fit_method, judge_alarms
from pipelines.eval_lying_threshold import alarms_at
from pipelines.lying_manifest import FOLDS
from pipelines.lying_pose_dataset import load_dataset
from models.fall_detector import lying_alarms
from collections import defaultdict
from config.config import LYING_TRANSITION_FRAMES


def oof_clip_scores(ds, method):
    """{영상 이름: 사람별 점수} (영상을 학습에 안 쓴 모델로 채점)."""
    folds = np.array([c["fold"] for c in ds])
    out = {}
    for k in range(FOLDS):
        train = [c for c, f in zip(ds, folds) if f != k]
        score = fit_method(train, method)
        for c, f in zip(ds, folds):
            if f == k and not c.get("aug"):
                out[c["clip"]] = score(c)
    return out


def grid_counts(clips, scores, thr, hold, link_iou=None):
    """(적중, 전도 클립 수, 오경보 수, 정상 분) 한 (기준, 유지 시간)."""
    hits = fall_clips = false = 0
    minutes = 0.0
    for c in clips:
        by_frame = defaultdict(list)
        for f, box, s in zip(c["frame"], c["box"], scores[c["clip"]]):
            if s >= thr:
                by_frame[int(f)].append(tuple(map(float, box)))
        alarms = lying_alarms([(f, by_frame.get(f, [])) for f in range(c["frames"])], FPS, hold_sec=hold, link_iou=link_iou)
        hit, fa, _ = judge_alarms(c, alarms)
        hits, fall_clips, false = hits + hit, fall_clips + bool(c["falls"]), false + fa
        first = max(min(f["start"] for f in c["falls"]) - LYING_TRANSITION_FRAMES, 0) if c["falls"] else c["frames"]
        minutes += first / FPS / 60
    return hits, fall_clips, false, minutes


def choose(grid, budget_per_day):
    ok = [g for g in grid if g["fp_per_min"] <= budget_per_day / MINUTES_PER_DAY]
    if not ok:
        return None
    return min(ok, key=lambda g: (-g["recall"], g["hold"], g["thr"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="mlp")
    args = parser.parse_args()
    ds = load_dataset(include_normal=True, include_aug=False)
    clips = [c for c in ds if not c.get("aug")]
    scores = oof_clip_scores(ds, args.method)
    grid = []
    for thr in LYING_GRID_THRESHOLDS:
        for hold in LYING_GRID_HOLDS:
            h, fc, fa, minutes = grid_counts(clips, scores, thr, hold)
            grid.append({"thr": thr, "hold": hold, "recall": round(h / fc, 4), "hits": h, "fall_clips": fc, "false": fa,
                         "minutes": round(minutes, 1), "fp_per_min": fa / minutes})
    chosen = {str(b): choose(grid, b) for b in LYING_FA_BUDGETS_PER_DAY}
    path = APP_ROOT / "output" / "lying_eval" / "operating_point.json"
    path.write_text(json.dumps({"method": args.method, "grid": grid, "chosen": chosen}, ensure_ascii=False, indent=1), encoding="utf-8")
    for b, g in chosen.items():
        print(f"목표 하루 {b}건: " + ("조건 만족 조합 없음" if g is None else
              f"기준 {g['thr']} 유지 {g['hold']}초 -> 재현율 {g['recall']} ({g['hits']}/{g['fall_clips']}), 오경보 {g['false']}건/{g['minutes']}분"))


if __name__ == "__main__":
    main()
