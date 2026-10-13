"""OOF 탐색으로 정한 전도 운영점(output/lying_search/selected_params.json)을 반품 정상 풀에 적용해 오경보를 센다.

- 반품 풀은 학습·운영점 선택에 쓰지 않았다. 이 결과로 운영점을 다시 고르지 않는다(AGENTS.md 7과 같은 취지). 모델은 전도 first_pass + 매장이동 정상 학습.
- 오경보는 data/hard_normal_review.json 판정(검수자 표기 있음)으로 분류한다. 판정 없는 경보는 unreviewed로 센다.
출력: output/hard_normal/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from config.config import LYING_RECALL_TARGETS
from models.fall_detector import lying_alarms
from models.new_object_detector import _iou
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_lying import fit_method
from pipelines.hard_normal_candidates import FPS, OUT, POSE_DIR
from pipelines.lying_pose_dataset import load_dataset, pose_features

REVIEW = APP_ROOT / "data" / "hard_normal_review.json"
MATCH_FRAMES = 30  # 판정 후보와 같은 경보로 보는 프레임 차(3fps 10초)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(LYING_RECALL_TARGETS[1]), help="selected_params.json의 재현율 제약 키")
    args = parser.parse_args()
    params = json.loads((APP_ROOT / "output" / "lying_search" / "selected_params.json").read_text(encoding="utf-8"))["selected_by_recall_target"][args.target]
    review = json.loads(REVIEW.read_text(encoding="utf-8"))["candidates"]
    score = fit_method(load_dataset(include_normal=True, include_aug=False), "mlp")
    counts, minutes, per_group, unreviewed = Counter(), 0.0, Counter(), []
    for npz in sorted(POSE_DIR.glob("*.npz")):
        data = np.load(npz)
        feats = np.array([pose_features(b, k) for b, k in zip(data["box"], data["kpt"])]).reshape(len(data["box"]), -1)
        by_frame = defaultdict(list)
        for f, box, s in zip(data["frame"], data["box"], score({"feat": feats})):
            if s >= params["thr"]:
                by_frame[int(f)].append(tuple(map(float, box)))
        n = int(data["n_frames"])
        minutes += n / FPS / 60
        for f, box in lying_alarms([(f, by_frame.get(f, [])) for f in range(n)], FPS, hold_sec=params["hold"], link_iou=params["link_iou"]):
            hit = next((r for r in review.values() if r["clip"] == npz.stem and abs(r["frame"] - f) <= MATCH_FRAMES and _iou(r["box"], box) >= 0.3), None)
            label = hit["label"] if hit else "unreviewed"
            counts[label] += 1
            per_group["_".join(npz.stem.split("_")[4:7:2])] += 1
            if not hit:
                unreviewed.append({"clip": npz.stem, "frame": f, "box": box})
    total = sum(counts.values())
    result = {"params": {k: params[k] for k in ("thr", "link_iou", "hold")}, "pool_minutes": round(minutes, 1), "alarms": total,
              "false_alarm_per_hour": round(total / minutes * 60, 2), "by_review_label": dict(counts), "by_site_date": dict(per_group), "unreviewed": unreviewed}
    (OUT / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "unreviewed"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
