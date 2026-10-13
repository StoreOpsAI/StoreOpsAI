"""합성 영상(make_abandon_composites)에서 유기 재현율을 방식별로 잰다.

- 적중: 경보 박스가 붙인 물건 박스와 IoU >= 0.2이고, 물건이 나타난 뒤에 확정된 것. 결합 방식은 사건 안 물건 하나라도 맞으면 적중.
- 그 밖의 경보는 바탕 영상에서 나온 것이므로 오경보로 따로 센다(같은 바탕 원본의 오경보는 eval_abandon_false_alarms 반품).
- 붙인 물건이 제대로 보이는지 사람이 확인하도록 물건 장면 타일을 남긴다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import OBJECT_VERIFY_WEIGHTS
from models.new_object_detector import _iou
from models.object_verifier import ObjectVerifier
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import load_xclip
from pipelines.eval_abandon_false_alarms import METHODS, _tile, video_alarms
from pipelines.eval_abandon_fusion import full_head


def is_hit(alarm, label):
    items = alarm if isinstance(alarm, list) else [alarm]
    # 물건이 나타나기 전부터 있던 후보(first_frame 이전)는 같은 자리여도 붙인 물건이 아니다
    return any(item["first_frame"] >= label["appear_frame"] and _iou(item["box"], label["box"]) >= 0.2
               for item in items)


def main():
    parser = argparse.ArgumentParser(description="합성 영상 유기 재현율 (방식별)")
    parser.add_argument("--composite-dir", type=Path, default=APP_ROOT / "output" / "abandon_composites")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "abandonment_training" / "samples.csv")
    parser.add_argument("--feature-cache", type=Path, default=APP_ROOT / "output" / "backbone_compare" / "xclip.npz")
    args = parser.parse_args()

    labels = json.loads((args.composite_dir / "labels.json").read_text(encoding="utf-8"))
    verifier = ObjectVerifier(APP_ROOT / OBJECT_VERIFY_WEIGHTS)
    head = full_head(args.manifest, args.feature_cache)
    processor, model = load_xclip(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    counts = defaultdict(lambda: {"total": 0, "hit": 0, "other_alarms": 0})
    per_video, tiles = [], []
    for label in labels:
        video = args.composite_dir / label["video"]
        # 합성 영상의 사람 박스·추적 캐시는 바탕 원본과 이름이 달라 새로 계산된다(붙인 물건 영향 포함)
        alarms, _ = video_alarms(video, verifier, head, processor, model, whole_runs=True)
        row = {"video": label["video"], "scenario": label["scenario"]}
        for method in METHODS:
            hits = [is_hit(alarm, label) for alarm in alarms[method]]
            for key in (method, f"{method}/{label['scenario']}"):
                counts[key]["total"] += 1
                counts[key]["hit"] += any(hits)
                counts[key]["other_alarms"] += hits.count(False)
            row[method] = any(hits)
        per_video.append(row)
        tiles.append(_tile(video, {"confirmed_frame": label["appear_frame"] + int(5 * label["fps"]), "box": label["box"]}))
        print(json.dumps(row, ensure_ascii=False))

    summary = {key: {**value, "recall": round(value["hit"] / value["total"], 3)} for key, value in sorted(counts.items())}
    (args.composite_dir / "results.json").write_text(
        json.dumps({"summary": summary, "per_video": per_video}, ensure_ascii=False, indent=2), encoding="utf-8")
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        cv2.imencode(".jpg", sheet)[1].tofile(str(args.composite_dir / "objects.jpg"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
