"""유기 경보(행동 AND 새 물건 + 확인 + 손 + 주인 떠남, eval_abandon_hand의 fused_hand)가 다른 행동·반품 영상에서
울린 장면을 이미지와 목록으로 남긴다. 오경보를 유형별로 사람이 분류하기 위한 것(경보마다 번호를 이미지에 적음)."""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import torch

from config.config import OBJECT_VERIFY_WEIGHTS
from models.object_verifier import ObjectVerifier
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import load_xclip
from pipelines.eval_abandon_false_alarms import list_sources
from pipelines.eval_abandon_fusion import full_head
from pipelines.eval_abandon_hand import alarms_with_hand

OUT = APP_ROOT / "output" / "abandon_alarms_v4"


def tile(video, item, number, size=240):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, item["confirmed_frame"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((size, size, 3), np.uint8)
    x1, y1, x2, y2 = item["box"]
    pad = max(x2 - x1, y2 - y1)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    crop = frame[cy1:y2 + pad, cx1:x2 + pad].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
    crop = cv2.resize(crop, (size, size))
    cv2.putText(crop, str(number), (6, 30), 0, 1.0, (0, 255, 255), 2)
    return crop


def main():
    verifier = ObjectVerifier(APP_ROOT / OBJECT_VERIFY_WEIGHTS)
    head = full_head(APP_ROOT / "output" / "abandonment_training" / "samples.csv",
                     APP_ROOT / "output" / "backbone_compare" / "xclip.npz")
    processor, model = load_xclip(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    records, tiles = [], []
    for source, videos in list_sources().items():
        for video in videos:
            alarms, _, fps = alarms_with_hand(video, verifier, head, processor, model)
            for run in alarms["fused_hand"]:
                item = run[0]
                records.append({"n": len(records), "source": source, "video": video.name if hasattr(video, "name") else str(video),
                                "second": round(item["confirmed_frame"] / fps, 1), "box": list(item["box"]),
                                "verify_label": item.get("verify_label")})
                tiles.append(tile(video, item, records[-1]["n"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alarms.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))[1].tofile(
            str(OUT / "alarms.jpg"))
    print(len(records), {s: sum(r["source"] == s for r in records) for s in {r["source"] for r in records}})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
