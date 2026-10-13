"""파인튜닝 물건 탐지기가 정상 영상(매장이동 train, 유기 없음)에서 얼마나 물건을 찾는지 센다.

탐지기 단독 진단이다: 여기 나온 탐지가 곧 유기 경보는 아니다(경보는 사람·행동 조건과 결합해야 함).
- 프레임당 탐지 수와 탐지가 있는 프레임 비율
- 같은 자리(IoU >= OBJECT_FA_SAME_IOU)에 NEW_OBJECT_PERSIST_SEC 이상 계속 있는 탐지(정지 물건 후보) 개수/분
공식 validation은 쓰지 않는다. 영상은 폴더 전체에서 일정 간격으로 OBJECT_FA_VIDEOS개를 고른다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from config.config import (AIHUB_ORGANIZED_ROOT, NEW_OBJECT_PERSIST_SEC, OBJECT_FA_CONFS, OBJECT_FA_SAME_IOU,
                           OBJECT_FA_VIDEOS, OBJECT_RECALL_CONF, OBJECT_RECALL_IMGSZ)
from models.new_object_detector import _iou

BATCH = 8


def detect_video(model, path: str):
    """프레임별 [(박스, 점수)] 와 fps."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
    frames, out = [], []

    def flush():
        for r in model.predict(frames, conf=OBJECT_RECALL_CONF, imgsz=OBJECT_RECALL_IMGSZ, verbose=False):
            out.append([(b, float(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())])
        frames.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) == BATCH:
            flush()
    if frames:
        flush()
    cap.release()
    return out, fps


def count_static(per_frame, conf: float, fps: float) -> int:
    """같은 자리 박스가 이어진 길이가 PERSIST_SEC 이상인 묶음 수(한 프레임 끊김은 허용)."""
    need = round(NEW_OBJECT_PERSIST_SEC * fps)
    chains = []  # [마지막 박스, 마지막 프레임, 길이]
    done = 0
    for i, dets in enumerate(per_frame):
        boxes = [b for b, c in dets if c >= conf]
        used = set()
        for chain in chains:
            match = next((j for j, b in enumerate(boxes) if j not in used and _iou(chain[0], b) >= OBJECT_FA_SAME_IOU), None)
            if match is not None:
                used.add(match)
                chain[0], chain[1], chain[2] = boxes[match], i, chain[2] + 1
        alive = []
        for chain in chains:
            if i - chain[1] <= 1:
                alive.append(chain)
            elif chain[2] >= need:
                done += 1
        chains = alive + [[b, i, 1] for j, b in enumerate(boxes) if j not in used]
    return done + sum(1 for c in chains if c[2] >= need)


def summarize(per_frame, fps: float) -> dict:
    minutes = len(per_frame) / fps / 60
    out = {"minutes": round(minutes, 2)}
    for conf in OBJECT_FA_CONFS:
        counts = [sum(1 for _, c in dets if c >= conf) for dets in per_frame]
        out[str(conf)] = {"dets_per_frame": round(sum(counts) / len(counts), 3),
                          "frames_with_det": round(sum(1 for n in counts if n) / len(counts), 3),
                          "static_per_min": round(count_static(per_frame, conf, fps) / minutes, 2)}
    return out


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--folder", default="매장이동")
    parser.add_argument("--tag", default="ft")
    args = parser.parse_args()
    videos = sorted((Path(AIHUB_ORGANIZED_ROOT) / args.folder / "train" / "videos").glob("*.mp4"))
    step = max(len(videos) // OBJECT_FA_VIDEOS, 1)
    picked = videos[::step][:OBJECT_FA_VIDEOS]
    model = YOLO(args.weights)
    rows = []
    for path in picked:
        per_frame, fps = detect_video(model, str(path))
        rows.append({"video": path.name, **summarize(per_frame, fps)})
        print(rows[-1]["video"], rows[-1]["minutes"], {c: rows[-1][str(c)]["static_per_min"] for c in OBJECT_FA_CONFS}, flush=True)
    total = sum(r["minutes"] for r in rows)
    agg = {str(c): {k: round(sum(r[str(c)][k] * r["minutes"] for r in rows) / total, 3)
                    for k in ("dets_per_frame", "frames_with_det", "static_per_min")} for c in OBJECT_FA_CONFS}
    out = Path("output") / f"object_fa_{args.tag}_{args.folder}.json"
    out.write_text(json.dumps({"minutes": total, "videos": len(rows), "agg": agg, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("합계", round(total, 1), "분", json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
