"""사람 검출·추적 시계열: 서비스와 같은 설정(yolo11n, ByteTrack, conf 0.25, 사람 클래스, 기본 입력 크기)으로 영상마다 3fps 표본의 사람 박스를 저장한다.

크롭 S3D 입력과 물건-사람 관계 특징의 재료다. 영상마다 추적기를 새로 시작하고(model.predictor 초기화) 번호는 영상 안에서만 의미가 있다.
출력: C:/data/features/persons/<영상>.npz = frame(표본 번호), tid(없으면 -1), box(x1,y1,x2,y2 원본 좌표), conf, n_frames
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT, CONF_THRESHOLD, PERSON_CLASS_ID, TRACKER_CONFIG
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.fullscene_windows import video_path
from pipelines.global_split import SPLIT_PATH
from pipelines.video_sampling import is_sample

OUT = Path(BEHAVIOR_FEATURE_ROOT) / "persons"


def track_clip(model, video: Path, device: str = "cuda") -> dict[str, np.ndarray]:
    model.predictor = None
    cap = cv2.VideoCapture(str(video))
    fps, raw, sample = cap.get(cv2.CAP_PROP_FPS), 0, 0
    frames, tids, boxes, confs = [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if not is_sample(raw - 1, fps):
            continue
        r = model.track(frame, tracker=TRACKER_CONFIG, classes=[PERSON_CLASS_ID], conf=CONF_THRESHOLD, persist=True, verbose=False, device=device)[0]
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
            for tid, b, c in zip(ids, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                frames.append(sample), tids.append(tid), boxes.append(b), confs.append(float(c))
        sample += 1
    cap.release()
    return {"frame": np.array(frames, np.int32), "tid": np.array(tids, np.int32), "box": np.array(boxes, np.float32).reshape(-1, 4),
            "conf": np.array(confs, np.float32), "n_frames": np.int32(sample)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--with-validation", action="store_true")
    parser.add_argument("--device", default="cuda", help="cuda 또는 cpu. GPU가 다른 학습에 쓰이는 동안은 cpu(yolo11n은 가볍다)")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    from ultralytics import YOLO

    model = YOLO(str(APP_ROOT / "yolo11n.pt"))
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))
            if (r["official_split"] == "train" and r["q_fall"] != "True") or (args.with_validation and r["official_split"] == "validation")]
    OUT.mkdir(parents=True, exist_ok=True)
    priority = {"유기": 0, "절도": 0, "반품": 1, "매장이동": 1, "전도": 2}  # 물건 증거가 필요한 행동을 먼저, 전도(S3D만으로 충분)는 마지막
    for k, r in enumerate(sorted(rows, key=lambda r: (priority.get(r["behavior"], 2), r["clip"]))[i::n], 1):
        target = OUT / f"{r['clip']}.npz"
        if target.is_file():
            continue
        video = next((p for p in (video_path(r["clip"], r["behavior"], s) for s in (r["official_split"], "train", "validation")) if p.is_file()), None)
        if video is not None:
            tmp = target.with_name(target.stem + ".tmp.npz")
            np.savez_compressed(tmp, **track_clip(model, video, args.device))
            tmp.replace(target)
        if k % 50 == 0:
            print(f"[{k}] {r['clip']}", flush=True)
    print("PERSON_TRACKS_DONE", flush=True)


if __name__ == "__main__":
    main()
