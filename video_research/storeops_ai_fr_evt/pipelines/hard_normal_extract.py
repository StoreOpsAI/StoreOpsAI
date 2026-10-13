"""구매행동 반품 train 영상의 포즈(YOLO11s-pose)를 캐시한다(output/lying_pose_return). 어려운 정상(앉기·숙이기) 후보 풀. 공식 validation은 쓰지 않는다.

전도 포즈 캐시(lying_pose_extract)와 같은 형식·같은 추출기. 이미 있는 캐시는 건너뛴다.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from ultralytics import YOLO

from config.config import AIHUB_ORGANIZED_ROOT, POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_pose_extract import extract_clip
from pathlib import Path

OUT = APP_ROOT / "output" / "lying_pose_return"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(APP_ROOT / POSE_WEIGHTS))
    videos = sorted(v for split in ("train", "validation") for v in (Path(AIHUB_ORGANIZED_ROOT) / "반품" / split / "videos").glob("*.mp4"))  # 101클립 전체가 개발용 풀(날짜 fold)
    start = time.time()
    for i, video in enumerate(videos, 1):
        path = OUT / f"{video.stem}.npz"
        if path.is_file():
            continue
        np.savez_compressed(path, **extract_clip(model, {"video": str(video)}))
        print(f"[{i}/{len(videos)}] {video.stem} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
