"""불 탐지 원시 캐시: organized/<행동>/<split>/videos 영상마다 FIRE_BASE_WEIGHTS 탐지(점수 0.02 이상 전부, FIRE_SAMPLE_FPS 간격)를 output/fire_cache/에 저장한다.

eval_fire.load_fire_detections를 FIRE_IMGSZ(작은 불용 큰 입력)로 호출한다. 점수 하한·지속 시간은 여기서 정하지 않고 search_fire_params가 OOF 후보에서 고른다.
공식 validation은 추출하지 않는다(--splits train만).
"""
from __future__ import annotations

import sys
from pathlib import Path

from config.config import AIHUB_ORGANIZED_ROOT, FIRE_BASE_WEIGHTS, FIRE_IMGSZ
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_fire import load_fire_detections

CACHE = APP_ROOT / "output" / "fire_cache"
SETS = ("방화", "절도", "전도", "반품", "매장이동")  # 방화=사건 영상, 나머지=불 없는 영상(오경보 측정)


def main() -> None:
    from ultralytics import YOLO

    model = YOLO(str(APP_ROOT / FIRE_BASE_WEIGHTS))
    for behavior in SETS:
        videos = sorted((Path(AIHUB_ORGANIZED_ROOT) / behavior / "train" / "videos").glob("*.mp4"))
        for i, video in enumerate(videos, 1):
            load_fire_detections(video, model, CACHE, imgsz=FIRE_IMGSZ)
            if i % 20 == 0 or i == len(videos):
                print(f"{behavior} {i}/{len(videos)}", flush=True)
    print("FIRE_CACHE_DONE", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
