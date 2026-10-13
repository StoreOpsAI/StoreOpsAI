"""전도 영상 증강(train 전용, AGENTS.md 규칙 3): 웹캠 화질을 흉내 낸 열화 영상에서 포즈를 다시 뽑아 output/lying_pose_aug/에 저장한다.

클립마다 열화 정도를 시드 고정으로 하나 뽑아 클립 전체에 같은 값을 쓴다(웹캠은 한 대의 성질이 일정하다):
저해상도(축소 후 원래 크기로 복원), 블러, 밝기·대비, JPEG 압축. 크기는 원래대로 복원하므로 박스 좌표·라벨이 그대로 쓰인다.
대상: 전도 first_pass와 매장이동 정상의 train 클립. dev·val·test 클립에는 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib

import cv2
import numpy as np

from config.config import POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.lying_pose_extract import extract_clip, select_clips

AUG_DIR = APP_ROOT / "output" / "lying_pose_aug"


def clip_params(name):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return {"scale": rng.uniform(0.4, 1.0), "blur": rng.uniform(0.0, 1.5), "beta": rng.uniform(-40, 40),
            "alpha": rng.uniform(0.7, 1.3), "jpeg": int(rng.integers(30, 91))}


def degrade(image, p):
    h, w = image.shape[:2]
    out = cv2.resize(image, (max(int(w * p["scale"]), 2), max(int(h * p["scale"]), 2)), interpolation=cv2.INTER_AREA)
    if p["blur"] > 0.1:
        out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    out = cv2.convertScaleAbs(out, alpha=p["alpha"], beta=p["beta"])
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])[1], cv2.IMREAD_COLOR)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description="전도·정상 train 클립 영상 증강 포즈 캐시")
    parser.add_argument("--set", choices=["first_pass", "normal"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(APP_ROOT / POSE_WEIGHTS))
    clips = select_clips(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), args.set)
    clips = clips[:args.limit] if args.limit else clips
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for i, clip in enumerate(clips, 1):
        path = AUG_DIR / f"{clip['clip']}.npz"
        if path.is_file():
            continue
        params = clip_params(clip["clip"])
        data = extract_clip(model, clip, lambda image, p=params: degrade(image, p))
        assert data["n_frames"] == clip["frames"], f"프레임 수 불일치 {clip['clip']}"
        np.savez(path, **data)
        print(f"{i}/{len(clips)} {clip['clip']} 사람 {len(data['frame'])}건 {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
