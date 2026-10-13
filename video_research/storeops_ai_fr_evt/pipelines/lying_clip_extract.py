"""전도 방식 A: 포즈 캐시의 사람 박스를 잘라 동결 CLIP 이미지 인코더 특징(512)을 클립마다 저장한다(output/lying_clip).

박스 순서는 output/lying_pose/<클립>.npz와 같다. 사진 만들기는 train_lying.square_crop(회색 정사각 패딩)을 그대로 쓰고,
CLIP 전처리는 정사각 사진이라 224×224 bicubic 리사이즈 + CLIP 평균·표준편차 정규화로 대신한다(HF CLIPProcessor와 같은 절차, 속도 때문에 GPU 배치로 처리).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch

from config.config import LYING_BACKBONE
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.lying_pose_extract import OUTPUT_DIR as POSE_DIR, select_clips
from pipelines.train_lying import square_crop

OUTPUT_DIR = APP_ROOT / "output" / "lying_clip"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
BATCH = 128


def prep(image, box):
    crop = cv2.cvtColor(cv2.resize(square_crop(image, box), (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return ((crop.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


@torch.no_grad()
def encode(model, tensors, device):
    out = []
    for i in range(0, len(tensors), BATCH):
        x = torch.from_numpy(np.stack(tensors[i:i + BATCH])).to(device).half()
        e = model.get_image_features(pixel_values=x)
        e = getattr(e, "pooler_output", e)
        out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def extract_clip(model, clip, device):
    pose = np.load(POSE_DIR / f"{clip['clip']}.npz")
    by_frame = {}
    for i, f in enumerate(pose["frame"]):
        by_frame.setdefault(int(f), []).append(i)
    tensors = [None] * len(pose["frame"])
    cap = cv2.VideoCapture(clip["video"])
    n = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        for i in by_frame.get(n, []):
            tensors[i] = prep(image, pose["box"][i])
        n += 1
    cap.release()
    assert all(t is not None for t in tensors), f"박스에 대응하는 프레임을 못 읽음: {clip['clip']}"
    return encode(model, tensors, device).astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="전도 방식 A: 사람 사진 CLIP 특징 캐시")
    parser.add_argument("--set", choices=["first_pass", "train", "validation", "normal"], default="first_pass")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from transformers import CLIPModel

    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(LYING_BACKBONE).to(device).half().eval()
    clips = select_clips(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), args.set)
    clips = clips[:args.limit] if args.limit else clips
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for i, clip in enumerate(clips, 1):
        path = OUTPUT_DIR / f"{clip['clip']}.npz"
        if path.is_file():
            continue
        np.savez(path, feat=extract_clip(model, clip, device))
        print(f"{i}/{len(clips)} {clip['clip']} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
