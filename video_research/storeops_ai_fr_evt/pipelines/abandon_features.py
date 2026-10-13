"""유기 첫 패스 영상에서 사람별 시간 창의 포즈 시퀀스·S3D·X-CLIP 특징을 한 번에 뽑아 영상별 npz로 저장한다.

영상은 저장 뒤 지워도 되도록 분할 정보(영상 ID·세션·fold·공식 split)와 라벨(1/-1/0)을 같이 저장한다.
크롭: 창 안 사람 박스 합집합을 한 변당 ABANDON_CROP_PAD만큼 넓힌 정사각형(주변 바닥·물건이 보이게).
S3D 입력 16프레임, X-CLIP 8프레임(16프레임 중 격프레임)을 창에서 고르게 뽑는다. 백본은 얼려 둔다.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import ABANDON_CROP_PAD, ABANDON_FEATURE_DIR, ABANDON_WIN_FRAMES, POSE_WEIGHTS
from pipelines.abandon_manifest import ABANDON_MANIFEST_PATH
from pipelines.abandon_windows import label_windows, make_windows, track_frames
from pipelines.video_sampling import is_sample
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import load_xclip, xclip_embed
from pipelines.object_dataset import read_track_boxes
from pipelines.train_abandonment import MEAN, STD

CROP = 224
CLIP_FRAMES = 16
BATCH = 8


def window_region(seen, start, shape):
    """창 안 사람 박스 합집합을 PAD만큼 넓힌 정사각형 (x1, y1, x2, y2), 프레임 안으로 이동."""
    boxes = np.array([seen[f][0] for f in range(start, start + ABANDON_WIN_FRAMES) if f in seen])
    x1, y1, x2, y2 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    w, h = x2 - x1, y2 - y1
    side = min(max(w * (1 + 2 * ABANDON_CROP_PAD), h * (1 + 2 * ABANDON_CROP_PAD)), min(shape[:2]))
    cx = float(np.clip((x1 + x2) / 2, side / 2, shape[1] - side / 2))
    cy = float(np.clip((y1 + y2) / 2, side / 2, shape[0] - side / 2))
    return cx - side / 2, cy - side / 2, side


def crop_window(frames, start, region):
    x, y, side = region
    idx = np.linspace(0, ABANDON_WIN_FRAMES - 1, CLIP_FRAMES).round().astype(int) + start
    out = [cv2.resize(frames[i][int(y):int(y + side), int(x):int(x + side)], (CROP, CROP)) for i in idx]
    return np.stack(out)[..., ::-1].copy()  # BGR -> RGB


def pose_sequence(seen, start, region):
    x, y, side = region
    seq = np.zeros((ABANDON_WIN_FRAMES, 17, 3), np.float32)
    for k in range(ABANDON_WIN_FRAMES):
        if start + k in seen:
            kpt = seen[start + k][1]
            seq[k] = np.stack([(kpt[:, 0] - x) / side, (kpt[:, 1] - y) / side, kpt[:, 2]], axis=1)
    return seq


class Backbones:
    def __init__(self, device):
        from torchvision.models.video import S3D_Weights, s3d

        self.device = device
        self.s3d = s3d(weights=S3D_Weights.DEFAULT).features.to(device).eval()
        self.processor, self.xclip = load_xclip(device)

    @torch.no_grad()
    def embed(self, clips):
        """clips: (N, 16, 224, 224, 3) RGB uint8 -> (S3D (N,1024), X-CLIP (N,512))."""
        s3d_out, x_out = [], []
        for i in range(0, len(clips), BATCH):
            part = clips[i:i + BATCH]
            x = torch.from_numpy(part).permute(0, 4, 1, 2, 3).float() / 255
            x = ((x - MEAN) / STD).to(self.device)
            s3d_out.append(self.s3d(x).mean(dim=(2, 3, 4)).cpu().numpy())
            x_out.append(xclip_embed(self.processor, self.xclip, list(part)))
        return np.concatenate(s3d_out), np.concatenate(x_out)


def read_frames(path: str):
    cap, frames = cv2.VideoCapture(path), []
    native_fps, raw = cap.get(cv2.CAP_PROP_FPS), 0  # 3fps보다 빠른 영상(반품 10fps)은 정확히 3fps 표본만 쓴다
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if is_sample(raw - 1, native_fps):
            frames.append(frame)
    cap.release()
    return frames


def process_clip(clip, pose_model, backbones, out_path: Path) -> int:
    frames = read_frames(clip["video"])
    tracks = track_frames(pose_model, frames)
    windows = make_windows(tracks, len(frames))
    if not windows:
        np.savez_compressed(out_path, n=0)
        return 0
    xml = Path(clip["video"]).parent.parent / "labels" / f"{clip['clip']}.xml"
    labels = label_windows(windows, tracks, clip["events"], read_track_boxes(xml))
    regions = [window_region(tracks[t], s, frames[0].shape) for t, s in windows]
    crops = np.stack([crop_window(frames, s, r) for (t, s), r in zip(windows, regions)])
    s3d_feat, xclip_feat = backbones.embed(crops)
    pose = np.stack([pose_sequence(tracks[t], s, r) for (t, s), r in zip(windows, regions)])
    np.savez_compressed(out_path, n=len(windows), pose=pose, s3d=s3d_feat, xclip=xclip_feat, label=labels,
                        tid=np.array([t for t, _ in windows]), start=np.array([s for _, s in windows]),
                        clip=clip["clip"], session=clip["session"], fold=clip["fold"],
                        official_split=clip["official_split"], camera=clip["camera"])
    return len(windows)


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reverse", action="store_true", help="뒤에서부터 처리(작업자를 하나 더 띄울 때)")
    parser.add_argument("--mem-fraction", type=float, default=0, help="이 프로세스의 GPU 메모리 상한 비율(0이면 제한 없음)")
    args = parser.parse_args()
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    manifest = json.loads(Path(ABANDON_MANIFEST_PATH).read_text(encoding="utf-8"))
    clips = [c for c in manifest["train"] if c["first_pass"] and not c["quarantine"]]
    clips = clips[:args.limit] if args.limit else clips
    clips = clips[::-1] if args.reverse else clips
    out_dir = Path(ABANDON_FEATURE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_model = YOLO(str(APP_ROOT / POSE_WEIGHTS))
    backbones = Backbones("cuda")
    for i, clip in enumerate(clips):
        out = out_dir / f"{clip['clip']}.npz"
        if out.is_file():
            continue
        tmp = out.with_name(out.stem + ".tmp.npz")  # 두 작업자가 같은 영상을 만나도 반쯤 쓴 파일이 남지 않게
        n = process_clip(clip, pose_model, backbones, tmp)
        os.replace(tmp, out)
        print(f"[{i + 1}/{len(clips)}] {clip['clip']} 창 {n}개", flush=True)


if __name__ == "__main__":
    main()
