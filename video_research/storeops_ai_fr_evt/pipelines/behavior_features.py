"""행동 공통: 정리된 영상(organized/<행동>/<split>/videos + labels)에서 사람별 시간 창의 포즈·S3D·X-CLIP 특징과 라벨을 영상별 npz로 저장한다.

저장 필드는 abandon_features와 같다(+ behavior, site, group). 영상은 저장 뒤 지워도 된다(영상 ID·세션·fold·split 포함).
session = (장면, 사이트, 날짜), fold = session 해시 % BEHAVIOR_FOLDS (영상이 사라져도 같은 값). 같은 장면의 다중 카메라는 session이 같다.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np

from config.config import AIHUB_ORGANIZED_ROOT, BEHAVIOR_FEATURE_ROOT, BEHAVIOR_FOLDS, POSE_WEIGHTS
from pipelines.abandon_features import Backbones, crop_window, pose_sequence, read_frames, window_region
from pipelines.abandon_windows import make_windows, track_frames
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.behavior_labels import PREFIX, label_windows_actors, parse_label

NAME = re.compile(r"C_\d_\d+_(\d+)_BU_([A-Z]+)_(\d\d-\d\d)_\d\d-\d\d-\d\d_(C[A-F]|[a-f])_")


def clip_meta(video: Path, split: str, behavior: str) -> dict:
    m = NAME.match(video.name)
    if not m:
        raise ValueError(f"파일 이름 형식이 다름: {video.name}")
    session = "_".join(m.group(1, 2, 3))
    fold = int(hashlib.md5(session.encode()).hexdigest(), 16) % BEHAVIOR_FOLDS
    return {"clip": video.stem, "session": session, "site": m.group(2), "camera": m.group(4), "fold": fold,
            "official_split": split, "behavior": behavior}


def process_clip(video: Path, xml: Path, meta: dict, pose_model, backbones, out_path: Path) -> int:
    events, actors = parse_label(xml, meta["behavior"]) if xml else ([], {})  # 정상 풀(반품 등)은 라벨 없이 모든 창 0
    frames = read_frames(str(video))
    tracks = track_frames(pose_model, frames)
    windows = make_windows(tracks, len(frames))
    if not windows:
        np.savez_compressed(out_path, n=0, **meta)
        return 0
    labels = label_windows_actors(windows, tracks, events, actors)
    regions = [window_region(tracks[t], s, frames[0].shape) for t, s in windows]
    crops = np.stack([crop_window(frames, s, r) for (t, s), r in zip(windows, regions)])
    s3d_feat, xclip_feat = backbones.embed(crops)
    pose = np.stack([pose_sequence(tracks[t], s, r) for (t, s), r in zip(windows, regions)])
    np.savez_compressed(out_path, n=len(windows), pose=pose, s3d=s3d_feat, xclip=xclip_feat, label=labels,
                        tid=np.array([t for t, _ in windows]), start=np.array([s for _, s in windows]), **meta)
    return len(windows)


def main():
    import torch
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, choices=sorted(PREFIX))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--only", type=Path, help="클립 이름(확장자 없음)을 한 줄씩 적은 파일. 있으면 그 클립만 처리")
    parser.add_argument("--mem-fraction", type=float, default=0)
    args = parser.parse_args()
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    src = Path(AIHUB_ORGANIZED_ROOT) / args.behavior / args.split
    out_dir = Path(BEHAVIOR_FEATURE_ROOT) / f"{PREFIX[args.behavior]}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted((src / "videos").glob("*.mp4"))
    if args.only:
        wanted = set(args.only.read_text(encoding="utf-8").split())
        videos = [v for v in videos if v.stem in wanted]
    videos = videos[::-1] if args.reverse else videos
    pose_model = YOLO(str(APP_ROOT / POSE_WEIGHTS))
    backbones = Backbones("cuda")
    for i, video in enumerate(videos):
        out = out_dir / f"{video.stem}.npz"
        xml = src / "labels" / f"{video.stem}.xml"
        normal = PREFIX[args.behavior] == "normal"
        if out.is_file() or not (normal or xml.is_file()):
            continue
        tmp = out.with_name(out.stem + ".tmp.npz")
        n = process_clip(video, None if normal else xml, clip_meta(video, args.split, args.behavior), pose_model, backbones, tmp)
        os.replace(tmp, out)
        print(f"[{i + 1}/{len(videos)}] {video.stem} 창 {n}개", flush=True)


if __name__ == "__main__":
    main()
