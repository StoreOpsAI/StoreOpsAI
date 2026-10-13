"""AI Hub 구매행동(71549) 영상에서 어려운 정상 클립을 만든다 (기본: 반품 = 상품을 선반에 되돌려 놓음).

- 촬영 세션이 한쪽으로 몰리지 않게 세션(sha1 순서)을 돌아가며 영상 N개를 고르고, 영상마다 첫 동작 1건을 쓴다.
- 영상 zip은 풀지 않고 고른 영상만 꺼낸다.
- 유기 영상(3fps)과 시간 간격을 맞추려고 동작 구간을 3fps 격자로 뽑은 뒤 16프레임을 고른다.
- 기존 samples.csv 행 + 새 정상 행을 samples_purchase.csv로 쓴다(원본 manifest는 건드리지 않음).
  새 행은 split=train이고, 교차검증에서 세션 단위 fold로 다시 나뉜다. 기존 유기 세션과 겹치면 멈춘다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.train_abandonment import _interpolate_box, _read_crop_clip, session_key

SOURCE_FPS = 3.0  # 유기 학습 영상 fps
CLIP_SEC = 4


def sample_indices_at(start, end, fps, count=16):
    """start~end(최대 CLIP_SEC초)를 SOURCE_FPS 간격 격자로 뽑고 그중 count개를 고른다. 3fps 영상이면 기존 방식과 같다."""
    last = min(end, start + round(CLIP_SEC * fps) - 1)
    grid = np.unique(np.round(np.arange(start, last + 1e-6, fps / SOURCE_FPS)).astype(int))
    return grid[np.linspace(0, len(grid) - 1, count).round().astype(int)].tolist()


def select_videos(names, count):
    """세션을 돌아가며 영상 이름을 하나씩 뽑는다."""
    by_session = defaultdict(list)
    for name in sorted(names):
        by_session[session_key(name)].append(name)
    order = sorted(by_session, key=lambda value: hashlib.sha1(value.encode()).hexdigest())
    chosen = []
    while len(chosen) < count and any(by_session.values()):
        for session in order:
            if by_session[session] and len(chosen) < count:
                chosen.append(by_session[session].pop(0))
    return chosen


def extract_videos(video_zip, stems, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(video_zip) as archive:
        members = {Path(name).stem: name for name in archive.namelist() if name.lower().endswith(".mp4")}
        missing = [stem for stem in stems if stem not in members]
        if missing:
            raise FileNotFoundError(f"영상 zip에 없는 영상: {missing[:5]} 외 {max(0, len(missing) - 5)}개")
        paths = {}
        for stem in stems:
            target = dest / f"{stem}.mp4"
            if not target.is_file():
                with archive.open(members[stem]) as source, target.open("wb") as out:
                    while chunk := source.read(8 * 1024 * 1024):
                        out.write(chunk)
            paths[stem] = target
    return paths


def build_clips(label_zip, video_zip, action, count, output_dir, video_dir, normal_label):
    from ultralytics import YOLO

    detector = YOLO(str(APP_ROOT / "yolo11n.pt"))
    with zipfile.ZipFile(label_zip) as labels:
        xml_names = {Path(name).stem: name for name in labels.namelist() if name.endswith(".xml")}
        chosen = select_videos(xml_names, count)
        events = {stem: parse_annotation(labels.open(xml_names[stem]), action) for stem in chosen}
    videos = extract_videos(video_zip, chosen, video_dir)

    samples_dir = output_dir / "samples"
    rows = []
    for stem in chosen:
        if not events[stem]:
            continue
        event = events[stem][0]
        cap = cv2.VideoCapture(str(videos[stem]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps:
            raise RuntimeError(f"fps를 읽지 못했습니다: {videos[stem]}")
        indices = sample_indices_at(event["start_frame"], event["end_frame"], fps)
        span = max(1, event["end_frame"] - event["start_frame"])
        references = [_interpolate_box(event["start_box"], event["end_box"], (frame - event["start_frame"]) / span)
                      for frame in indices]
        clip = _read_crop_clip(detector, videos[stem], indices, references)
        if clip is None:
            print(f"건너뜀(사람 검출 실패): {stem}")
            continue
        sample_id = f"{stem}_{event['event_index']:02d}_{action}"
        path = samples_dir / f"{sample_id}.npz"
        np.savez_compressed(path, frames=clip)
        rows.append({"sample_id": sample_id, "split": "train", "label": normal_label, "group": session_key(stem),
                     "video": str(videos[stem]), "path": str(path), "source": f"purchase_{action}"})
    return rows


def main():
    purchase_root = WORKSPACE_ROOT / "구매행동"
    parser = argparse.ArgumentParser(description="구매행동 어려운 정상 클립 만들기 (공식 유기 validation 미사용)")
    parser.add_argument("--label-zip", type=Path,
                        default=next(purchase_root.rglob("VL_02.구매행동_05.반품.zip"), None))
    parser.add_argument("--video-zip", type=Path,
                        default=next(purchase_root.rglob("VS_02.구매행동_05.반품.zip"), None))
    parser.add_argument("--action", default="return")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "abandonment_training")
    parser.add_argument("--normal-label", default="0", help="정상 라벨 값 (유기 이진 manifest는 0, 행동 통합 manifest는 정상)")
    args = parser.parse_args()
    for name in ("label_zip", "video_zip"):
        if getattr(args, name) is None or not getattr(args, name).is_file():
            raise SystemExit(f"--{name.replace('_', '-')} 파일을 찾지 못했습니다.")

    with (args.output_dir / "samples.csv").open(encoding="utf-8-sig") as file:
        base = [{"source": "abandon", **row} for row in csv.DictReader(file)]  # source가 있으면 유지
    new = build_clips(args.label_zip, args.video_zip, args.action, args.count, args.output_dir,
                      purchase_root / "videos", args.normal_label)
    overlap = {row["group"] for row in base} & {row["group"] for row in new}
    if overlap:
        raise RuntimeError(f"구매행동 세션이 유기 세션과 겹칩니다: {sorted(overlap)}")
    manifest = args.output_dir / "samples_purchase.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(base[0].keys()))
        writer.writeheader()
        writer.writerows(base + new)
    print(f"{manifest}: 기존 {len(base)} + 구매행동 정상 {len(new)} (세션 {len({row['group'] for row in new})}개)")


if __name__ == "__main__":
    main()
