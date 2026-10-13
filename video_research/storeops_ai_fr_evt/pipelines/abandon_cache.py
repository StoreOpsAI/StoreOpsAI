"""유기 영상의 포즈 추적과 물건 탐지 원본을 영상별 npz로 저장한다(모델을 다시 돌리지 않고 관계 특징을 여러 방식으로 만들 수 있게).

저장: 사람 추적(tid, frame, 박스, 손목 두 점 x·y·신뢰도), 물건 탐지(frame, 박스, 점수), 창(tid, start)과 라벨.
창·라벨은 abandon_features와 같은 규칙. 탐지 점수 하한은 CACHE_DET_CONF(낮게): 점수는 헤드의 입력으로 쓴다.
--weights에 {fold}를 쓰면 fold별(교차 학습) 탐지기를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from config.config import ABANDON_CACHE_DIR, CACHE_DET_CONF, OBJECT_RECALL_IMGSZ, POSE_WEIGHTS
from pipelines.abandon_features import read_frames
from pipelines.abandon_manifest import ABANDON_MANIFEST_PATH
from pipelines.abandon_windows import label_windows, make_windows, track_frames
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.object_dataset import read_track_boxes

BATCH = 8
WRISTS = (9, 10)


def detect_all(model, frames):
    rows = []
    for i in range(0, len(frames), BATCH):
        for k, r in enumerate(model.predict(frames[i:i + BATCH], conf=CACHE_DET_CONF, imgsz=OBJECT_RECALL_IMGSZ, verbose=False)):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append((i + k, *b, c))
    return np.array(rows, np.float32).reshape(-1, 6)


def process_clip(clip, pose_model, det_model, out_path: Path) -> int:
    frames = read_frames(clip["video"])
    tracks = track_frames(pose_model, frames)
    windows = make_windows(tracks, len(frames))
    xml = Path(clip["video"]).parent.parent / "labels" / f"{clip['clip']}.xml"
    labels = label_windows(windows, tracks, clip["events"], read_track_boxes(xml)) if windows else np.zeros(0, np.int8)
    trk = [(t, f, *box, *kpt[list(WRISTS)].reshape(-1)) for t, seen in tracks.items() for f, (box, kpt) in seen.items()]
    np.savez_compressed(out_path, n=len(windows), trk=np.array(trk, np.float32).reshape(-1, 12), det=detect_all(det_model, frames),
                        win_tid=np.array([t for t, _ in windows], np.int32), win_start=np.array([s for _, s in windows], np.int32),
                        label=labels, n_frames=len(frames), clip=clip["clip"], fold=clip["fold"])
    return len(windows)


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", default=ABANDON_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    manifest = json.loads(Path(ABANDON_MANIFEST_PATH).read_text(encoding="utf-8"))
    clips = [c for c in manifest["train"] if c["first_pass"] and not c["quarantine"]]
    clips = clips[:args.limit] if args.limit else clips
    index, total = (int(v) for v in args.shard.split("/"))
    clips = clips[index::total]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_model, detectors = YOLO(str(APP_ROOT / POSE_WEIGHTS)), {}
    for i, clip in enumerate(clips):
        out = out_dir / f"{clip['clip']}.npz"
        if out.is_file():
            continue
        path = args.weights.format(fold=clip["fold"])
        detectors.setdefault(path, YOLO(path))
        tmp = out.with_name(out.stem + ".tmp.npz")
        n = process_clip(clip, pose_model, detectors[path], tmp)
        os.replace(tmp, out)
        print(f"[{i + 1}/{len(clips)}] {clip['clip']} 창 {n}개", flush=True)


if __name__ == "__main__":
    main()
