"""유기 영상에서 포즈 추적 + 물건 탐지를 돌려 사람별 시간 창의 사람-물건 연결 특징(object_relation)을 영상별 npz로 저장한다.

창 목록과 라벨은 abandon_features와 같은 규칙(같은 추적기·같은 창)으로 다시 만든다. 저장: tid, start, label, rel(N, 15).
--weights에 {fold}가 있으면 영상의 fold별 탐지기를 쓴다(교차 학습: 영상을 학습에 안 쓴 탐지기로 탐지해 특징이 낙관적이지 않게).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from config.config import ABANDON_REL_DIR, OBJECT_RECALL_IMGSZ, POSE_WEIGHTS, REL_DET_CONF
from pipelines.abandon_features import read_frames
from pipelines.abandon_manifest import ABANDON_MANIFEST_PATH
from pipelines.abandon_windows import label_windows, make_windows, track_frames
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.object_dataset import read_track_boxes
from pipelines.object_relation import window_features

BATCH = 8


def detect_frames(model, frames):
    out = []
    for i in range(0, len(frames), BATCH):
        for r in model.predict(frames[i:i + BATCH], conf=REL_DET_CONF, imgsz=OBJECT_RECALL_IMGSZ, verbose=False):
            out.append([(b, float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())])
    return out


def process_clip(clip, pose_model, det_model, out_path: Path) -> int:
    from pipelines.object_relation import link_objects

    frames = read_frames(clip["video"])
    tracks = track_frames(pose_model, frames)
    windows = make_windows(tracks, len(frames))
    if not windows:
        np.savez_compressed(out_path, n=0)
        return 0
    xml = Path(clip["video"]).parent.parent / "labels" / f"{clip['clip']}.xml"
    labels = label_windows(windows, tracks, clip["events"], read_track_boxes(xml))
    objects = link_objects(detect_frames(det_model, frames))
    rel = np.stack([window_features(tracks[t], objects, s, min(tracks[t]), max(tracks[t])) for t, s in windows])
    np.savez_compressed(out_path, n=len(windows), rel=rel, label=labels, tid=np.array([t for t, _ in windows]),
                        start=np.array([s for _, s in windows]), clip=clip["clip"], fold=clip["fold"])
    return len(windows)


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="탐지기 가중치. {fold}가 있으면 영상 fold별로 바꿔 쓴다")
    parser.add_argument("--out-dir", default=ABANDON_REL_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--shard", default="0/1", help="i/n: 영상을 n조각으로 나눠 i번째만(프로세스를 여럿 띄울 때)")
    args = parser.parse_args()
    manifest = json.loads(Path(ABANDON_MANIFEST_PATH).read_text(encoding="utf-8"))
    clips = [c for c in manifest["train"] if c["first_pass"] and not c["quarantine"]]
    clips = clips[:args.limit] if args.limit else clips
    clips = clips[::-1] if args.reverse else clips
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
        if path not in detectors:
            detectors[path] = YOLO(path)
        tmp = out.with_name(out.stem + ".tmp.npz")
        n = process_clip(clip, pose_model, detectors[path], tmp)
        os.replace(tmp, out)
        print(f"[{i + 1}/{len(clips)}] {clip['clip']} 창 {n}개", flush=True)


if __name__ == "__main__":
    main()
