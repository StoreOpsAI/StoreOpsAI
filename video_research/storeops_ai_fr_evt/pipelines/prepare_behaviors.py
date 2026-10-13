"""유기·절도·파손·방화 통합 클립 준비 (행동별 판정기 교차검증용).

- 공식 validation 영상은 split=test로 표시만 하고 교차검증·학습에 쓰지 않는다(최종 동결 시험용).
- 양성: 사건 구간을 4초 창으로 나눠 사건당 최대 MAX_WINDOWS개. 사람 박스는 라벨 시작·끝 박스 보간 기준.
- 정상: 같은 영상에서 사건 전후 2초 여유를 둔 구간, 영상당 NORMALS_PER_VIDEO개.
- group = 촬영 세션. 행동이 달라도 같은 세션이 나오면 멈춘다(다른 split에 섞이는 것 방지).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.prepare_purchase_negatives import sample_indices_at
from pipelines.train_abandonment import _interpolate_box, _normal_windows, _read_crop_clip, session_key

BEHAVIORS = {"유기": "abandon", "절도": "theft", "파손": "broken", "방화": "fire"}
MAX_WINDOWS = 3
NORMALS_PER_VIDEO = 2
WINDOW_SEC = 4


def list_videos(folder, action):
    """(xml, mp4, split) 목록. 경로에 'val'이 있으면 공식 validation(test). 영상이 없는 라벨은 건너뛰고 센다."""
    videos = {path.stem: path for path in folder.rglob("*.mp4")}
    items, missing = [], 0
    for xml in sorted(folder.rglob("*.xml")):
        if xml.stem not in videos:
            missing += 1
            continue
        split = "test" if "val" in str(xml.relative_to(folder)).lower() else "train"
        items.append((xml, videos[xml.stem], split))
    return items, missing


def event_windows(event, fps):
    length = round(WINDOW_SEC * fps)
    starts = range(event["start_frame"], event["end_frame"] + 1, length)
    return [(start, min(start + length - 1, event["end_frame"])) for start in starts][:MAX_WINDOWS]


def _save(samples_dir, sample_id, clip):
    path = samples_dir / f"{sample_id}.npz"
    np.savez_compressed(path, frames=clip)
    return str(path)


def clips_for_video(detector, behavior, xml, video, split, samples_dir):
    events = parse_annotation(xml, BEHAVIORS[behavior])
    cap = cv2.VideoCapture(str(video))
    frame_count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 3.0
    cap.release()
    base = {"split": split, "group": session_key(video), "video": str(video), "source": behavior}
    rows = []
    for event in events:
        span = max(1, event["end_frame"] - event["start_frame"])
        for number, (start, end) in enumerate(event_windows(event, fps), 1):
            indices = sample_indices_at(start, end, fps)
            references = [_interpolate_box(event["start_box"], event["end_box"], (frame - event["start_frame"]) / span)
                          for frame in indices]
            clip = _read_crop_clip(detector, video, indices, references)
            if clip is not None:
                sample_id = f"{video.stem}_{event['event_index']:02d}_{number}_pos"
                rows.append({**base, "sample_id": sample_id, "label": behavior,
                             "path": _save(samples_dir, sample_id, clip)})
    normals = 0
    for start, end in _normal_windows(frame_count, fps, events):
        clip = _read_crop_clip(detector, video, sample_indices_at(start, end, fps))
        if clip is None:
            continue
        normals += 1
        sample_id = f"{video.stem}_{normals:02d}_neg"
        rows.append({**base, "sample_id": sample_id, "label": "정상", "path": _save(samples_dir, sample_id, clip)})
        if normals == NORMALS_PER_VIDEO:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description="행동 통합 클립 준비 (공식 validation은 test로만 표시)")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "behavior_training")
    args = parser.parse_args()
    from ultralytics import YOLO

    detector = YOLO(str(APP_ROOT / "yolo11n.pt"))
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows, audit = [], {}
    for behavior, action in BEHAVIORS.items():
        items, missing = list_videos(WORKSPACE_ROOT / behavior, action)
        before = len(rows)
        for xml, video, split in items:
            rows += clips_for_video(detector, behavior, xml, video, split, samples_dir)
        audit[behavior] = {"videos": len(items), "labels_without_video": missing,
                           "clips": {split: {label: sum(r["split"] == split and r["label"] == label for r in rows[before:])
                                             for label in (behavior, "정상")} for split in ("train", "test")}}
        print(behavior, json.dumps(audit[behavior], ensure_ascii=False), flush=True)

    split_of_group = {}
    for row in rows:
        if split_of_group.setdefault(row["group"], row["split"]) != row["split"]:
            raise RuntimeError(f"세션이 train과 test에 동시에 있습니다: {row['group']}")
    manifest = args.output_dir / "samples.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "split", "label", "group", "video", "path", "source"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{manifest}: {len(rows)} clips")


if __name__ == "__main__":
    main()
