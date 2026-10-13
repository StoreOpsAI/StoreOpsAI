from __future__ import annotations

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parent
CONFIG_ROOT = APP_ROOT / "output" / "ultralytics_config"
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))


def _attributes(shape):
    return {item.get("name"): (item.text or "").strip() for item in shape.findall("attribute")}


def _visible_boxes(track):
    boxes = []
    for shape in track.findall("box"):
        if shape.get("outside") == "1":
            continue
        boxes.append(
            {
                "frame": int(shape.get("frame")),
                "box": tuple(int(float(shape.get(k))) for k in ("xtl", "ytl", "xbr", "ybr")),
                "attributes": _attributes(shape),
            }
        )
    return boxes


def parse_annotation(xml_path, action="abandon"):
    """<action>_start/<action>_end 라벨 쌍을 사건 목록으로. xml_path는 경로 또는 파일 객체(zip 안 파일)."""
    start_label, end_label = f"{action}_start", f"{action}_end"
    tracks = {start_label: [], end_label: []}
    for track in ET.parse(xml_path).findall(".//track"):
        if track.get("label") in tracks:
            boxes = _visible_boxes(track)
            if boxes:
                tracks[track.get("label")].append(boxes)

    starts = sorted((boxes[0] for boxes in tracks[start_label]), key=lambda item: item["frame"])
    ends = sorted((boxes[0] for boxes in tracks[end_label]), key=lambda item: item["frame"])
    if len(starts) != len(ends):
        raise ValueError(f"시작/종료 라벨 수가 다릅니다: {xml_path} ({len(starts)} != {len(ends)})")

    events = []
    for index, (start, end) in enumerate(zip(starts, ends), 1):
        if start["frame"] > end["frame"]:
            raise ValueError(f"{action} 시작이 종료보다 늦습니다: {xml_path} event={index}")
        events.append(
            {
                "event_index": index,
                "start_frame": start["frame"],
                "end_frame": end["frame"],
                "owner_id": start["attributes"].get("ID", ""),
                "start_box": start["box"],
                "end_box": end["box"],
            }
        )
    return events


def parse_object_boxes(xml_path):
    """라벨의 유기 물건(object_abandon) 박스를 프레임별로 모은다. 시각화 확인용."""
    by_frame = {}
    for track in ET.parse(xml_path).findall(".//track"):
        if track.get("label") == "object_abandon":
            for item in _visible_boxes(track):
                by_frame.setdefault(item["frame"], []).append(item["box"])
    return by_frame


def build_manifest(dataset_root, output_path):
    rows = []
    for split in ("AI_data", "AI_validation"):
        # 유기/AI_train/AI_data 처럼 한 단계 더 중첩된 배치도 허용
        candidates = [dataset_root / split, *dataset_root.glob(f"*/{split}")]
        split_dir = next((p for p in candidates if (p / "Labels").is_dir()), None)
        if split_dir is None:
            raise FileNotFoundError(f"{split}/Labels 폴더를 찾을 수 없습니다: {dataset_root}")
        label_dir = split_dir / "Labels"
        video_dir = split_dir / "videos"
        for xml_path in sorted(label_dir.glob("*.xml")):
            video_path = video_dir / f"{xml_path.stem}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"라벨과 짝인 영상이 없습니다: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            events = parse_annotation(xml_path)
            for event in events:
                start, end = event["start_frame"], event["end_frame"]
                rows.append(
                    {
                        "split": "train" if split == "AI_data" else "validation",
                        "event_id": f"{xml_path.stem}_{event['event_index']:02d}",
                        "video": str(video_path.resolve()),
                        "annotation": str(xml_path.resolve()),
                        "fps": fps,
                        "start_frame": start,
                        "end_frame": end,
                        "duration_sec": (end - start + 1) / fps if fps else 0.0,
                        "label": "유기",
                        "owner_id": event["owner_id"],
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def smoke_test(video_path, xml_path, output_dir, detect_model):
    from ultralytics import YOLO

    import torch

    from config.config import TRACKER_CONFIG
    from models.yolo_detector import tracker_config_for_fps

    events = parse_annotation(xml_path)
    object_boxes = parse_object_boxes(xml_path)
    detector = YOLO(str(detect_model))
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{video_path.stem}_botsort_action.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"결과 영상 생성 실패: {output_video}")

    owner_matches = []
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tracked = detector.track(
                frame, persist=True, tracker=tracker_config_for_fps(TRACKER_CONFIG, fps), classes=[0], conf=0.25,
                device=device, verbose=False,
            )[0]
            person_boxes = []
            if tracked.boxes is not None:
                ids = tracked.boxes.id
                for index, box in enumerate(tracked.boxes.xyxy.cpu().tolist()):
                    coords = tuple(map(int, box))
                    track_id = int(ids[index].item()) if ids is not None else -1
                    person_boxes.append((track_id, coords))
                    cv2.rectangle(frame, coords[:2], coords[2:], (0, 220, 0), 2)
                    cv2.putText(frame, f"person {track_id}", (coords[0], max(24, coords[1] - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

            for coords in object_boxes.get(frame_no, []):
                cv2.rectangle(frame, coords[:2], coords[2:], (0, 0, 255), 2)
                cv2.putText(frame, "GT object", (coords[0], max(24, coords[1] - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            state = "NORMAL"
            for event in events:
                if frame_no == event["start_frame"]:
                    best = max(((_iou(event["start_box"], box), track_id) for track_id, box in person_boxes),
                               default=(0.0, -1))
                    owner_matches.append({"event_index": event["event_index"], "owner_id": event["owner_id"],
                                          "track_id": best[1], "iou": round(best[0], 4)})
                if event["start_frame"] <= frame_no <= event["end_frame"]:
                    state = "GT_ABANDON_ACTION"

            cv2.putText(frame, f"frame={frame_no} state={state}", (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            writer.write(frame)
            frame_no += 1
    finally:
        cap.release()
        writer.release()

    summary = {
        "video": str(video_path.resolve()),
        "output_video": str(output_video.resolve()),
        "frames": frame_no,
        "events": len(events),
        "owner_matches": owner_matches,
        "note": "AI Hub XML의 유기 행동 구간과 사람 추적 연결을 확인한 스모크 테스트입니다.",
    }
    summary_path = output_video.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="AI Hub 유기 행동 구간 준비 및 BoT-SORT 스모크 테스트")
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "유기")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "aihub_abandonment")
    parser.add_argument("--smoke-video", type=Path)
    parser.add_argument("--detect-model", type=Path, default=APP_ROOT / "yolo11n.pt")
    args = parser.parse_args()

    manifest = args.output_dir / "manifest.csv"
    rows = build_manifest(args.dataset_root, manifest)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")}
    if counts != {"train": 47, "validation": 16}:
        raise RuntimeError(f"예상한 이벤트 수와 다릅니다: {counts}")
    print(json.dumps({"manifest": str(manifest.resolve()), "events": counts}, ensure_ascii=False))

    if args.smoke_video:
        video_path = args.smoke_video.resolve()
        split_dir = video_path.parent.parent
        xml_path = split_dir / "Labels" / f"{video_path.stem}.xml"
        summary = smoke_test(video_path, xml_path, args.output_dir, args.detect_model)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
