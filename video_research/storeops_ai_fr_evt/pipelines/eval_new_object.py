"""새 물건 감지기(NewObjectDetector)를 AI Hub 유기 라벨로 평가한다.

공식 validation은 쓰지 않고 train(AI_data) 영상만 사용한다.
- 적중: 확정된 박스가 해당 영상의 라벨 물건(object_abandon) 박스와 IoU >= 0.2이고,
  유기 시작 ~ 종료 후 MATCH_AFTER_SEC 안에 확정된 경우
- 오탐: 어떤 유기 사건과도 맞지 않는 확정 박스
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from config.config import OBJECT_VERIFY_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT, parse_object_boxes
from models.new_object_detector import NewObjectDetector, _iou

MATCH_AFTER_SEC = 15


def load_person_boxes(video_path, cache_dir):
    cache = cache_dir / f"{Path(video_path).stem}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    from ultralytics import YOLO

    model = YOLO(str(APP_ROOT / "yolo11n.pt"))
    cap = cv2.VideoCapture(str(video_path))
    boxes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, classes=[0], conf=0.25, verbose=False)[0]
        boxes.append([list(map(int, box)) for box in result.boxes.xyxy.tolist()])
    cap.release()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(boxes), encoding="utf-8")
    return boxes


def detect_new_objects(video_path, person_boxes, kinds=("added",), verifier=None):
    """영상 전체에 새 물건 감지기를 돌려 (확정 목록, fps)를 돌려준다. 기본은 새로 놓인 물건(added)만.

    verifier(ObjectVerifier)를 주면 확정 순간의 프레임으로 물건 여부를 확인해 verified/verify_score/verify_label을 붙인다.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
    detector = NewObjectDetector(fps)
    detections = []
    for frame_no in range(len(person_boxes)):
        ok, frame = cap.read()
        if not ok:
            break
        confirmed = [item for item in detector.update(frame, person_boxes[frame_no]) if item["kind"] in kinds]
        for item in confirmed if verifier is not None else ():
            item["verified"], item["verify_score"], item["verify_label"] = verifier.verify(frame, item["box"])
        detections += confirmed
    cap.release()
    return detections, fps


def match_detections(detections, events, object_boxes, fps):
    """(맞힌 사건 번호 집합, 오탐 목록). 기준은 모듈 설명 참고."""
    gt_boxes = [box for boxes in object_boxes.values() for box in boxes]
    matched_events, false_positives = set(), []
    for detection in detections:
        hit = None
        for event in events:
            in_time = event["start_frame"] <= detection["confirmed_frame"] <= event["end_frame"] + MATCH_AFTER_SEC * fps
            if in_time and any(_iou(detection["box"], gt) >= 0.2 for gt in gt_boxes):
                hit = event["event_index"]
                break
        if hit is None:
            false_positives.append(detection)
        else:
            matched_events.add(hit)
    return matched_events, false_positives


def evaluate_video(video_path, events, object_boxes, person_boxes, verifier=None):
    detections, fps = detect_new_objects(video_path, person_boxes, verifier=verifier)
    if verifier is not None:
        detections = [item for item in detections if item["verified"]]
    matched_events, false_positives = match_detections(detections, events, object_boxes, fps)
    return {"events": len(events), "matched": len(matched_events), "false_positives": false_positives,
            "detections": len(detections), "seconds": len(person_boxes) / fps}


def main():
    parser = argparse.ArgumentParser(description="새 물건 감지기 평가 (train 영상만)")
    parser.add_argument("--manifest", type=Path,
                        default=APP_ROOT / "output" / "abandonment_training" / "source_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "new_object_eval")
    parser.add_argument("--verify", action="store_true", help="YOLO-World 물건 확인을 통과한 것만 센다")
    args = parser.parse_args()
    verifier = None
    if args.verify:
        from models.object_verifier import ObjectVerifier

        verifier = ObjectVerifier(APP_ROOT / OBJECT_VERIFY_WEIGHTS)

    from pipelines.aihub_abandonment import parse_annotation

    with args.manifest.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "train"]
    videos = defaultdict(list)
    for row in rows:
        videos[(row["video"], row["annotation"])].append(row)

    per_video = []
    for (video, annotation), _ in sorted(videos.items()):
        result = evaluate_video(video, parse_annotation(Path(annotation)), parse_object_boxes(Path(annotation)),
                                load_person_boxes(video, args.output_dir / "persons"), verifier)
        per_video.append({"video": Path(video).name, **result})

    events = sum(item["events"] for item in per_video)
    matched = sum(item["matched"] for item in per_video)
    false_positives = sum(len(item["false_positives"]) for item in per_video)
    minutes = sum(item["seconds"] for item in per_video) / 60
    summary = {"videos": len(per_video), "events": events, "recall": matched / events if events else 0.0,
               "false_positives": false_positives, "fp_per_video": false_positives / len(per_video),
               "fp_per_minute": false_positives / minutes if minutes else 0.0, "per_video": per_video}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "results_verified.json" if args.verify else "results.json"
    (args.output_dir / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
