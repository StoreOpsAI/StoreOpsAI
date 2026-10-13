"""유기 결합 평가: 행동 모델(X-CLIP 영상 임베딩 + 선형 끝단) AND 새 물건 감지기.

- AI_train 영상만 사용한다. 공식 validation은 쓰지 않는다.
- 행동 끝단은 cv_abandonment와 같은 세션 fold로 학습하고, 각 영상에는 그 영상 세션을 뺀 fold의 끝단만 적용한다.
- 결합 규칙: 행동 확률 >= 0.5인 사람 창이 있고, 그 사람 박스 근처(FUSION_NEAR_RATIO)에서
  창 시작 ~ 창 끝 + FUSION_AFTER_SEC 사이에 새 물건이 생기기 시작했으면 유기.
- 결합 결과는 사건(같은 사람의 연속 양성 구간) 단위로 센다. 사건의 물건 박스 중 하나라도
  eval_new_object.match_detections 기준으로 맞으면 적중, 아니면 오경보 1건.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import (
    BEHAVIOR_STRIDE_SEC, BEHAVIOR_WINDOW_SEC, CONF_THRESHOLD, FUSION_AFTER_SEC, FUSION_NEAR_RATIO, OBJECT_VERIFY_WEIGHTS,
    TRACKER_CONFIG,
)
from pipelines.aihub_abandonment import APP_ROOT, _iou, parse_annotation, parse_object_boxes
from pipelines.compare_backbones import _fit_linear, load_xclip, xclip_embed
from pipelines.cv_abandonment import assign_folds
from pipelines.eval_new_object import detect_new_objects, load_person_boxes, match_detections
from pipelines.train_abandonment import _crop, _sample_indices, session_key

MIN_COVERAGE = 0.75  # 창 안에서 그 사람이 실제로 검출된 프레임 비율 하한


def load_tracks(video_path, cache_dir):
    """프레임별 [[track_id, x1, y1, x2, y2], ...]. BoT-SORT 결과를 캐시한다."""
    cache = cache_dir / f"{Path(video_path).stem}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    from models.yolo_detector import YOLODetector

    detector = YOLODetector(APP_ROOT / "yolo11n.pt", TRACKER_CONFIG, CONF_THRESHOLD)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
    tracks = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detector.track(frame, fps).boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else []
        tracks.append([[track_id, *map(int, box)] for track_id, box in zip(ids, boxes.xyxy.tolist())])
    cap.release()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(tracks), encoding="utf-8")
    return tracks


def build_windows(tracks, fps):
    """사람별 창 목록. 창 안의 빈 프레임은 그 사람의 가장 가까운 검출 박스로 채운다."""
    by_track = defaultdict(dict)
    for frame_no, items in enumerate(tracks):
        for track_id, *box in items:
            by_track[track_id][frame_no] = tuple(box)
    length, stride = max(1, round(BEHAVIOR_WINDOW_SEC * fps)), max(1, round(BEHAVIOR_STRIDE_SEC * fps))
    windows = []
    for track_id, boxes in by_track.items():
        seen = sorted(boxes)
        for start in range(seen[0], seen[-1] - length + 2, stride):
            frames = range(start, start + length)
            if sum(frame in boxes for frame in frames) < MIN_COVERAGE * length:
                continue
            filled = {frame: boxes[min(seen, key=lambda value: abs(value - frame))] for frame in frames}
            windows.append({"track": track_id, "start": start, "end": start + length - 1, "boxes": filled})
    return windows


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def window_clips(video_path, windows):
    """학습 때와 같은 방식(16프레임, 사람 박스 확장 crop)으로 창마다 클립을 만든다."""
    needed = defaultdict(set)  # frame -> {(track, box)}
    for window in windows:
        window["indices"] = _sample_indices(window["start"], window["end"])
        for frame_no in window["indices"]:
            needed[frame_no].add((window["track"], window["boxes"][frame_no]))
    crops = {}
    cap = cv2.VideoCapture(str(video_path))
    for frame_no in range(max(needed) + 1 if needed else 0):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"프레임을 읽지 못했습니다: {video_path} #{frame_no}")
        for track_id, (x1, y1, x2, y2) in needed.get(frame_no, ()):
            w, h = x2 - x1, y2 - y1
            crops[(track_id, frame_no)] = _crop(frame, (x1 - w * 0.35, y1 - h * 0.20, x2 + w * 0.35, y2 + h * 0.20))
    cap.release()
    return [np.stack([crops[(window["track"], frame_no)] for frame_no in window["indices"]]) for window in windows]


def fuse(windows, probabilities, detections, fps):
    """행동 양성 창과 시간·위치가 맞는 새 물건을 사건 단위로 묶는다.

    같은 사람의 겹치는 양성 창은 한 구간(사건)으로 보고, 그 구간에 연결된 물건 박스들을 한 사건에 모은다.
    """
    positive = sorted((window for window, probability in zip(windows, probabilities) if probability >= 0.5),
                      key=lambda window: (window["track"], window["start"]))
    run_of, runs = [], []
    for window in positive:
        if not runs or runs[-1]["track"] != window["track"] or window["start"] > runs[-1]["end"]:
            runs.append({"track": window["track"], "end": window["end"], "detections": []})
        runs[-1]["end"] = window["end"]
        run_of.append(runs[-1])
    for detection in detections:
        cx, cy = (detection["box"][0] + detection["box"][2]) / 2, (detection["box"][1] + detection["box"][3]) / 2
        for window, run in zip(positive, run_of):
            if not window["start"] <= detection["first_frame"] <= window["end"] + FUSION_AFTER_SEC * fps:
                continue
            x1, y1, x2, y2 = _union(list(window["boxes"].values()))
            dx, dy = (x2 - x1) * FUSION_NEAR_RATIO, (y2 - y1) * FUSION_NEAR_RATIO
            if x1 - dx <= cx <= x2 + dx and y1 - dy <= cy <= y2 + dy:
                run["detections"].append(detection)
                break
    return [run["detections"] for run in runs if run["detections"]]


def match_alarms(alarms, events, object_boxes, fps):
    """사건 하나라도 라벨 물건과 맞으면 적중, 아니면 오경보 1건. (맞힌 사건 번호 집합, 오경보 목록)."""
    matched, false_alarms = set(), []
    for alarm in alarms:
        hit, _ = match_detections(alarm, events, object_boxes, fps)
        matched |= hit
        if not hit:
            false_alarms.append(alarm)
    return matched, false_alarms


def behavior_only(windows, probabilities, events, fps):
    """행동 모델 단독: (맞힌 사건 수, 사건과 무관한 양성 구간 수). 같은 사람의 연속 양성 창은 한 구간으로 센다."""
    buffer = 2 * fps
    matched, false_runs, last_false = set(), 0, {}
    for window, probability in sorted(zip(windows, probabilities), key=lambda item: item[0]["start"]):
        if probability < 0.5:
            continue
        union = _union(list(window["boxes"].values()))
        hit = next((event["event_index"] for event in events
                    if window["start"] <= event["end_frame"] and window["end"] >= event["start_frame"]
                    and max(_iou(union, event["start_box"]), _iou(union, event["end_box"])) > 0), None)
        if hit is not None:
            matched.add(hit)
            continue
        near_event = any(window["start"] <= event["end_frame"] + buffer and window["end"] >= event["start_frame"] - buffer
                         for event in events)
        previous = last_false.get(window["track"])
        if not near_event and (previous is None or window["start"] > previous["end"]):
            false_runs += 1
        if not near_event:
            last_false[window["track"]] = window
    return len(matched), false_runs


def _load_features(manifest, feature_cache):
    """(train·dev 행, 특징, 라벨). 공식 validation(test) 행은 읽지 않는다."""
    with manifest.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] in ("train", "dev")]
    data = np.load(feature_cache)
    if data["ids"].tolist() != [row["sample_id"] for row in rows]:
        raise RuntimeError("특징 캐시와 samples.csv가 다릅니다. compare_backbones를 다시 실행하세요.")
    return rows, data["features"], np.array([int(row["label"]) for row in rows])


def full_head(manifest, feature_cache):
    """유기 train 전체로 학습한 선형 끝단. 유기 학습에 쓰지 않은 영상(다른 행동)에만 적용한다."""
    _, features, labels = _load_features(manifest, feature_cache)
    return _fit_linear(features, labels)


def fold_heads(manifest, feature_cache, folds):
    """세션 fold별 선형 끝단과 {세션: fold}."""
    rows, features, labels = _load_features(manifest, feature_cache)
    fold_of = assign_folds(rows, folds)
    fold_ids = np.array([fold_of[row["group"]] for row in rows])
    return [_fit_linear(features[fold_ids != fold], labels[fold_ids != fold]) for fold in range(folds)], fold_of


def main():
    parser = argparse.ArgumentParser(description="유기 결합 평가 (행동 AND 새 물건, train 영상만)")
    parser.add_argument("--source-manifest", type=Path,
                        default=APP_ROOT / "output" / "abandonment_training" / "source_manifest.csv")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "abandonment_training" / "samples.csv")
    parser.add_argument("--feature-cache", type=Path, default=APP_ROOT / "output" / "backbone_compare" / "xclip.npz")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "fusion_eval")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--verify", action="store_true", help="새 물건을 YOLO-World 물건 확인을 통과한 것만 쓴다")
    args = parser.parse_args()
    verifier = None
    if args.verify:
        from models.object_verifier import ObjectVerifier

        verifier = ObjectVerifier(APP_ROOT / OBJECT_VERIFY_WEIGHTS)

    heads, fold_of = fold_heads(args.manifest, args.feature_cache, args.folds)
    with args.source_manifest.open(encoding="utf-8-sig") as file:
        videos = sorted({(row["video"], row["annotation"]) for row in csv.DictReader(file) if row["split"] == "train"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_xclip(device)

    per_video = []
    for video, annotation in videos:
        fold = fold_of[session_key(video)]  # 없으면 KeyError: fold 밖 영상은 평가하지 않는다
        events, object_boxes = parse_annotation(Path(annotation)), parse_object_boxes(Path(annotation))
        persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons")
        detections, fps = detect_new_objects(video, persons, verifier=verifier)
        if verifier is not None:
            detections = [item for item in detections if item["verified"]]
        windows = build_windows(load_tracks(video, args.output_dir / "tracks"), fps)
        clips = window_clips(video, windows)
        features = [xclip_embed(processor, model, clips[i:i + 8]) for i in range(0, len(clips), 8)]
        probabilities = heads[fold](np.concatenate(features)).tolist() if features else []

        alarms = fuse(windows, probabilities, detections, fps)
        matched_fused, fp_fused = match_alarms(alarms, events, object_boxes, fps)
        matched_object, fp_object = match_detections(detections, events, object_boxes, fps)
        matched_behavior, fp_behavior = behavior_only(windows, probabilities, events, fps)
        per_video.append({
            "video": Path(video).name, "fold": fold, "events": len(events), "seconds": len(persons) / fps,
            "windows": len(windows), "positive_windows": sum(p >= 0.5 for p in probabilities),
            "object_only": {"matched": len(matched_object), "false_positives": len(fp_object)},
            "behavior_only": {"matched": matched_behavior, "false_positives": fp_behavior},
            "fused": {"matched": len(matched_fused), "false_positives": len(fp_fused),
                      "false_alarms": fp_fused},
        })
        print(json.dumps({key: per_video[-1][key] for key in ("video", "object_only", "behavior_only")}
                         | {"fused": {k: v for k, v in per_video[-1]["fused"].items() if k != "false_alarms"}},
                         ensure_ascii=False))
    write_summary(per_video, args.output_dir, "results_verified.json" if args.verify else "results.json")


def write_summary(per_video, output_dir, name="results.json"):
    events = sum(item["events"] for item in per_video)
    minutes = sum(item["seconds"] for item in per_video) / 60
    summary = {"videos": len(per_video), "events": events, "minutes": minutes}
    for name in ("object_only", "behavior_only", "fused"):
        matched = sum(item[name]["matched"] for item in per_video)
        false_positives = sum(item[name]["false_positives"] for item in per_video)
        summary[name] = {"recall": matched / events if events else 0.0, "false_positives": false_positives,
                         "fp_per_minute": false_positives / minutes if minutes else 0.0}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(json.dumps({**summary, "per_video": per_video}, ensure_ascii=False,
                                                        indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
