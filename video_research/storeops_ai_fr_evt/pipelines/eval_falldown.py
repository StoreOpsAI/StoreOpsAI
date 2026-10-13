"""전도 경보 평가: 누운 사람 분류기(train_lying) + 같은 자리 LYING_HOLD_SEC 이상(models/fall_detector).

어려운 음성 채굴(Shrivastava 외 CVPR 2016 응용): fold k마다 1차 분류기(fold k 제외 학습)가 fold k 밖 영상에서
누움으로 착각한 사람 사진을 '누움 아님'으로 더해 2차 분류기를 맞춘다. 1차·2차 모두 fold k로만 평가한다.
- 양성: 전도/train 클립(프레임 jpg, 3fps). 클립 안에서 경보가 한 번이라도 나면 적중
- 음성: 넘어짐이 없는 반품·유기·절도·파손·방화 train 영상. 3fps로 사람 박스를 판정, 경보는 모두 오경보
- 임계값 0.5, 유지 시간, 채굴 개수는 사전 값. 오경보 장면은 이미지로 남긴다. 폭행 영상·공식 validation 미사용
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from models.fall_detector import lying_alarms
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.eval_new_object import load_person_boxes
from pipelines.make_splits import FOLDS, load_splits
from pipelines.prepare_behaviors import list_videos
from pipelines.train_lying import _read, clip_features, load_clip, square_crop

SAMPLE_FPS = 3.0
HARD_PER_VIDEO = 20  # 영상 하나에서 모을 착각 사진 최대 수(점수 높은 순)
CACHE = APP_ROOT / "output" / "falldown_eval" / "features"


def cached_crops(key, make, processor, model):
    """{frames, boxes, features, thumbs}: 표본 프레임마다 사람 박스와 CLIP 특징. make() → [(프레임, [(박스, 사진)])]."""
    path = CACHE / f"{key}.npz"
    if path.is_file():
        return dict(np.load(path))
    rows = [(f, box, crop) for f, people in make() for box, crop in people]
    features = clip_features([c for *_, c in rows], processor, model) if rows else np.zeros((0, 512), np.float32)
    data = {"frames": np.array([f for f, *_ in rows], np.int64), "boxes": np.array([b for _, b, _ in rows], np.float32).reshape(-1, 4),
            "features": features, "thumbs": np.array([cv2.resize(c, (96, 96)) for *_, c in rows], np.uint8).reshape(-1, 96, 96, 3)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def video_people(video):
    persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons")
    cap = cv2.VideoCapture(str(video))
    step, frame_no = max(1, round((cap.get(cv2.CAP_PROP_FPS) or SAMPLE_FPS) / SAMPLE_FPS)), 0
    while True:
        ok, image = cap.read()
        if not ok or frame_no >= len(persons):
            break
        if frame_no % step == 0:
            yield frame_no, [(box, square_crop(image, box)) for box in persons[frame_no]]
        frame_no += 1
    cap.release()


def clip_people(folder, detector):
    for i, image in enumerate(map(_read, sorted(folder.glob("*.jpg")))):
        boxes = detector.predict(image, classes=[0], conf=0.25, verbose=False)[0].boxes.xyxy.tolist()
        yield i, [(box, square_crop(image, box)) for box in boxes]


def alarms_for(data, predict, fps, sample_frames):
    """분류 점수 >= 0.5인 박스로 프레임별 누움 목록을 만들어 경보를 낸다."""
    scores = predict(data["features"]) if len(data["features"]) else np.zeros(0)
    by_frame = {f: [] for f in sample_frames}
    for f, box, s in zip(data["frames"], data["boxes"], scores):
        if s >= 0.5:
            by_frame[int(f)].append(tuple(box))
    return lying_alarms(sorted(by_frame.items()), fps)


def video_paths():
    paths = {p.name: p for p in (WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4")}
    for name in ("유기", "절도", "파손", "방화"):
        paths.update({Path(mp4).name: mp4 for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"})
    return paths


def evaluate(predict, clips, videos):
    """(적중 클립 수, 오경보 목록, 음성 분)."""
    hits = sum(bool(alarms_for(d, predict, SAMPLE_FPS, range(int(d["frames"].max()) + 1 if len(d["frames"]) else 0)))
               for d in clips)
    false, minutes = [], 0.0
    for name, (data, fps, frames) in videos.items():
        minutes += len(frames) / SAMPLE_FPS / 60
        for frame_no, box in alarms_for(data, predict, fps, frames):
            i = next(j for j, (f, b) in enumerate(zip(data["frames"], data["boxes"])) if f == frame_no and tuple(b) == box)
            false.append((name, frame_no, data["thumbs"][i]))
    return hits, false, minutes


def main():
    parser = argparse.ArgumentParser(description="전도 경보 평가 (누운 사람 분류기 + 어려운 음성)")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "falldown_eval")
    args = parser.parse_args()
    from ultralytics import YOLO

    detector = YOLO(str(APP_ROOT / "yolo11n.pt"))
    processor, model = load_clip(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    splits, base = load_splits(), np.load(APP_ROOT / "output" / "lying_cv" / "features.npz")
    clips = {f.name: cached_crops(f.name, lambda f=f: clip_people(f, detector), processor, model)
             for f in sorted((WORKSPACE_ROOT / "전도" / "train").glob("*__fall"))}
    videos = {}
    for name, path in video_paths().items():
        data = cached_crops(Path(name).stem, lambda p=path: video_people(p), processor, model)
        fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) or SAMPLE_FPS
        step = max(1, round(fps / SAMPLE_FPS))
        length = len(load_person_boxes(path, APP_ROOT / "output" / "new_object_eval" / "persons"))
        videos[name] = (data, fps, list(range(0, length, step)))

    summary, tiles = {"stage1": [0, 0, 0.0], "stage2": [0, 0, 0.0], "clips": len(clips), "hard_negatives": 0}, []
    for k in range(FOLDS):
        x, y = base["features"][base["folds"] != k], base["labels"][base["folds"] != k]
        stage1 = _fit_linear(x, y)
        mined = []
        for name, (data, _, _) in videos.items():
            if splits[name]["fold"] != k and len(data["features"]):
                scores = stage1(data["features"])
                top = np.argsort(-scores)[:HARD_PER_VIDEO]
                mined += [data["features"][i] for i in top if scores[i] >= 0.5]
        stage2 = _fit_linear(np.concatenate([x, np.array(mined).reshape(-1, x.shape[1])]),
                             np.concatenate([y, np.zeros(len(mined), y.dtype)]))
        summary["hard_negatives"] += len(mined)
        fold_clips = [d for n, d in clips.items() if splits[n]["fold"] == k]
        fold_videos = {n: v for n, v in videos.items() if splits[n]["fold"] == k}
        for stage, predict in (("stage1", stage1), ("stage2", stage2)):
            hits, false, minutes = evaluate(predict, fold_clips, fold_videos)
            summary[stage][0] += hits
            summary[stage][1] += len(false)
            summary[stage][2] += minutes
            if stage == "stage2":
                tiles += [t for *_, t in false]
    for stage in ("stage1", "stage2"):
        hits, false, minutes = summary[stage]
        summary[stage] = {"recall": round(hits / len(clips), 2), "detected": hits, "false_alarms": false,
                          "minutes": round(minutes, 1), "fp_per_min": round(false / minutes, 3)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))[1].tofile(
            str(args.output_dir / "false_alarms_stage2.jpg"))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
