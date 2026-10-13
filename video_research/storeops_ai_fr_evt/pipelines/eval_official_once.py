"""고정된 방화·전도 규칙으로 공식 validation을 각 한 번 평가한다.

기존 train/validation의 같은 촬영 회차 중복은 audit_official.py에 기록되어 있다.
이 결과는 해당 한계가 있는 평가이며, val 결과로 설정을 바꾸거나 재평가하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import FIRE_CONF, FIRE_IMGSZ
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.compare_backbones import _fit_linear
from pipelines.eval_falldown import (CACHE, HARD_PER_VIDEO, SAMPLE_FPS, cached_crops,
                                     clip_people, evaluate as evaluate_fall, video_paths, video_people)
from pipelines.eval_new_object import load_person_boxes
from pipelines.prepare_behaviors import list_videos
from pipelines.train_fire import load_samples, mine_negatives, train_fold, video_scores
from pipelines.train_fire_mix import load_dfire, write_images
from pipelines.train_lying import load_clip

OUT = APP_ROOT / "output" / "official_validation"


def save_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fire():
    from ultralytics import YOLO
    import random

    result = OUT / "fire.json"
    assert not result.exists(), f"이미 공식 평가함: {result}"
    folder = OUT / "fire_model"
    weights = folder / "run" / "weights" / "last.pt"
    train = load_samples()  # 방화 train만
    others = [mp4 for name in ("절도", "파손", "유기")
              for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]
    others += sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))
    val = [(mp4, parse_annotation(xml, "fire")) for xml, mp4, split
           in list_videos(WORKSPACE_ROOT / "방화", "방화")[0] if split == "test"]
    val_other = [(mp4, []) for name in ("절도", "파손", "유기")
                 for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "test"]
    assert len(train) == 24 and len(val) == 8, "방화 데이터 목록이 기존 평가 시점과 다름"
    assert {v["video"].name for v in train}.isdisjoint({v.name for v, _ in val})
    if not weights.exists():
        stage1 = train_fold(train, folder / "stage1", 40)
        negatives = mine_negatives(stage1, others, folder / "mine", seed=0)
        dfire, _ = load_dfire(random.Random(0), with_patches=False)
        extra = write_images(dfire, folder / "extra")
        train_fold(train + negatives, folder, 25, extra_paths=extra)
    model = YOLO(str(weights))
    # val 점수 계산은 이 아래에서 딱 한 번만 수행한다.
    result_data = {
        "model_sha256": digest(weights),
        "rule": {"confidence": FIRE_CONF, "image_size": FIRE_IMGSZ, "training": "all train; stage1 40 epochs; stage3 25 epochs + D-Fire"},
        "validation_fire": video_scores(model, val, OUT / "fire_detections"),
        "validation_other": video_scores(model, val_other, OUT / "fire_detections"),
        "caveat": "train과 validation에 같은 매장·날짜 촬영 회차가 있어 독립 외부 평가가 아니다",
    }
    save_once(result, result_data)
    print(json.dumps(result_data, ensure_ascii=False, indent=2))


def fall():
    from ultralytics import YOLO

    result = OUT / "fall.json"
    assert not result.exists(), f"이미 공식 평가함: {result}"
    base_path = APP_ROOT / "output" / "lying_cv" / "features.npz"
    base = np.load(base_path)
    x, y = base["features"], base["labels"]
    assert len(x) == len(y) and len(x) > 0
    stage1 = _fit_linear(x, y)
    mined = []
    for name in video_paths():
        data = dict(np.load(CACHE / f"{Path(name).stem}.npz"))
        scores = stage1(data["features"])
        top = np.argsort(-scores)[:HARD_PER_VIDEO]
        mined += [data["features"][i] for i in top if scores[i] >= 0.5]
    predict = _fit_linear(np.concatenate([x, np.array(mined).reshape(-1, x.shape[1])]),
                          np.concatenate([y, np.zeros(len(mined), y.dtype)]))
    detector = YOLO(str(APP_ROOT / "yolo11n.pt"))
    processor, model = load_clip(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    clips = [cached_crops(folder.name, lambda f=folder: clip_people(f, detector), processor, model)
             for folder in sorted((WORKSPACE_ROOT / "전도" / "val").glob("*__fall"))]
    val_paths = {mp4.name: mp4 for name in ("유기", "절도", "파손", "방화")
                 for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "test"}
    videos = {}
    for name, path in val_paths.items():
        data = cached_crops(path.stem, lambda p=path: video_people(p), processor, model)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or SAMPLE_FPS
        cap.release()
        step = max(1, round(fps / SAMPLE_FPS))
        length = len(load_person_boxes(path, APP_ROOT / "output" / "new_object_eval" / "persons"))
        videos[name] = (data, fps, list(range(0, length, step)))
    hits, false, minutes = evaluate_fall(predict, clips, videos)
    result_data = {
        "train_features_sha256": digest(base_path),
        "rule": {"threshold": 0.5, "sample_fps": SAMPLE_FPS, "hard_negatives": len(mined), "training": "all train, two-stage linear classifier"},
        "validation": {"clips": len(clips), "detected": hits, "recall": hits / len(clips),
                       "false_alarms": len(false), "negative_minutes": minutes,
                       "false_alarms_per_min": len(false) / minutes if minutes else None},
        "caveat": "train과 validation에 같은 매장·날짜 촬영 회차가 있어 독립 외부 평가가 아니다",
    }
    save_once(result, result_data)
    print(json.dumps(result_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("fire", "fall"))
    args = parser.parse_args()
    (fire if args.target == "fire" else fall)()
