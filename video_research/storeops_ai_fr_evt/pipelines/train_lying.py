"""전도: 누워 있는 사람 분류기(사람 박스 잘라낸 사진 → 누움/아님). 2단계 방식(사람 검출 → 분류).

근거: Lafuente-Arroyo 외, "RGB camera-based fallen person detection system embedded on a mobile platform",
Expert Systems with Applications, 2022 (YOLO 사람 검출 뒤 쓰러짐 분류). 적은 데이터라 백본(CLIP 이미지 인코더)은 얼리고 선형만 학습.
- 양성: 전도/train 클립 마지막 LYING_LAST 프레임에서, 첫 프레임의 어떤 사람과도 가장 덜 겹치는 박스(새로 쓰러진 사람).
  고른 박스는 review_positive.jpg로 사람이 확인한다
- 음성: 전도 클립 첫 프레임의 사람(쓰러지기 5초 전 선 자세·점원), 다른 행동·반품 영상의 사람(숙이기·쪼그려 앉기 포함)
- 공통 분할(data/splits.json) 5-fold. 증강 없음, 임계값 0.5 고정. 공식 validation 미사용
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import LYING_BACKBONE
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.eval_new_object import load_person_boxes
from pipelines.make_splits import FOLDS, load_splits
from pipelines.prepare_behaviors import list_videos
from pipelines.train_abandonment import _metrics
from models.new_object_detector import _iou

LYING_LAST = 3
NEG_PER_VIDEO = 10


def square_crop(frame, box, pad=0.1):
    """박스를 여백 pad만큼 넓혀 자르고 회색으로 채워 정사각형으로 만든다(가운데 자르기로 머리·발이 잘리지 않게)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - w * pad)), max(0, int(y1 - h * pad))
    x2, y2 = min(frame.shape[1], int(x2 + w * pad)), min(frame.shape[0], int(y2 + h * pad))
    crop = frame[y1:y2, x1:x2]
    side = max(crop.shape[:2])
    canvas = np.full((side, side, 3), 114, np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


def _read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def fall_samples(detector):
    """[(crop, label, source 이름, 설명)]: 전도 클립의 양성(마지막 프레임 쓰러진 사람)과 음성(첫 프레임 사람)."""
    samples = []
    rejected = json.loads((APP_ROOT / "data" / "lying_review.json").read_text(encoding="utf-8"))
    for clip in sorted((WORKSPACE_ROOT / "전도" / "train").glob("*__fall")):
        frames = sorted(clip.glob("*.jpg"))
        # 검수에서 잘못 골라진 것으로 확인된 클립은 양성을 쓰지 않는다(첫 프레임 음성은 씀)
        last = [] if clip.name in rejected else frames[-LYING_LAST:]
        detect = lambda image: detector.predict(image, classes=[0], conf=0.25, verbose=False)[0].boxes.xyxy.tolist()
        first = _read(frames[0])
        first_boxes = detect(first)
        samples += [(square_crop(first, b), 0, clip.name, "first") for b in first_boxes]
        for path in last:
            image = _read(path)
            boxes = detect(image)
            if boxes:
                fallen = min(boxes, key=lambda b: max((_iou(b, f) for f in first_boxes), default=0.0))
                samples.append((square_crop(image, fallen), 1, clip.name, path.name))
    return samples


def video_negatives(rng):
    samples = []
    videos = sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))
    for name in ("유기", "절도", "파손", "방화"):
        videos += [mp4 for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]
    for video in videos:
        persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons")
        picks = rng.sample([(f, b) for f, boxes in enumerate(persons) for b in boxes],
                           min(NEG_PER_VIDEO, sum(map(len, persons))))
        cap = cv2.VideoCapture(str(video))
        for frame_no, box in sorted(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if ok:
                samples.append((square_crop(frame, box), 0, Path(video).name, f"f{frame_no}"))
        cap.release()
    return samples


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor

    return CLIPProcessor.from_pretrained(LYING_BACKBONE), CLIPModel.from_pretrained(LYING_BACKBONE).to(device).eval()


@torch.no_grad()
def clip_features(crops, processor, model):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(crops), 64):
        inputs = processor(images=[cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops[i:i + 64]], return_tensors="pt").to(device)
        embeds = model.get_image_features(**inputs)
        embeds = getattr(embeds, "pooler_output", embeds)
        out.append((embeds / embeds.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def fold_predictors(cache=APP_ROOT / "output" / "lying_cv" / "features.npz"):
    """fold k를 뺀 학습 사진으로 맞춘 분류기 목록(fold k 평가용). train_lying을 먼저 돌려 캐시를 만든다."""
    data = np.load(cache)
    return [_fit_linear(data["features"][data["folds"] != k], data["labels"][data["folds"] != k]) for k in range(FOLDS)]


def save_sheet(crops, path, size=128, per_row=12):
    tiles = [cv2.resize(c, (size, size)) for c in crops]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % per_row)
    cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]))[1].tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description="누워 있는 사람 분류기 (공통 분할 5-fold)")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "lying_cv")
    args = parser.parse_args()
    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = fall_samples(YOLO(str(APP_ROOT / "yolo11n.pt"))) + video_negatives(random.Random(0))
    save_sheet([s[0] for s in samples if s[1] == 1], args.output_dir / "review_positive.jpg")
    splits = load_splits()
    folds = np.array([splits[s[2]]["fold"] for s in samples])  # 분할에 없는 출처(공식 validation 등)는 KeyError로 멈춘다
    labels = np.array([s[1] for s in samples])
    sources = np.array([splits[s[2]]["behavior"] for s in samples])
    features = clip_features([s[0] for s in samples], *load_clip(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    np.savez(args.output_dir / "features.npz", features=features, labels=labels, folds=folds)  # eval_falldown이 fold별로 다시 학습

    scores, per_fold = np.zeros(len(samples)), []
    for k in range(FOLDS):
        dev, train = folds == k, folds != k
        predict = _fit_linear(features[train], labels[train])
        scores[dev] = predict(features[dev])
        per_fold.append({"fold": k, "train": _metrics(labels[train].tolist(), predict(features[train]).tolist()),
                         "dev": _metrics(labels[dev].tolist(), scores[dev].tolist())})
    predicted = scores >= 0.5
    summary = {"positives": int(labels.sum()), "negatives": int((labels == 0).sum()),
               "dev": _metrics(labels.tolist(), scores.tolist()),
               "train_dev_f1_gap": round(float(np.mean([f["train"]["f1"] - f["dev"]["f1"] for f in per_fold])), 3),
               "negative_false_rate_by_source": {src: round(float(predicted[(labels == 0) & (sources == src)].mean()), 3)
                                                 for src in sorted(set(sources[labels == 0]))},
               "folds": per_fold}
    save_sheet([s[0] for s, p in zip(samples, predicted) if p and s[1] == 0][:48] or [np.zeros((8, 8, 3), np.uint8)],
               args.output_dir / "false_positives.jpg")
    save_sheet([s[0] for s, p in zip(samples, predicted) if not p and s[1] == 1][:48] or [np.zeros((8, 8, 3), np.uint8)],
               args.output_dir / "missed.jpg")
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
