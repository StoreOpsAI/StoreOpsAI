"""유기: 손 상태 분류기(손 주변 사진 → 쥐고 있음 / 빈손). 손이 물건을 놓는 전환을 보려는 첫 단계.

근거: Shan 외, "Understanding Human Hands in Contact at Internet Scale", CVPR 2020 (손 접촉 상태 인식).
공개 모델(100DOH, MIT)은 PyTorch 1.12·CUDA 확장 컴파일이 필요해 우리 환경에 맞지 않아, 같은 과제를 우리 라벨로 학습한다.
- 라벨(사람이 단 것에서 직접): 유기 train 사건 구간에서 손 정답 좌표(Left/Right hand)가 물건 박스(object_abandon,
  한 변의 HAND_TOUCH_MARGIN만큼 넓힘) 안이면 '쥐고 있음', 그 사건에서 마지막으로 닿은 뒤의 프레임이면 '빈손'. 그 밖은 쓰지 않음
- 사진: 손 좌표 중심, 한 변 = 사람 키(머리~발 정답 좌표)의 HAND_CROP_RATIO. 추론 때는 YOLO-pose 손목을 쓰므로 넉넉히 자른다
- 동결 CLIP 이미지 인코더 + 선형(train_lying과 같음), 공통 분할 5-fold, 임계값 0.5. 공식 validation 미사용
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation, parse_object_boxes
from pipelines.compare_backbones import _fit_linear
from pipelines.make_splits import FOLDS, load_splits
from pipelines.prepare_behaviors import list_videos
from pipelines.train_abandonment import _metrics
from pipelines.train_lying import clip_features, load_clip, save_sheet

HAND_TOUCH_MARGIN = 0.3
HAND_CROP_RATIO = 0.35


def keypoints(xml):
    """{프레임: {사람 ID: {부위: (x, y)}}} (손·머리·발만)."""
    parts = {"Left hand": "lh", "Right hand": "rh", "Center head": "head", "Left foot": "lf", "Right foot": "rf"}
    out = {}
    for track in ET.parse(xml).findall(".//track"):
        part = parts.get(track.get("label"))
        for p in track.findall("points") if part else ():
            if p.get("outside") == "1":
                continue
            pid = next((a.text for a in p.findall("attribute") if a.get("name") == "ID"), "0")
            x, y = map(float, p.get("points").split(";")[0].split(","))
            out.setdefault(int(p.get("frame")), {}).setdefault(pid, {})[part] = (x, y)
    return out


def hand_crop(frame, center, person_height):
    side = max(32, int(person_height * HAND_CROP_RATIO))
    x, y = int(center[0]), int(center[1])
    h, w = frame.shape[:2]
    x1, y1 = max(0, x - side // 2), max(0, y - side // 2)
    return cv2.resize(frame[y1:min(h, y1 + side), x1:min(w, x1 + side)], (160, 160))


def _touch(point, box):
    x1, y1, x2, y2 = box
    m = HAND_TOUCH_MARGIN * max(x2 - x1, y2 - y1)
    return x1 - m <= point[0] <= x2 + m and y1 - m <= point[1] <= y2 + m


def samples():
    """[(사진, 라벨 1=쥐고 있음, 영상 이름)]."""
    out = []
    for xml, mp4, split in list_videos(WORKSPACE_ROOT / "유기", "유기")[0]:
        if split != "train":
            continue
        boxes, points = parse_object_boxes(xml), keypoints(xml)
        cap = cv2.VideoCapture(str(mp4))
        for event in parse_annotation(xml):
            frames = [f for f in sorted(boxes) if event["start_frame"] <= f <= event["end_frame"]]
            hands = [(f, pid, side, xy, person) for f in frames for pid, person in points.get(f, {}).items()
                     for side, xy in person.items() if side in ("lh", "rh")]
            touched = [(f, xy) for f, _, _, xy, _ in hands if any(_touch(xy, b) for b in boxes[f])]
            last_touch = max((f for f, _ in touched), default=None)
            for f, pid, side, xy, person in hands:
                touch = any(_touch(xy, b) for b in boxes[f])
                if not touch and (last_touch is None or f <= last_touch):
                    continue  # 놓기 전인데 안 닿은 프레임은 애매하므로 쓰지 않는다
                feet = [person[k][1] for k in ("lf", "rf") if k in person]
                height = (max(feet) - person["head"][1]) if feet and "head" in person else 300
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if ok:
                    out.append((hand_crop(frame, xy, max(100, height)), int(touch), Path(mp4).name))
        cap.release()
    return out


def main():
    parser = argparse.ArgumentParser(description="손 상태 분류기 (쥐고 있음 / 빈손)")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "hand_state_cv")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = samples()
    splits = load_splits()
    labels = np.array([s[1] for s in data])
    folds = np.array([splits[s[2]]["fold"] for s in data])
    save_sheet([s[0] for s in data if s[1] == 1][:96], args.output_dir / "holding.jpg")
    save_sheet([s[0] for s in data if s[1] == 0][:96], args.output_dir / "empty.jpg")
    features = clip_features([s[0] for s in data], *load_clip(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    np.savez(args.output_dir / "features.npz", features=features, labels=labels, folds=folds)

    scores, per_fold = np.zeros(len(data)), []
    for k in range(FOLDS):
        dev, train = folds == k, folds != k
        if not dev.any():
            continue
        predict = _fit_linear(features[train], labels[train])
        scores[dev] = predict(features[dev])
        per_fold.append({"fold": k, "train_f1": _metrics(labels[train].tolist(), predict(features[train]).tolist())["f1"],
                         "dev": _metrics(labels[dev].tolist(), scores[dev].tolist())})
    summary = {"holding": int(labels.sum()), "empty": int((labels == 0).sum()),
               "dev": _metrics(labels.tolist(), scores.tolist()),
               "train_dev_f1_gap": round(float(np.mean([f["train_f1"] - f["dev"]["f1"] for f in per_fold])), 3),
               "folds": per_fold}
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
