"""유기/정상 데이터 준비와 S3D 사전 검증 학습.

최종 7분류 I3D용 데이터 흐름을 먼저 검증하는 이진 실험이다.
공식 AI_validation은 threshold 선택에 쓰지 않고 마지막 test에만 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import S3D_Weights, s3d

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, _iou, build_manifest, parse_annotation


MEAN = torch.tensor([0.43216, 0.394666, 0.37645])[:, None, None, None]
STD = torch.tensor([0.22803, 0.22145, 0.216989])[:, None, None, None]


def session_key(video_path):
    return re.sub(r"_C[A-F]_RGB.*$", "", Path(video_path).stem)


def assign_splits(rows):
    train_groups = sorted({session_key(row["video"]) for row in rows if row["split"] == "train"},
                          key=lambda value: hashlib.sha1(value.encode()).hexdigest())
    dev_groups = set(train_groups[:max(1, round(len(train_groups) * 0.2))])
    result = {}
    for row in rows:
        group = session_key(row["video"])
        result[row["event_id"]] = "test" if row["split"] == "validation" else (
            "dev" if group in dev_groups else "train"
        )
    return result


def _sample_indices(start, end, count=16):
    return np.linspace(start, end, count).round().astype(int).tolist()


def _crop(frame, box, size=224):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    image = frame[y1:y2, x1:x2]
    if image.size == 0:
        raise ValueError("빈 사람 crop")
    scale = min(size / image.shape[1], size / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas


def _interpolate_box(start_box, end_box, ratio):
    return tuple(a + (b - a) * ratio for a, b in zip(start_box, end_box))


def _read_crop_clip(detector, video_path, indices, reference_boxes=None):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                return None
            frames.append(frame)
    finally:
        cap.release()
    predictions = detector.predict(frames, classes=[0], conf=0.20, verbose=False)
    crops = []
    for offset, (frame, prediction) in enumerate(zip(frames, predictions)):
        boxes = [tuple(box) for box in prediction.boxes.xyxy.cpu().tolist()]
        if not boxes:
            return None
        if reference_boxes:
            reference = reference_boxes[offset]
            box = max(boxes, key=lambda candidate: _iou(reference, candidate))
        elif crops:
            # 정상 클립: 첫 프레임에서 고른 사람을 IoU로 계속 따라가 한 클립에 한 사람만 담는다
            previous = box
            box = max(boxes, key=lambda candidate: _iou(previous, candidate))
            if _iou(previous, box) == 0.0:
                return None  # 따라가던 사람을 놓치면 다른 사람이 섞이지 않게 클립을 버린다
        else:
            box = max(boxes, key=lambda value: (value[2] - value[0]) * (value[3] - value[1]))
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        crops.append(_crop(frame, (x1 - w * 0.35, y1 - h * 0.20, x2 + w * 0.35, y2 + h * 0.20)))
    return np.stack(crops)


def _normal_windows(frame_count, fps, events):
    length = max(1, round(4 * fps))
    buffer_frames = round(2 * fps)
    candidates = []
    for start in range(0, max(1, frame_count - length + 1), length):
        end = start + length - 1
        if all(end < event["start_frame"] - buffer_frames or start > event["end_frame"] + buffer_frames
               for event in events):
            candidates.append((start, end))
    return candidates


def prepare(dataset_root, output_dir):
    from ultralytics import YOLO

    detector = YOLO(str(APP_ROOT / "yolo11n.pt"))
    source_manifest = output_dir / "source_manifest.csv"
    rows = build_manifest(dataset_root, source_manifest)
    split_by_event = assign_splits(rows)
    events_by_video = defaultdict(list)
    for row in rows:
        events_by_video[Path(row["video"])].append(row)

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for video_path, video_rows in sorted(events_by_video.items(), key=lambda item: str(item[0])):
        xml_path = Path(video_rows[0]["annotation"])
        events = parse_annotation(xml_path)
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
        cap.release()
        event_by_index = {event["event_index"]: event for event in events}

        for row in video_rows:
            event_index = int(row["event_id"].rsplit("_", 1)[1])
            event = event_by_index[event_index]
            end = min(event["end_frame"], event["start_frame"] + round(4 * fps) - 1)
            indices = _sample_indices(event["start_frame"], end)
            references = [_interpolate_box(event["start_box"], event["end_box"],
                                           (frame - event["start_frame"]) /
                                           max(1, event["end_frame"] - event["start_frame"]))
                          for frame in indices]
            clip = _read_crop_clip(detector, video_path, indices, references)
            if clip is None:
                raise RuntimeError(f"유기 crop 생성 실패: {row['event_id']}")
            sample_id = f"{row['event_id']}_pos"
            path = samples_dir / f"{sample_id}.npz"
            np.savez_compressed(path, frames=clip)
            samples.append({"sample_id": sample_id, "split": split_by_event[row["event_id"]], "label": 1,
                            "group": session_key(video_path), "video": str(video_path), "path": str(path)})

        split = split_by_event[video_rows[0]["event_id"]]
        normal_target = len(video_rows) * (2 if split == "train" else 1)
        normal_windows = _normal_windows(frame_count, fps, events)
        normal_count = 0
        for start, end in normal_windows:
            clip = _read_crop_clip(detector, video_path, _sample_indices(start, end))
            if clip is None:
                continue
            normal_count += 1
            sample_id = f"{video_path.stem}_{normal_count:02d}_neg"
            path = samples_dir / f"{sample_id}.npz"
            np.savez_compressed(path, frames=clip)
            samples.append({"sample_id": sample_id, "split": split, "label": 0,
                            "group": session_key(video_path), "video": str(video_path), "path": str(path)})
            if normal_count == normal_target:
                break

    manifest = output_dir / "samples.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)

    groups = {split: {row["group"] for row in samples if row["split"] == split}
              for split in ("train", "dev", "test")}
    assert not groups["train"] & groups["dev"]
    assert not groups["train"] & groups["test"]
    assert not groups["dev"] & groups["test"]
    summary = {split: {label: sum(row["split"] == split and row["label"] == label for row in samples)
                       for label in (0, 1)} for split in groups}
    audit = {
        "samples": summary,
        "group_counts": {split: len(values) for split, values in groups.items()},
        "group_overlaps": {
            "train_dev": sorted(groups["train"] & groups["dev"]),
            "train_test": sorted(groups["train"] & groups["test"]),
            "dev_test": sorted(groups["dev"] & groups["test"]),
        },
    }
    (output_dir / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "samples": summary}, ensure_ascii=False))
    return manifest


class ClipDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames = np.load(row["path"])["frames"]
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255
        if self.augment:
            if torch.rand(()) < 0.5:
                clip = clip.flip(-1)
            clip = (clip * (0.8 + torch.rand(()) * 0.4)).clamp(0, 1)
        return (clip - MEAN) / STD, int(row["label"])


def _metrics(labels, probabilities, threshold=0.5):
    predictions = [int(value >= threshold) for value in probabilities]
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    normal_precision = tn / (tn + fn) if tn + fn else 0.0
    normal_recall = tn / (tn + fp) if tn + fp else 0.0
    normal_f1 = (2 * normal_precision * normal_recall / (normal_precision + normal_recall)
                 if normal_precision + normal_recall else 0.0)
    return {
        "accuracy": (tn + tp) / len(labels) if labels else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "per_class": {
            "정상": {"precision": normal_precision, "recall": normal_recall, "f1": normal_f1},
            "유기": {"precision": precision, "recall": recall, "f1": f1},
        },
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for clips, target in loader:
        probability = model(clips.to(device)).softmax(1)[:, 1].cpu().tolist()
        labels.extend(target.tolist())
        probabilities.extend(probability)
    return labels, probabilities


def train(manifest, output_dir, epochs=8, batch_size=2, evaluate_test=False, select_on_dev=True):
    """select_on_dev=False면 epoch·threshold를 dev로 고르지 않는다(마지막 epoch, 0.5 고정).

    교차검증처럼 dev 점수를 그대로 성능으로 보고할 때 낙관적 편향을 막기 위해 쓴다.
    """
    os.environ.setdefault("TORCH_HOME", str(APP_ROOT / "output" / "torch_cache"))
    with manifest.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    datasets = {split: ClipDataset([row for row in rows if row["split"] == split], split == "train")
                for split in ("train", "dev", "test")}
    loaders = {split: DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0)
               for split, dataset in datasets.items()}
    train_eval_loader = DataLoader(
        ClipDataset([row for row in rows if row["split"] == "train"]),
        batch_size=batch_size,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = s3d(weights=S3D_Weights.DEFAULT)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[1] = nn.Conv3d(1024, 2, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    train_counts = [sum(int(row["label"]) == label and row["split"] == "train" for row in rows)
                    for label in (0, 1)]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        [sum(train_counts) / (2 * count) for count in train_counts], device=device
    ))
    best_f1, best_state = -1.0, None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        model.features.eval()
        total_loss = 0.0
        for clips, labels in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(clips.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        dev_labels, dev_probabilities = evaluate(model, loaders["dev"], device)
        dev = _metrics(dev_labels, dev_probabilities)
        history.append({"epoch": epoch, "loss": total_loss / len(datasets["train"]), "dev": dev})
        print(json.dumps(history[-1], ensure_ascii=False))
        if dev["f1"] > best_f1 or not select_on_dev:
            best_f1 = dev["f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    dev_labels, dev_probabilities = evaluate(model, loaders["dev"], device)
    candidates = np.linspace(0.05, 0.95, 19)
    threshold = (max(candidates, key=lambda value: _metrics(dev_labels, dev_probabilities, value)["f1"])
                 if select_on_dev else 0.5)
    train_labels, train_probabilities = evaluate(model, train_eval_loader, device)
    train_metrics = _metrics(train_labels, train_probabilities, threshold)
    dev_metrics = _metrics(dev_labels, dev_probabilities, threshold)
    results = {
        "model": "torchvision_s3d_kinetics400_binary_prototype",
        "classes": ["정상", "유기"],
        "threshold": float(threshold),
        "train": train_metrics,
        "dev": dev_metrics,
        "generalization_gap": {
            "accuracy": train_metrics["accuracy"] - dev_metrics["accuracy"],
            "f1": train_metrics["f1"] - dev_metrics["f1"],
        },
        "history": history,
    }
    if evaluate_test:
        test_labels, test_probabilities = evaluate(model, loaders["test"], device)
        results["test"] = _metrics(test_labels, test_probabilities, threshold)
    checkpoint = output_dir / "abandonment_s3d.pt"
    torch.save({"state_dict": best_state, **{key: results[key] for key in ("model", "classes", "threshold")}}, checkpoint)
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), **results}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "유기")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "abandonment_training")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="설정을 모두 동결한 최종 실행에서 한 번만 사용")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "samples.csv"
    if not args.train_only:
        manifest = prepare(args.dataset_root, args.output_dir)
    if not args.prepare_only:
        train(manifest, args.output_dir, args.epochs, args.batch_size, args.evaluate_test)


if __name__ == "__main__":
    main()
