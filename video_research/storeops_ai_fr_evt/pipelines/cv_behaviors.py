"""행동별 판정기(얼린 X-CLIP + 선형, one-vs-rest = 공유 특징 위 행동별 sigmoid 다중 라벨) 교차검증. 기본은 세션 fold, --shared-splits면 공통 분할.

- split=test(공식 validation) 행은 특징도 뽑지 않는다.
- 증강은 항상 쓴다. 증강(좌우 반전·밝기·확대 자르기)본 특징은 미리 뽑아 두되, 그 샘플의 세션이 해당 fold의 학습 쪽일 때만 쓴다.
  dev 평가는 항상 원본 특징만 쓴다.
- 행동 b 판정기의 음성 = 정상 + 다른 모든 행동 클립. threshold 0.5 고정, 설정은 dev로 고르지 않는다.
- 최종 분류 = 0.5 이상인 판정기 중 최고점 행동, 없으면 정상.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import _cached, _fit_linear, _load_frames, load_xclip, xclip_embed
from pipelines.cv_abandonment import _summary, assign_folds
from pipelines.train_abandonment import _metrics

AUGMENTATIONS = 2


def augment(clip, seed, kind):
    """clip: (T, H, W, 3) uint8. kind 0 = 좌우 반전 + 밝기, 1 = 확대 자르기 + 밝기. 클립 전체에 같은 변환."""
    rng = np.random.default_rng(seed)
    out = clip[:, :, ::-1] if kind == 0 else clip
    if kind == 1:
        size = clip.shape[1]
        crop = int(size * rng.uniform(0.8, 0.95))
        y, x = rng.integers(0, size - crop + 1, 2)
        out = np.stack([cv2.resize(frame[y:y + crop, x:x + crop], (size, size)) for frame in clip])
    return np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)


def extract_features(rows, device):
    processor, model = load_xclip(device)
    original, augmented = [], []
    for row in rows:
        clip = _load_frames(row)
        seed = int(hashlib.sha1(row["sample_id"].encode()).hexdigest()[:8], 16)
        variants = [clip] + [augment(clip, seed + kind, kind) for kind in range(AUGMENTATIONS)]
        embeds = xclip_embed(processor, model, variants)
        original.append(embeds[0])
        augmented.append(embeds[1:])
    return {"features": np.stack(original), "augmented": np.stack(augmented)}


def fit_pca(x_train, dims):
    """train 쪽 특징으로만 주성분을 구해 투영 함수를 돌려준다. dims=0이면 그대로."""
    if not dims:
        return lambda values: values
    mean = x_train.mean(0)
    components = np.linalg.svd(x_train - mean, full_matrices=False)[2][:dims]
    return lambda values: (values - mean) @ components.T


def cross_validate(data, labels, fold_ids, folds, use_augmentation, pca_dims=0):
    behaviors = sorted(set(labels) - {"정상"})
    scores = np.zeros((len(labels), len(behaviors)))
    per_behavior = {behavior: [] for behavior in behaviors}
    for fold in range(folds):
        dev, train = np.where(fold_ids == fold)[0], np.where(fold_ids != fold)[0]
        assert not set(dev) & set(train)
        x_train = data["features"][train]
        if use_augmentation:  # 학습 쪽 샘플의 증강본만 추가한다
            x_train = np.concatenate([x_train, data["augmented"][train].reshape(-1, x_train.shape[1])])
        repeat = 1 + AUGMENTATIONS if use_augmentation else 1
        train_labels = np.concatenate([labels[train]] + [np.repeat(labels[train], AUGMENTATIONS)] * (repeat > 1))
        project = fit_pca(x_train, pca_dims)  # dev는 투영만 한다
        x_train, x_dev, x_train_original = project(x_train), project(data["features"][dev]), project(data["features"][train])
        for column, behavior in enumerate(behaviors):
            predict = _fit_linear(x_train, (train_labels == behavior).astype(np.int64))
            scores[dev, column] = predict(x_dev)
            train_metrics = _metrics((labels[train] == behavior).astype(int).tolist(),
                                     predict(x_train_original).tolist())
            dev_metrics = _metrics((labels[dev] == behavior).astype(int).tolist(), scores[dev, column].tolist())
            per_behavior[behavior].append({"train_f1": train_metrics["f1"], **{
                key: dev_metrics[key] for key in ("precision", "recall", "f1")}})
    return behaviors, scores, per_behavior


def summarize(behaviors, scores, per_behavior, labels, sources):
    predicted = np.where(scores.max(1) >= 0.5, np.array(behaviors)[scores.argmax(1)], "정상")
    classes = ["정상"] + behaviors
    confusion = [[int(((labels == truth) & (predicted == guess)).sum()) for guess in classes] for truth in classes]
    per_class = {}
    for index, name in enumerate(classes):
        tp = confusion[index][index]
        precision = tp / max(1, sum(row[index] for row in confusion))
        recall = tp / max(1, sum(confusion[index]))
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {
        "one_vs_rest": {behavior: {key: _summary([fold[key] for fold in folds]) for key in ("precision", "recall", "f1")}
                        | {"gap_f1": float(np.mean([fold["train_f1"] - fold["f1"] for fold in folds]))}
                        for behavior, folds in per_behavior.items()},
        "multiclass": {"classes": classes, "confusion_matrix(row=정답, col=예측)": confusion, "per_class": per_class},
        "normal_false_alarm_rate_by_source": {
            source: float((predicted[(labels == "정상") & (sources == source)] != "정상").mean())
            for source in sorted(set(sources[labels == "정상"]))},
    }


def main():
    parser = argparse.ArgumentParser(description="행동별 판정기 세션 교차검증 (공식 validation 미사용, 증강은 학습 쪽만)")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "behavior_training" / "samples.csv")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "behavior_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca", type=int, nargs="+", default=[0, 64, 32, 16], help="비교할 특징 차원 (0 = 축소 없음)")
    parser.add_argument("--shared-splits", action="store_true", help="세션 fold 대신 공통 분할(data/splits.json) 사용")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] != "test"]
    if args.shared_splits:  # 모든 모듈 공통 분할(촬영 날짜 그룹, data/splits.json). 없는 영상은 KeyError로 멈춘다
        from pipelines.make_splits import FOLDS, load_splits

        splits = load_splits()
        args.folds = FOLDS
        fold_ids = np.array([splits[Path(row["video"]).name]["fold"] for row in rows])
        groups = np.array([splits[Path(row["video"]).name]["group"] for row in rows])
    else:
        fold_of = assign_folds(rows, args.folds)
        fold_ids = np.array([fold_of[row["group"]] for row in rows])
        groups = np.array([row["group"] for row in rows])
    for fold in range(args.folds):  # 누수 검사: 한 세션이 학습과 dev에 동시에 있으면 멈춘다
        shared = set(groups[fold_ids == fold]) & set(groups[fold_ids != fold])
        if shared:
            raise RuntimeError(f"fold {fold}: 학습/dev 세션 겹침 {sorted(shared)[:3]}")
    labels, sources = np.array([row["label"] for row in rows]), np.array([row["source"] for row in rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _cached(args.output_dir / "xclip_features.npz", rows, lambda: extract_features(rows, device))

    results = {"samples": {label: int((labels == label).sum()) for label in sorted(set(labels))}}
    for dims in args.pca:
        results[f"pca_{dims or 'none'}"] = summarize(
            *cross_validate(data, labels, fold_ids, args.folds, use_augmentation=True, pca_dims=dims), labels, sources)
    # 대조 실험: 학습 라벨을 무작위로 섞으면 dev 점수가 찍기 수준이어야 한다(누수가 있으면 높게 나온다)
    shuffled = np.random.default_rng(0).permutation(labels)
    control = summarize(*cross_validate(data, shuffled, fold_ids, args.folds, use_augmentation=True), labels, sources)
    results["shuffled_label_control"] = {"per_class_f1": {name: value["f1"] for name, value
                                                          in control["multiclass"]["per_class"].items()}}
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
