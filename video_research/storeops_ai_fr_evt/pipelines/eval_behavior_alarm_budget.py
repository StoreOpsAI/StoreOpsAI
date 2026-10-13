"""Train-only, session-nested check of clip alarms at low false-positive budgets.

Reuses cached X-CLIP features from cv_behaviors. The official test split is never read.
"""

import csv
import json
from pathlib import Path

import numpy as np

from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.cv_abandonment import assign_folds
from pipelines.cv_behaviors import AUGMENTATIONS, cross_validate, fit_pca


def metrics(labels, predictions, classes):
    matrix = [[int(np.sum((labels == truth) & (predictions == guess))) for guess in classes]
              for truth in classes]
    per_class = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        precision = tp / max(1, sum(row[i] for row in matrix))
        recall = tp / max(1, sum(matrix[i]))
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {"confusion_matrix": matrix, "per_class": per_class,
            "normal_false_alarms": sum(matrix[0][1:]), "normal_clips": sum(matrix[0])}


def main():
    root = APP_ROOT / "output"
    manifest = root / "behavior_training" / "samples_purchase.csv"
    cache = root / "behavior_cv_purchase" / "xclip_features.npz"
    with manifest.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "train"]
    data = dict(np.load(cache))
    assert len(rows) == len(data["features"]) == len(data["augmented"])
    labels = np.array([row["label"] for row in rows])
    groups = np.array([row["group"] for row in rows])
    behaviors = sorted(set(labels) - {"정상"})
    classes = ["정상", *behaviors]
    outer_map = assign_folds(rows, 5)
    outer_ids = np.array([outer_map[group] for group in groups])
    predictions = {"0.5": np.full(len(rows), "정상", dtype=object),
                   "1pct": np.full(len(rows), "정상", dtype=object),
                   "0pct": np.full(len(rows), "정상", dtype=object)}
    chosen = []

    for fold in range(5):
        train, dev = np.where(outer_ids != fold)[0], np.where(outer_ids == fold)[0]
        assert not set(groups[train]) & set(groups[dev])
        inner_rows = [rows[i] for i in train]
        inner_map = assign_folds(inner_rows, 4)
        inner_ids = np.array([inner_map[row["group"]] for row in inner_rows])
        inner_data = {key: values[train] for key, values in data.items()}
        inner_behaviors, inner_scores, _ = cross_validate(
            inner_data, labels[train], inner_ids, 4, use_augmentation=True, pca_dims=32)
        assert inner_behaviors == behaviors

        x_train = np.concatenate([data["features"][train],
                                  data["augmented"][train].reshape(-1, data["features"].shape[1])])
        y_train = np.concatenate([labels[train], np.repeat(labels[train], AUGMENTATIONS)])
        project = fit_pca(x_train, 32)
        x_train, x_dev = project(x_train), project(data["features"][dev])
        scores = np.column_stack([_fit_linear(x_train, (y_train == behavior).astype(int))(x_dev)
                                  for behavior in behaviors])
        thresholds = {"0.5": np.full(len(behaviors), 0.5), "1pct": [], "0pct": []}
        for col, behavior in enumerate(behaviors):
            negatives = np.sort(inner_scores[labels[train] != behavior, col])
            # At most 1% negative clips in inner CV; zero-FP diagnostic is intentionally conservative.
            thresholds["1pct"].append(float(negatives[max(0, len(negatives) - max(1, len(negatives) // 100))]))
            thresholds["0pct"].append(float(np.nextafter(negatives[-1], 1.0)))
        for setting, limits in thresholds.items():
            limits = np.asarray(limits)
            eligible = scores >= limits
            best = np.where(eligible, scores, -1).argmax(axis=1)
            predictions[setting][dev] = np.where(eligible.any(axis=1), np.array(behaviors)[best], "정상")
        chosen.append({"fold": fold, "thresholds": {key: dict(zip(behaviors, map(float, value)))
                                                      for key, value in thresholds.items()}})

    result = {"samples": len(rows), "classes": classes, "settings": {
        setting: metrics(labels, prediction, classes) for setting, prediction in predictions.items()},
        "fold_thresholds": chosen,
        "note": "Four behaviors only; fall train data is absent. Clip errors are not alarms per day."}
    path = root / "codex_behavior_alarm_budget.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {"normal_false_alarms": value["normal_false_alarms"],
                            "recall": {name: round(score["recall"], 3) for name, score in value["per_class"].items()}}
                      for key, value in result["settings"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
