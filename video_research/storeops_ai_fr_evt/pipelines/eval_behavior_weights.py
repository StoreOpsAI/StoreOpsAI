"""Compare linear-head class weights with session-held-out train clips only."""

import csv
import json

import numpy as np
import torch
from torch import nn

from pipelines.aihub_abandonment import APP_ROOT
from pipelines.cv_abandonment import assign_folds
from pipelines.cv_behaviors import AUGMENTATIONS, fit_pca
from pipelines.eval_behavior_alarm_budget import metrics


def fit(x, y, positive_weight):
    mean, std = x.mean(0), x.std(0) + 1e-6
    features = torch.tensor((x - mean) / std, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)
    torch.manual_seed(42)
    model = nn.Linear(x.shape[1], 2)
    loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, positive_weight]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(300):
        optimizer.zero_grad()
        loss(model(features), labels).backward()
        optimizer.step()

    @torch.no_grad()
    def predict(values):
        transformed = torch.tensor((values - mean) / std, dtype=torch.float32)
        return model(transformed).softmax(1)[:, 1].numpy()

    return predict


def main():
    root = APP_ROOT / "output"
    with (root / "behavior_training" / "samples_purchase.csv").open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "train"]
    data = np.load(root / "behavior_cv_purchase" / "xclip_features.npz")
    assert len(rows) == len(data["features"]) == len(data["augmented"])
    labels = np.array([row["label"] for row in rows])
    groups = np.array([row["group"] for row in rows])
    behaviors = sorted(set(labels) - {"정상"})
    classes = ["정상", *behaviors]
    fold_map = assign_folds(rows, 5)
    folds = np.array([fold_map[group] for group in groups])
    predictions = {setting: np.full(len(rows), "정상", dtype=object)
                   for setting in ("unweighted", "sqrt_balanced", "balanced")}

    for fold in range(5):
        train, dev = np.where(folds != fold)[0], np.where(folds == fold)[0]
        assert not set(groups[train]) & set(groups[dev])
        x = np.concatenate([data["features"][train],
                            data["augmented"][train].reshape(-1, data["features"].shape[1])])
        y = np.concatenate([labels[train], np.repeat(labels[train], AUGMENTATIONS)])
        project = fit_pca(x, 32)
        x, x_dev = project(x), project(data["features"][dev])
        for setting in predictions:
            scores = []
            for behavior in behaviors:
                binary = (y == behavior).astype(int)
                ratio = float((binary == 0).sum() / binary.sum())
                weight = {"unweighted": 1.0, "sqrt_balanced": ratio ** 0.5,
                          "balanced": ratio}[setting]
                scores.append(fit(x, binary, weight)(x_dev))
            scores = np.stack(scores, axis=1)
            predictions[setting][dev] = np.where(scores.max(1) >= 0.5,
                                                  np.array(behaviors)[scores.argmax(1)], "정상")

    result = {"samples": len(rows), "classes": classes,
              "settings": {setting: metrics(labels, prediction, classes)
                           for setting, prediction in predictions.items()},
              "note": "Five session-held-out folds within train; fall absent. No official validation or test."}
    (root / "codex_behavior_weights.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {"normal_false_alarms": value["normal_false_alarms"],
                            "recall": {name: round(score["recall"], 3)
                                       for name, score in value["per_class"].items()}}
                      for key, value in result["settings"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
