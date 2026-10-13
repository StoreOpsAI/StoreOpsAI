"""유기/정상 S3D 세션 단위 교차검증.

- 공식 validation(test) 샘플은 목록에서 제외해 어떤 fold에서도 쓰지 않는다.
- train+dev 샘플을 촬영 세션(group) 단위로 k개 fold에 나눠, 한 세션이 train과 dev에 동시에 들어가지 않게 한다.
- fold 안에서는 epoch·threshold를 dev로 고르지 않는다(select_on_dev=False). dev 점수를 그대로 보고하기 때문이다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from pipelines.aihub_abandonment import APP_ROOT
from pipelines.train_abandonment import train


def assign_folds(rows, folds):
    groups = sorted({row["group"] for row in rows}, key=lambda value: hashlib.sha1(value.encode()).hexdigest())
    return {group: index % folds for index, group in enumerate(groups)}


def _summary(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(min(values))}


def cross_validate(manifest, output_dir, folds=5, epochs=8, batch_size=2):
    with manifest.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] in ("train", "dev")]
    fold_of = assign_folds(rows, folds)
    results = []
    for fold in range(folds):
        fold_rows = [{**row, "split": "dev" if fold_of[row["group"]] == fold else "train"} for row in rows]
        groups = {split: {row["group"] for row in fold_rows if row["split"] == split} for split in ("train", "dev")}
        if groups["train"] & groups["dev"]:
            raise RuntimeError(f"fold {fold}: train/dev 세션이 겹칩니다: {sorted(groups['train'] & groups['dev'])}")

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_manifest = fold_dir / "samples.csv"
        with fold_manifest.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fold_rows[0].keys())
            writer.writeheader()
            writer.writerows(fold_rows)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        train(fold_manifest, fold_dir, epochs, batch_size, evaluate_test=False, select_on_dev=False)
        result = json.loads((fold_dir / "results.json").read_text(encoding="utf-8"))
        results.append({
            "fold": fold,
            "dev_groups": sorted(groups["dev"]),
            "dev_samples": {label: sum(row["split"] == "dev" and row["label"] == label for row in fold_rows)
                            for label in ("0", "1")},
            "train": result["train"],
            "dev": result["dev"],
            "generalization_gap": result["generalization_gap"],
        })

    confusion = np.sum([item["dev"]["confusion_matrix"] for item in results], axis=0).tolist()
    summary = {
        "folds": folds,
        "threshold": 0.5,
        "dev_f1_유기": _summary([item["dev"]["f1"] for item in results]),
        "dev_f1_정상": _summary([item["dev"]["per_class"]["정상"]["f1"] for item in results]),
        "dev_accuracy": _summary([item["dev"]["accuracy"] for item in results]),
        "gap_f1": _summary([item["generalization_gap"]["f1"] for item in results]),
        "dev_confusion_matrix_total": confusion,
        "per_fold": results,
    }
    (output_dir / "cv_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="유기/정상 세션 단위 교차검증 (공식 validation 미사용)")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "abandonment_training" / "samples.csv")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "abandonment_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    summary = cross_validate(args.manifest, args.output_dir, args.folds, args.epochs, args.batch_size)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_fold"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
