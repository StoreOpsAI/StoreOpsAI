"""공식 validation 창 채점(최종 모델, 입력 하나): S3D_INPUT=full -> 최종 A, S3D_INPUT=c1 -> 최종 C1. 로짓 4개를 output/official_val/logits_<입력>.npz에 저장한다.

설정 동결 후 1회용. 학습·임계값 조정에 쓰지 않는다. 이미 결과가 있으면 FORCE_REEVAL=1이 아니면 거부한다(AGENTS.md: test는 동결 후 한 번만).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

from pipelines import finetune_s3d as F
from pipelines.global_split import SPLIT_PATH

OUT = F.APP_ROOT / "output" / "official_val"


def val_rows() -> list[dict]:
    rows = []
    for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")):
        p = F.FULL / f"{r['clip']}.npz"
        if r["official_split"] != "validation" or not p.is_file() or not (F.FRAMES / f"{r['clip']}.pkl").is_file():
            continue
        z = np.load(p)
        starts = z["start"].tolist()
        if r["behavior"] == "전도":
            labels = z["label"].tolist()
        elif r["behavior"] in F.CLASS_OF:
            labels = F.event_window_labels(r["behavior"], r["clip"], starts, "validation")
        else:
            labels = [0] * len(starts)
        for s0, y in zip(starts, labels):
            rows.append({"clip": r["clip"], "behavior": r["behavior"], "start": s0, "label": int(y), "cls": F.CLASS_OF.get(r["behavior"], 0) if y == 1 else 0})
    return rows


def main() -> None:
    target = OUT / f"logits_{F.INPUT}.npz"
    if target.is_file() and os.getenv("FORCE_REEVAL") != "1":
        sys.exit(f"이미 채점했습니다: {target}. 공식 validation은 한 번만 평가합니다(다시 하려면 사유를 기록하고 FORCE_REEVAL=1).")
    rows = val_rows()
    print(f"[{F.INPUT}] 공식 validation 클립 {len(set(r['clip'] for r in rows))}개, 창 {len(rows)}개", flush=True)
    ckpt_dir = "finetune_s3d" if F.INPUT == "full" else f"finetune_s3d_{F.INPUT}"
    ckpt = torch.load(F.APP_ROOT / "output" / ckpt_dir / "s3d_fold-1.pt", map_location="cpu", weights_only=True)
    model = F.make_model()
    model.load_state_dict(ckpt["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = F.score_all(model.to(device), rows, device, 4)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, clip=np.array([r["clip"] for r in rows]), start=np.array([r["start"] for r in rows]), label=np.array([r["label"] for r in rows]),
                        cls=np.array([r["cls"] for r in rows]), behavior=np.array([r["behavior"] for r in rows]), logits=logits)
    print("저장:", target, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
