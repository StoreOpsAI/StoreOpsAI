"""학습 창 채점(최종 모델이 학습한 영상, train 성능 = 재대입 성능): S3D_INPUT=full -> 최종 A, c1 -> 최종 C1. 로짓을 output/train_eval/logits_<입력>.npz에 저장한다.

train-validation 차이를 기록하기 위한 용도(AGENTS.md). 모델·임계값 선정에는 쓰지 않는다.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from pipelines import finetune_s3d as F

OUT = F.APP_ROOT / "output" / "train_eval"


def main() -> None:
    rows = F.build_index()          # 학습에 쓴 모든 클립(격리 제외)의 창. 사건 라벨(label, cls) 포함
    print(f"[{F.INPUT}] 학습 창 {len(rows)}개, 클립 {len(set(r['clip'] for r in rows))}개", flush=True)
    ckpt_dir = "finetune_s3d" if F.INPUT == "full" else f"finetune_s3d_{F.INPUT}"
    ckpt = torch.load(F.APP_ROOT / "output" / ckpt_dir / "s3d_fold-1.pt", map_location="cpu", weights_only=True)
    model = F.make_model()
    model.load_state_dict(ckpt["model"], strict=True)
    logits = F.score_all(model.to("cuda"), rows, "cuda", 4)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"logits_{F.INPUT}.npz", clip=np.array([r["clip"] for r in rows]), start=np.array([r["start"] for r in rows]), label=np.array([r["label"] for r in rows]),
                        cls=np.array([r["cls"] for r in rows]), behavior=np.array([r["behavior"] for r in rows]), fold=np.array([r["fold"] for r in rows]), logits=logits)
    print("저장 완료", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
