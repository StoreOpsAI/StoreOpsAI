"""6클래스 학습 창 채점(train 성능 = 재대입): 최종 모델(s3d_fold-1.pt)로 **최종 학습에 쓴 영상**의 모든 창을 다시 채점한다.

- S3D_CLASSES=6, S3D_INPUT=full -> 기준선 A(output/finetune_s3d6), c1 -> 기준선 C1(output/finetune_s3d_c16). 정상 구성을 바꾼 모델은 AUDIT_NORMALS=1.
- 학습에 쓴 영상 = select_clips가 고른 양성 영상 + 매장이동 영상(finetune_s3d.epoch_items가 정상을 뽑는 풀). 고르지 않은 영상의 창은 학습하지 않았으므로 제외한다.
- 로짓을 output/train_eval/logits6_<입력>[_audit].npz에 저장한다(OOF와 같은 형식). train-validation 차이를 기록하기 위한 용도이며 모델·임계값 선정에는 쓰지 않는다. 공식 validation은 읽지 않는다.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from pipelines import finetune_s3d as F

OUT = F.APP_ROOT / "output" / "train_eval"


def main() -> None:
    assert F.SIX, "S3D_CLASSES=6으로 실행해야 한다"
    rows = F.build_index()
    chosen = F.select_clips(rows)
    train = [r for r in rows if r["clip"] in chosen or r["behavior"] == "매장이동"]
    print(f"[{F.INPUT}] 전체 창 {len(rows)}개 중 학습에 쓴 영상의 창 {len(train)}개, 영상 {len(set(r['clip'] for r in train))}개", flush=True)
    ckpt_path = F.OUT / "s3d_fold-1.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = F.make_model()
    model.load_state_dict(ckpt["model"], strict=True)
    logits = F.score_all(model.to("cuda"), train, "cuda", 4)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"logits6_{F.INPUT}" + ("_audit" if F.AUDIT_NORMALS else "") + ".npz"
    np.savez_compressed(OUT / name, clip=np.array([r["clip"] for r in train]), start=np.array([r["start"] for r in train]),
                        label=np.array([r["label"] for r in train]), cls=np.array([r["cls"] for r in train]),
                        behavior=np.array([r["behavior"] for r in train]), fold=np.array([r["fold"] for r in train]), logits=logits)
    print(f"저장 완료: {OUT / name} (모델 {ckpt_path})", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
