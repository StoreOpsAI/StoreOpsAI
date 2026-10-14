"""얼린 S3D + 선형 분류기를 서비스 체크포인트 형식(best.pt)으로 내보낸다: 서비스에 '바꿔 끼우기'가 되는지 확인하는 시험용(최종 모델 아님).

- 학습: 전 행동 공통 분할표의 전도 헤드 학습 가능 창(공식 validation 미사용) 전체. 양성=전도 창(누운 시간>=3초), 나머지=0. 클래스 균형 로지스틱.
- 입력 표준화(평균·표준편차)는 분류 층 가중치에 합쳐 넣는다: W' = W/std, b' = b - W·(mean/std). 서비스 로더가 그대로 읽는다.
- 주의: 서비스 S3D는 avgpool(커널 (2,7,7)) 뒤 평균을 내서 우리가 쓴 특징(전체 평균)과 시간 가중치가 약간 다르다. 시험용이라 허용하며, 미세조정 모델은 서비스 forward 그대로 학습하므로 이 차이가 없다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pipelines.fullscene_probe import load
from config.config import SERVICE_STACK_DIR

SERVICE_RUN = SERVICE_STACK_DIR.parent / "ours_probe_v0"


def fit(x: np.ndarray, y: np.ndarray, epochs: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.tensor((x - mean) / std, dtype=torch.float32), torch.tensor(y)
    torch.manual_seed(42)
    lin = nn.Linear(x.shape[1], 2)
    counts = torch.bincount(yt, minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        criterion(lin(xt), yt).backward()
        opt.step()
    w = lin.weight.detach() / torch.tensor(std)
    b = lin.bias.detach() - (lin.weight.detach() * torch.tensor(mean / std)).sum(1)
    return w, b


def main() -> None:
    from torchvision.models.video import S3D_Weights, s3d

    d = load()
    w, b = fit(d["x"].astype(np.float32), d["y"].astype(np.int64))
    model = s3d(weights=S3D_Weights.DEFAULT)
    model.classifier[1] = nn.Conv3d(model.classifier[1].in_channels, 2, kernel_size=1, stride=1)
    with torch.no_grad():
        model.classifier[1].weight.copy_(w.reshape(2, -1, 1, 1, 1))
        model.classifier[1].bias.copy_(b)
    label_map = {"normal": 0, "fall": 1}
    SERVICE_RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "label_map": label_map, "arch": "s3d", "size": 224, "num_frames": 24, "epoch": 0, "val_acc": None}, SERVICE_RUN / "best.pt")
    (SERVICE_RUN / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", SERVICE_RUN / "best.pt", "| 학습 창", len(d["y"]), "양성", int(d["y"].sum()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
