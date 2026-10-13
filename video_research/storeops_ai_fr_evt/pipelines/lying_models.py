"""전도 분류기: 관절 특징 MLP. 학습·저장·불러오기를 한곳에 둔다(평가와 배포 실행기가 같은 코드를 쓴다).

입력은 표준화한 관절 특징(pipelines/lying_pose_dataset.pose_features), 출력은 누움 점수(0~1, 확률이 아님).
클래스 균형 가중 교차엔트로피, AdamW, 시드 고정. 설정은 config(LYING_MLP_*)의 사전 값이고 dev로 고르지 않는다.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from config.config import LYING_MLP_EPOCHS, LYING_MLP_HIDDEN

SCORE_MODE = os.getenv("LYING_SCORE", "prob")  # prob=소프트맥스 확률(기존), logit=두 클래스 출력의 차이(포화 없음, OOF 탐색용)


def _net(n_in, hidden=LYING_MLP_HIDDEN):
    layers, size = [], n_in
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    return nn.Sequential(*layers, nn.Linear(size, 2))


class LyingMLP:
    def __init__(self, net, mean, std):
        self.net, self.mean, self.std = net.eval(), mean, std

    @torch.no_grad()
    def __call__(self, x):
        z = torch.tensor((np.asarray(x, np.float32) - self.mean) / self.std, dtype=torch.float32)
        out = self.net(z)
        if SCORE_MODE == "logit":
            return (out[:, 1] - out[:, 0]).numpy()  # 로짓: 포화되지 않는다. 확률이 필요하면 calibration.fit_platt로 보정
        return out.softmax(1)[:, 1].numpy()

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "mean": self.mean, "std": self.std, "n_in": len(self.mean),
                    "hidden": list(LYING_MLP_HIDDEN)}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = _net(blob["n_in"], tuple(blob["hidden"]))
        net.load_state_dict(blob["state"])
        return cls(net, blob["mean"], blob["std"])


def fit_mlp(x, y, epochs=LYING_MLP_EPOCHS):
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.tensor((x - mean) / std, dtype=torch.float32), torch.tensor(y)
    torch.manual_seed(0)
    net = _net(x.shape[1])
    counts = torch.bincount(yt, minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 1024):
            batch = perm[i:i + 1024]
            optimizer.zero_grad()
            criterion(net(xt[batch]), yt[batch]).backward()
            optimizer.step()
    return LyingMLP(net, mean.astype(np.float32), std.astype(np.float32))


if __name__ == "__main__":
    import tempfile, pathlib
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, (300, 6)), rng.normal(2, 1, (300, 6))]).astype(np.float32)
    y = np.array([0] * 300 + [1] * 300, np.int64)
    model = fit_mlp(x, y, epochs=30)
    assert ((model(x) >= 0.5) == y).mean() > 0.9                       # 쉬운 두 덩어리는 나눈다
    path = pathlib.Path(tempfile.mkdtemp()) / "m.pt"
    model.save(path)
    assert np.allclose(model(x), LyingMLP.load(path)(x), atol=1e-6)    # 저장·불러오기 결과가 같다
    print("ok")
