"""P 스트림: 관절 시퀀스 헤드(작은 GRU, 6클래스 로짓). 연구 쪽 storeops_ai_fr_evt/pipelines/pose_head.py와 같은 특징·구조여야 한다.

- 프레임 특징 53차원: 관절 17개 x (박스 중심 기준 x, y를 박스 높이로 나눈 값, 신뢰도) + 박스 가로/세로 비 + 사람 있음 표시. 절대 위치·화면 크기는 넣지 않는다.
- 창(12표본, 3fps)마다 프레임별로 박스가 가장 큰 사람 한 명의 관절을 쓴다. 사람이 없는 프레임은 0 벡터(학습과 같다).
- 가중치가 없으면 임의 점수를 만들지 않고 PoseHeadNotConfiguredError를 던진다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from utils.relation_features import DIM as REL_DIM

WIN, DIM, NUM_CLASSES = 12, 53, 6   # 정상, 전도, 유기, 절도, 파손, 방화


class PoseHeadNotConfiguredError(RuntimeError):
    pass


def frame_feature(kpt: np.ndarray, box: Sequence[float]) -> np.ndarray:
    w, h = max(float(box[2] - box[0]), 1.0), max(float(box[3] - box[1]), 1.0)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    xy = np.stack([(kpt[:, 0] - cx) / h, (kpt[:, 1] - cy) / h, kpt[:, 2]], axis=1)
    return np.concatenate([xy.reshape(-1), [w / h, 1.0]]).astype(np.float32)


def window_feature(frames: Sequence[Optional[Tuple[np.ndarray, Sequence[float]]]]) -> np.ndarray:
    """표본 프레임 12개마다 (관절 17x3, 박스) 또는 None -> (12, 53). None이거나 모자라면 0."""
    x = np.zeros((WIN, DIM), np.float32)
    for i, item in enumerate(list(frames)[-WIN:]):
        if item is not None:
            x[i] = frame_feature(*item)
    return x


def pick_largest(persons: Sequence[Tuple[np.ndarray, Sequence[float]]]) -> Optional[Tuple[np.ndarray, Sequence[float]]]:
    """한 프레임의 (관절, 박스) 목록에서 박스 넓이가 가장 큰 사람. 없으면 None."""
    if not len(persons):
        return None
    return max(persons, key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))


class PoseHead:
    def __init__(self, weight_path: Path, device: str = "cpu") -> None:
        import torch
        from torch import nn

        weight_path = Path(weight_path)
        if not weight_path.is_file():
            raise PoseHeadNotConfiguredError(f"P 헤드 가중치가 없습니다: {weight_path}")
        state = torch.load(weight_path, map_location=device, weights_only=True)["model"]
        self.in_dim = int(state["inp.0.weight"].shape[1])   # 53 = 관절만(P), 73 = 관절 + 사람-물건 관계 20(P+R)
        if self.in_dim not in (DIM, DIM + REL_DIM):
            raise PoseHeadNotConfiguredError(f"P 헤드 입력 차원 {self.in_dim}을 알 수 없습니다: {weight_path}")
        in_dim = self.in_dim

        class PoseGRU(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())
                self.gru = nn.GRU(128, 128, batch_first=True, bidirectional=True)
                self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))

            def forward(self, x):
                h, _ = self.gru(self.inp(x))
                return self.out(torch.cat([h.mean(1), h.max(1).values], dim=1))

        self._torch, self.device = torch, device
        self.model = PoseGRU().to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @property
    def uses_relation(self) -> bool:
        return self.in_dim == DIM + REL_DIM

    def logits(self, window: np.ndarray) -> np.ndarray:
        """(12, in_dim) -> (6,) 로짓."""
        if window.shape != (WIN, self.in_dim):
            raise ValueError(f"P 헤드 입력 모양 {window.shape}, 기대 {(WIN, self.in_dim)}")
        with self._torch.inference_mode():
            return self.model(self._torch.from_numpy(window[None]).to(self.device)).float().cpu().numpy()[0]


def _self_check() -> None:
    kpt = np.zeros((17, 3), np.float32)
    kpt[:, 2] = 1.0
    kpt[1] = [10.0, 0.0, 1.0]
    f = frame_feature(kpt, [-20.0, -50.0, 20.0, 50.0])
    assert f.shape == (DIM,) and abs(f[3] - 0.1) < 1e-6 and abs(f[-2] - 0.4) < 1e-6 and f[-1] == 1.0   # 연구 쪽 점검과 같은 값
    w = window_feature([None] * 11 + [(kpt, [-20.0, -50.0, 20.0, 50.0])])
    assert w.shape == (WIN, DIM) and w[:11].sum() == 0 and w[11, -1] == 1.0
    big, small = (kpt, [0, 0, 100, 200]), (kpt, [0, 0, 10, 20])
    assert pick_largest([small, big]) is big and pick_largest([]) is None
    try:
        PoseHead(Path("없는/경로.pt"))
        raise AssertionError("가중치가 없으면 예외여야 한다")
    except PoseHeadNotConfiguredError:
        pass
    print("ok")


if __name__ == "__main__":
    _self_check()
