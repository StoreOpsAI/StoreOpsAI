"""전도(쓰러짐) 증거: 사람 박스·관절 -> 누움 로짓(관절 특징 MLP) -> 같은 자리에서 유지된 누움 점수.

- 점수는 확률이 아니라 로짓이다(소프트맥스 확률은 포화되어 기준을 고를 수 없다). 서비스 점수 변환은 behavior_router가 한다.
- "같은 자리"는 추적 번호가 아니라 박스 IoU로 이어 붙인다(넘어질 때 번호가 바뀌어도 된다). 서 있던 기록은 요구하지 않는다.
- 임계값·유지 시간·IoU는 모두 호출자가 config에서 가져와 넘긴다(여기에 상수를 두지 않는다).
연구 코드(storeops_ai_fr_evt)의 pose_features, fall_detector.lying_alarms와 같은 계산이다.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]
# 한 표본(프레임)의 사람들: [(박스, 관절 17x3(x, y, 신뢰도), 누움 로짓)]
Sample = List[Tuple[Box, np.ndarray, float]]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pose_features(box: Box, kpt: np.ndarray, kpt_conf: float) -> np.ndarray:
    """관절을 사람 박스 중심·긴 변 기준으로 정규화한 56차원 특징. 박스 위치는 넣지 않는다(카메라 배치 편향 방지)."""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    scale, cx, cy = max(w, h), (x1 + x2) / 2, (y1 + y2) / 2
    ok = kpt[:, 2] >= kpt_conf
    xy = np.where(ok[:, None], (kpt[:, :2] - [cx, cy]) / scale, 0.0)
    torso = np.zeros(3)
    if ok[[5, 6]].all() and ok[[11, 12]].all():  # 어깨·엉덩이가 모두 보일 때만 몸통 방향
        v = kpt[[5, 6], :2].mean(0) - kpt[[11, 12], :2].mean(0)
        norm = np.linalg.norm(v) + 1e-6
        torso = np.array([v[0] / norm, v[1] / norm, 1.0])
    return np.concatenate([xy.ravel(), kpt[:, 2], [h / w, w / h], torso]).astype(np.float32)


class PoseMLP:
    """연구 코드(LyingMLP.save)가 저장한 가중치를 읽는 추론 전용 MLP. 출력은 누움 로짓(양성 출력 - 음성 출력)."""

    def __init__(self, path: str):
        import torch
        from torch import nn

        blob = torch.load(path, map_location="cpu", weights_only=False)
        layers, size = [], blob["n_in"]
        for h in blob["hidden"]:
            layers += [nn.Linear(size, h), nn.ReLU()]
            size = h
        self._net = nn.Sequential(*layers, nn.Linear(size, 2))
        self._net.load_state_dict(blob["state"])
        self._net.eval()
        self._mean, self._std, self._torch = blob["mean"], blob["std"], torch

    def logits(self, feats: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            z = self._torch.tensor((np.asarray(feats, np.float32) - self._mean) / self._std, dtype=self._torch.float32)
            out = self._net(z)
            return (out[:, 1] - out[:, 0]).numpy()


def _has_alarm(samples: Sequence[Tuple[int, Sample]], thr: float, fps: float, hold_sec: float, link_iou: float, gap: int = 1) -> bool:
    """누움 로짓 >= thr인 사람이 같은 자리(IoU >= link_iou)에서 hold_sec 이상 이어지면 True. 표본 gap개까지 끊겨도 잇는다."""
    runs: list[dict] = []
    for index, people in samples:
        for run in runs:
            run["misses"] += 1
        for box, _, logit in people:
            if logit < thr:
                continue
            best = max(runs, key=lambda r: iou(r["box"], box), default=None)
            if best is not None and iou(best["box"], box) >= link_iou:
                best.update(box=box, last=index, misses=0)
            else:
                best = {"box": box, "start": index, "last": index, "misses": 0}
                runs.append(best)
            if best["last"] - best["start"] >= hold_sec * fps:
                return True
        runs = [r for r in runs if r["misses"] <= gap]
    return False


def sustained_lying_logit(samples: Sequence[Tuple[int, Sample]], fps: float, hold_sec: float, link_iou: float) -> float:
    """창 안에서 '같은 자리 hold_sec 유지'가 성립하는 가장 높은 로짓 기준. 없으면 -inf.

    경보 여부는 이 값이 설정 임계값 이상인가로 정해진다(임계값을 바꿔도 다시 계산하지 않는다).
    누움 판정이 임계값에 대해 단조(임계값을 낮출수록 경보가 늘어남)라고 보고 후보 로짓 위에서 이분 탐색한다.
    """
    logits = sorted({round(float(logit), 6) for _, people in samples for _, _, logit in people})
    if not logits or not _has_alarm(samples, logits[0], fps, hold_sec, link_iou):
        return float("-inf")
    lo, hi = 0, len(logits) - 1  # logits[lo]에서는 경보 성립
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _has_alarm(samples, logits[mid], fps, hold_sec, link_iou):
            lo = mid
        else:
            hi = mid - 1
    return logits[lo]


def score_window(samples: Sequence[Tuple[int, Sample]], scorer, fps: float, hold_sec: float, link_iou: float, kpt_conf: float) -> float:
    """samples: [(표본 번호, [(박스, 관절)])]. scorer.logits(특징) -> 누움 로짓. 창의 지속 누움 로짓을 돌려준다."""
    scored: list[tuple[int, Sample]] = []
    for index, people in samples:
        if people:
            feats = np.stack([pose_features(box, kpt, kpt_conf) for box, kpt in people])
            logits = scorer.logits(feats)
            scored.append((index, [(box, kpt, float(z)) for (box, kpt), z in zip(people, logits)]))
        else:
            scored.append((index, []))
    return sustained_lying_logit(scored, fps, hold_sec, link_iou)


if __name__ == "__main__":
    class Fake:  # 박스가 넓으면(누움) 높은 로짓
        def logits(self, feats):
            return np.where(feats[:, 52] > 1.0, 5.0, -5.0)  # 52열 = h/w 앞의 비율 특징 자리(w/h는 53열)

    kpt = np.zeros((17, 3), np.float32)
    lying, stand = (100.0, 200.0, 300.0, 260.0), (400.0, 100.0, 440.0, 300.0)
    # pose_features 열 배치: xy 34 + conf 17 + [h/w, w/h] + torso 3 -> h/w=51, w/h=52
    assert pose_features(lying, kpt, 0.3).shape == (56,) and pose_features(lying, kpt, 0.3)[52] > 1.0
    fps = 3.0
    window = [(i, [(lying, kpt), (stand, kpt)]) for i in range(12)]            # 한 사람은 계속 누움, 한 사람은 서 있음
    assert score_window(window, Fake(), fps, 3.0, 0.399, 0.3) == 5.0           # 3초 유지 -> 점수 성립
    short = [(i, [(lying, kpt)] if i < 6 else []) for i in range(12)]          # 누움이 2초 미만만 이어짐
    assert score_window(short, Fake(), fps, 3.0, 0.399, 0.3) == float("-inf")
    moving = [(i, [((100.0 + 120 * i, 200.0, 300.0 + 120 * i, 260.0), kpt)]) for i in range(12)]  # 자리가 계속 바뀜
    assert score_window(moving, Fake(), fps, 1.0, 0.399, 0.3) == float("-inf")
    assert score_window([(i, []) for i in range(12)], Fake(), fps, 1.0, 0.399, 0.3) == float("-inf")
    print("ok")
