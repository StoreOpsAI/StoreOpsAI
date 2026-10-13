"""전도 관절 특징 증강(train 전용, AGENTS.md 규칙 3). 원본 박스·관절을 흔들어 특징을 다시 만든다. 라벨은 그대로.

- 좌우 반전(관절 좌우 교체): 카메라 방향·누운 방향이 한쪽으로 치우치는 것을 줄인다
- 작은 회전(±LYING_AUG_ROT_DEG): 카메라 기울기 차이 흉내
- 관절 흔들림(LYING_AUG_JITTER)·관절 가림(LYING_AUG_DROPOUT): 포즈 추정 오차·가림 흉내
- 검출 박스 크기 흔들림(LYING_AUG_BOX_SCALE)
평가(dev·val·test)와 배포 추론에는 쓰지 않는다. 시드 고정.
"""

from __future__ import annotations

import numpy as np

from config.config import LYING_AUG_BOX_SCALE, LYING_AUG_COPIES, LYING_AUG_DROPOUT, LYING_AUG_JITTER, LYING_AUG_ROT_DEG
from pipelines.lying_pose_dataset import pose_features

FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]  # COCO 좌우 관절


def augment(box, kpt, rng):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    k = kpt.copy()
    if rng.random() < 0.5:
        k[:, 0] = 2 * cx - k[:, 0]
        for a, b in FLIP_PAIRS:
            k[[a, b]] = k[[b, a]]
    angle = np.deg2rad(rng.uniform(-LYING_AUG_ROT_DEG, LYING_AUG_ROT_DEG))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    k[:, :2] = (k[:, :2] - [cx, cy]) @ rot.T + [cx, cy]
    corners = (np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) - [cx, cy]) @ rot.T + [cx, cy]
    w2, h2 = np.ptp(corners[:, 0]) * rng.uniform(1 - LYING_AUG_BOX_SCALE, 1 + LYING_AUG_BOX_SCALE), np.ptp(corners[:, 1]) * rng.uniform(1 - LYING_AUG_BOX_SCALE, 1 + LYING_AUG_BOX_SCALE)
    k[:, :2] += rng.normal(0, LYING_AUG_JITTER * max(w, h), (17, 2))
    k[rng.random(17) < LYING_AUG_DROPOUT, 2] = 0.0
    return (cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2), k


def augmented_features(clips, copies=LYING_AUG_COPIES, seed=0):
    """clips의 라벨 있는 사람마다 증강 복사본 `copies`개의 (특징, 라벨). 학습 fold의 클립에만 부른다."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for c in clips:
        for i in np.flatnonzero(c["label"] >= 0):
            for _ in range(copies):
                feats.append(pose_features(*augment(c["box"][i], c["kpt"][i], rng)))
                labels.append(c["label"][i])
    return np.array(feats, np.float32).reshape(len(feats), -1), np.array(labels, np.int64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    box = (100.0, 200.0, 300.0, 260.0)  # 누운 사람처럼 넓은 박스
    kpt = np.zeros((17, 3), np.float32)
    kpt[:, 0] = np.linspace(110, 290, 17)
    kpt[:, 1], kpt[:, 2] = 230, 0.9
    for _ in range(100):
        b, k = augment(box, kpt, rng)
        assert b[2] > b[0] and b[3] > b[1] and np.isfinite(k).all()
        assert pose_features(b, k).shape == (56,)
    assert not np.array_equal(kpt, augment(box, kpt, rng)[1])  # 원본은 그대로, 복사본은 달라진다
    print("ok")
