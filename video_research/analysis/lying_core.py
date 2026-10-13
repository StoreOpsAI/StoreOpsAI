"""전도 판정 핵심(팀원 배포용, 이 폴더만으로 동작). 관절 특징 → MLP 누움 점수 → 같은 자리 N초 유지 경보.

storeops_ai_fr_evt/pipelines/lying_pose_dataset.pose_features, models/fall_detector.lying_alarms와 같은 계산이다
(analysis/check_parity.py가 두 구현이 같은 값을 내는지 검사한다).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pose_features(box, kpt, kpt_conf=0.3):
    """관절 17개를 사람 박스 중심·큰 변 기준으로 정규화한 56차원 특징(박스 위치는 넣지 않는다)."""
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


class LyingMLP:
    """lying_mlp.pt를 읽어 누움 점수(0~1, 확률이 아님)를 낸다."""

    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        layers, size = [], blob["n_in"]
        for h in blob["hidden"]:
            layers += [nn.Linear(size, h), nn.ReLU()]
            size = h
        self.net = nn.Sequential(*layers, nn.Linear(size, 2))
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.mean, self.std = blob["mean"], blob["std"]

    @torch.no_grad()
    def __call__(self, features):
        z = torch.tensor((np.asarray(features, np.float32) - self.mean) / self.std, dtype=torch.float32)
        return self.net(z).softmax(1)[:, 1].numpy()


class LyingTracker:
    """프레임을 순서대로 받아, 같은 자리(IoU >= same_spot_iou)의 누움이 hold_sec 이상 이어지면 한 번 경보(과거 프레임만 사용).

    표본 gap_samples개까지 끊겨도 잇는다. 추적 번호 대신 박스 위치로 이어 붙이므로 사람이 바뀌어도 된다.
    """

    def __init__(self, fps=3.0, hold_sec=2.0, same_spot_iou=0.3, gap_samples=1):
        self.hold_frames, self.iou, self.gap, self.runs = hold_sec * fps, same_spot_iou, gap_samples, []

    def update(self, sample_no, lying_boxes):
        """lying_boxes: 이번 표본에서 누움으로 판정된 사람 박스 목록. 새로 난 경보 [(표본 번호, 박스)]를 돌려준다."""
        alarms = []
        for run in self.runs:
            run["misses"] += 1
        for box in lying_boxes:
            run = max(self.runs, key=lambda r: iou(r["box"], box), default=None)
            if run is not None and iou(run["box"], box) >= self.iou:
                run.update(box=box, last=sample_no, misses=0)
            else:
                run = {"box": box, "start": sample_no, "last": sample_no, "misses": 0, "alarmed": False}
                self.runs.append(run)
            if not run["alarmed"] and run["last"] - run["start"] >= self.hold_frames:
                run["alarmed"] = True
                alarms.append((sample_no, box))
        self.runs = [r for r in self.runs if r["misses"] <= self.gap]
        return alarms

    def progress(self):
        """지금 이어지고 있는 누움 중 가장 긴 것이 유지 시간(hold_sec)의 몇 %인지(0~1). 화면 게이지용."""
        return max((min(1.0, (r["last"] - r["start"]) / self.hold_frames) for r in self.runs), default=0.0)

