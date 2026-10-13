"""행동 분류기 자리: 기존 I3DClassifier와 같은 predict() 모양을 유지하고, 쓰러짐 점수만 우리 전도 모델(포즈)로 채운다.

- S3D 가중치가 있으면 모든 카테고리의 기본 점수는 S3D 분류기가 낸다(사용자 새 S3D 모델이 들어오는 자리). 없으면 0.0.
- 쓰러짐은 S3D 점수와 포즈 증거를 함께 쓴다(config FALL_FUSION: and=둘 다 기준 이상일 때만 / pose / s3d). 단순 평균은 쓰지 않는다.
  포즈 증거: 창의 관절 표본 -> pose_fall.score_window(지속 누움 로짓 L) -> 서비스 점수 sigmoid((L - 기준)/T), 기준에서 정확히 0.5.
  S3D 증거: 확률 p -> sigmoid(logit(p) - logit(FALL_S3D_THRESHOLD)), 기준에서 정확히 0.5. 둘 다 0.5를 넘어야 and가 성립(min).
  결합 방식과 S3D 기준은 OOF 비교로 정한다(지금 값은 TEMPORARY_DEFAULT).
- 위험 단계는 로짓 구간으로 표시한다(확률이 아님). 구간 경계는 config FALL_RISK_CUTS.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

from config.config import (
    CATEGORIES, FALL_FUSION, FALL_HOLD_SEC, FALL_KPT_CONF, FALL_LINK_IOU, FALL_LOGIT_THRESHOLD, FALL_POSE_MLP_PATH, FALL_RISK_CUTS,
    FALL_S3D_THRESHOLD, FALL_SCORE_TEMPERATURE, I3D_BUFFER_FPS,
)
from models.i3d_classifier import I3DClassifier
from models.pose_fall import PoseMLP, Sample, score_window

logger = logging.getLogger("storeops_ai")

FALL_CATEGORY = "쓰러짐"
RISK_LABELS = ("낮음", "주의", "높음", "매우 높음")


def fall_service_score(logit: float) -> float:
    """지속 누움 로짓 -> 서비스 점수(0~1). 기준 로짓에서 정확히 0.5. 확률이 아니라 임계값 비교용 점수다."""
    if logit == float("-inf"):
        return 0.0
    return 1.0 / (1.0 + math.exp(-(logit - FALL_LOGIT_THRESHOLD) / FALL_SCORE_TEMPERATURE))


def s3d_service_score(p: float) -> float:
    """S3D 쓰러짐 확률 -> 기준(FALL_S3D_THRESHOLD)에서 정확히 0.5가 되는 점수."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    t = min(max(FALL_S3D_THRESHOLD, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) - math.log(t / (1 - t)))))


def fuse_fall(s3d_score: Optional[float], pose_score: Optional[float], mode: str = FALL_FUSION) -> float:
    """쓰러짐 증거 결합. 한쪽 증거가 없으면(S3D 가중치 없음, 관절 없음) 있는 쪽만 쓴다."""
    parts = {"s3d": s3d_score, "pose": pose_score}
    have = [v for v in parts.values() if v is not None]
    if not have:
        return 0.0
    if mode in parts:  # 단일 증거 모드
        return parts[mode] if parts[mode] is not None else have[0]
    if mode == "and":
        return min(have)  # 둘 다 0.5를 넘어야 0.5를 넘는다
    raise ValueError(f"알 수 없는 FALL_FUSION: {mode}")


def risk_level(logit: float, cuts: Sequence[float] = FALL_RISK_CUTS) -> str:
    """기준 이상일 때만 단계가 올라간다. cuts = (주의, 높음, 매우 높음) 로짓 경계(오름차순)."""
    if logit < cuts[0]:
        return RISK_LABELS[0]
    return RISK_LABELS[1 + sum(logit >= c for c in cuts[1:])]


class BehaviorRouter(I3DClassifier):
    def __init__(self, weight_path: Optional[str] = None, fall_scorer=None):
        super().__init__(weight_path)
        self._fall_scorer = fall_scorer
        if fall_scorer is None and FALL_POSE_MLP_PATH:
            try:
                self._fall_scorer = PoseMLP(FALL_POSE_MLP_PATH)
            except (OSError, KeyError, RuntimeError) as exc:  # 가중치가 없으면 쓰러짐 증거를 끄고 이유를 남긴다
                logger.warning("전도 포즈 모델 로드 실패(%s): %s - 쓰러짐은 기존 분류기 점수만 사용", FALL_POSE_MLP_PATH, exc)

    @property
    def is_ready(self) -> bool:
        return self.model is not None or self._fall_scorer is not None

    def predict(self, clip_frames: List, pose_window: Optional[Sequence[Tuple[int, List]]] = None) -> Tuple[str, float, Dict[str, float]]:
        """clip_frames: 기존과 같은 축소 프레임 목록. pose_window: [(표본 번호, [(박스, 관절)])] (3fps 표본)."""
        scores = {c: 0.0 for c in CATEGORIES}
        s3d_fall = None
        if self.model is not None and clip_frames:
            _, _, scores = super().predict(clip_frames)
            s3d_fall = s3d_service_score(scores[FALL_CATEGORY])
        pose_fall, logit = None, float("-inf")
        if self._fall_scorer is not None and pose_window:
            logit = score_window(pose_window, self._fall_scorer, I3D_BUFFER_FPS, FALL_HOLD_SEC, FALL_LINK_IOU, FALL_KPT_CONF)
            pose_fall = fall_service_score(logit)
        if s3d_fall is not None or pose_fall is not None:
            scores[FALL_CATEGORY] = fuse_fall(s3d_fall, pose_fall)
            self.last_fall = {"s3d": s3d_fall, "pose": pose_fall, "logit": logit, "risk": risk_level(logit), "mode": FALL_FUSION}
        return self.decide(scores)
