"""BehaviorRouter: 쓰러짐 점수는 우리 전도 모델(포즈)로, 서비스 판정 규칙(decide)은 그대로."""
import numpy as np

from config.config import CATEGORY_THRESHOLDS, FALL_LOGIT_THRESHOLD
from config.config import FALL_S3D_THRESHOLD
from models.behavior_router import BehaviorRouter, fall_service_score, fuse_fall, risk_level, s3d_service_score
from models.pose_fall import pose_features

KPT = np.zeros((17, 3), np.float32)
LYING, STANDING = (100.0, 200.0, 300.0, 260.0), (400.0, 100.0, 440.0, 300.0)


class FakeScorer:
    def logits(self, feats):  # 박스가 가로로 넓으면(w/h > 1) 누움
        return np.where(feats[:, 52] > 1.0, FALL_LOGIT_THRESHOLD + 3.0, -5.0)


def window(box, n=12):
    return [(i, [(box, KPT)]) for i in range(n)]


def test_service_score_is_half_at_threshold():
    assert abs(fall_service_score(FALL_LOGIT_THRESHOLD) - 0.5) < 1e-9
    assert fall_service_score(float("-inf")) == 0.0
    assert fall_service_score(FALL_LOGIT_THRESHOLD + 5) > CATEGORY_THRESHOLDS["쓰러짐"]


def test_risk_level_rises_with_logit():
    assert risk_level(-1.0) == "낮음" and risk_level(float("-inf")) == "낮음"
    assert risk_level(2.2) == "주의" and risk_level(4.5) == "높음" and risk_level(9.0) == "매우 높음"


def test_sustained_lying_becomes_fall_event():
    router = BehaviorRouter(weight_path=None, fall_scorer=FakeScorer())
    assert router.is_ready
    category, confidence, scores = router.predict([], pose_window=window(LYING))
    assert category == "쓰러짐" and confidence > CATEGORY_THRESHOLDS["쓰러짐"]
    assert router.exceeds_threshold(category, confidence)
    assert router.last_fall["risk"] != "낮음"


def test_standing_or_short_lying_is_not_event():
    router = BehaviorRouter(weight_path=None, fall_scorer=FakeScorer())
    for pose in (window(STANDING), [(i, [(LYING, KPT)] if i < 5 else []) for i in range(12)]):
        category, confidence, scores = router.predict([], pose_window=pose)
        assert category != "쓰러짐" and scores["쓰러짐"] < CATEGORY_THRESHOLDS["쓰러짐"]


def test_without_pose_window_other_scores_unchanged():
    router = BehaviorRouter(weight_path=None, fall_scorer=FakeScorer())
    category, confidence, scores = router.predict([], pose_window=None)
    assert all(v == 0.0 for v in scores.values())


def test_feature_shape():
    assert pose_features(LYING, KPT, 0.3).shape == (56,)


def test_s3d_score_is_half_at_its_threshold():
    assert abs(s3d_service_score(FALL_S3D_THRESHOLD) - 0.5) < 1e-9
    assert s3d_service_score(0.99) > 0.5 > s3d_service_score(0.1)


def test_fusion_modes():
    assert fuse_fall(0.9, 0.2, "and") == 0.2          # 둘 다 넘어야: 낮은 쪽이 결정
    assert fuse_fall(0.9, 0.8, "and") == 0.8
    assert fuse_fall(0.9, 0.2, "pose") == 0.2 and fuse_fall(0.9, 0.2, "s3d") == 0.9
    assert fuse_fall(None, 0.8, "and") == 0.8 and fuse_fall(0.7, None, "and") == 0.7  # 한쪽 증거가 없으면 있는 쪽만
    assert fuse_fall(None, None, "and") == 0.0


def test_and_gate_needs_both_evidences():
    class FakeS3D(BehaviorRouter):
        def __init__(self, p):
            super().__init__(weight_path=None, fall_scorer=FakeScorer())
            self.model, self._p = object(), p

        def predict(self, clip_frames, pose_window=None):  # 기존 분류기 predict를 대신하는 가짜 S3D 점수
            import models.behavior_router as m
            real = m.I3DClassifier.predict
            m.I3DClassifier.predict = lambda self_, frames: (None, None, {c: (self._p if c == "쓰러짐" else 0.0) for c in m.CATEGORIES})
            try:
                return BehaviorRouter.predict(self, clip_frames, pose_window)
            finally:
                m.I3DClassifier.predict = real

    frames = [object()] * 12
    hit = FakeS3D(0.95).predict(frames, window(LYING))
    assert hit[0] == "쓰러짐"                               # S3D도 포즈도 쓰러짐
    miss_pose = FakeS3D(0.95).predict(frames, window(STANDING))
    assert miss_pose[0] != "쓰러짐"                         # S3D만 쓰러짐 -> 게이트가 막는다
    miss_s3d = FakeS3D(0.05).predict(frames, window(LYING))
    assert miss_s3d[0] != "쓰러짐"                          # 포즈만 쓰러짐 -> 게이트가 막는다
