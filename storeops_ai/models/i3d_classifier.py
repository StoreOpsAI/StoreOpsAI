"""FR-EVT-05: 행동 분류 (I3D 계열 - torchvision S3D).

training/runs/*/best.pt (training/scripts/train_i3d.py 로 학습한 체크포인트)를 읽어서
클립(프레임 리스트)을 분류하고, 서비스 표준 7개 카테고리 점수로 변환한다.

학습 때와 반드시 같아야 하는 전처리 (training/scripts/train_i3d.py::load_clip):
  1) 프레임 수를 체크포인트의 num_frames(24)로 균일 샘플링
  2) 프레임마다 (size, size) 정사각형으로 리사이즈 (종횡비 유지 안 함, INTER_LINEAR)
  3) BGR -> RGB, /255, Kinetics-400 mean/std 정규화
  4) (C, T, H, W) 텐서, 배치 차원 추가

학습 라벨은 영문 6종(abandon/broken/fall/fire/normal/theft)이고 '폭행(fight)'은 학습하지 않았다.
서비스 카테고리 7종과 맞추기 위해 폭행 점수는 0.0으로 채운다 (=이 모델로는 폭행이 발급되지 않는다).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import (
    CATEGORIES, CATEGORY_THRESHOLDS, I3D_WEIGHT_PATH, I3D_DEVICE, MODEL_LABEL_TO_CATEGORY,
    NORMAL_ANOMALY_MARGIN,
)

logger = logging.getLogger("storeops_ai")

# Kinetics-400 정규화 상수. training/scripts/train_i3d.py 의 MEAN/STD 와 동일해야 한다.
_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

# (절대경로, device) -> 로드된 모델. API가 요청마다 파이프라인을 새로 만들어도 30MB 가중치를 매번 읽지 않는다.
_MODEL_CACHE: Dict[Tuple[str, str], dict] = {}
_CACHE_LOCK = threading.Lock()


# 실제 I3D 모델이 연결되지 않았다는 것을 나타내는 예외 클래스
class I3DNotConfiguredError(RuntimeError):
    pass


def _resolve_device(name: str) -> str:
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("I3D_DEVICE=cuda 이지만 CUDA를 사용할 수 없습니다.")
    return name


def _build_model(arch: str, n_classes: int):
    """train_i3d.py 와 같은 방식으로 모델 뼈대를 만든다 (사전학습 가중치는 체크포인트가 덮어쓰므로 받지 않는다)."""
    import torch.nn as nn
    import torchvision

    builder = getattr(torchvision.models.video, arch)
    model = builder(weights=None)
    if arch == "s3d":
        in_ch = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv3d(in_ch, n_classes, kernel_size=1, stride=1)
    else:
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def _load_checkpoint(path: Path, device_name: str) -> dict:
    import torch

    key = (str(path.resolve()), device_name)
    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        label_map: Dict[str, int] = ckpt["label_map"]
        arch = ckpt["arch"]

        unknown = [n for n in label_map if n not in MODEL_LABEL_TO_CATEGORY]
        if unknown:
            raise ValueError(
                f"체크포인트 라벨 {unknown} 이(가) config.MODEL_LABEL_TO_CATEGORY 에 없습니다.")

        model = _build_model(arch, len(label_map))
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device_name).eval()

        # 출력 인덱스 -> 서비스 카테고리(한글)
        index_to_category = {idx: MODEL_LABEL_TO_CATEGORY[name] for name, idx in label_map.items()}

        loaded = {
            "model": model,
            "device": device_name,
            "index_to_category": index_to_category,
            "num_frames": int(ckpt.get("num_frames", 24)),
            "size": int(ckpt.get("size", 224)),
            "arch": arch,
            "val_acc": ckpt.get("val_acc"),
            "epoch": ckpt.get("epoch"),
        }
        _MODEL_CACHE[key] = loaded
        return loaded


class I3DClassifier:
    # weight_path 를 명시하면 반드시 있어야 한다. 생략하면 config.I3D_WEIGHT_PATH 를 쓰고,
    # 그 파일도 없으면 '미설정' 상태로 남아 predict() 가 I3DNotConfiguredError 를 낸다.
    def __init__(self, weight_path: Optional[str] = None):
        explicit = weight_path is not None
        path = Path(weight_path or I3D_WEIGHT_PATH)

        self.weight_path = str(path)
        self.categories = list(CATEGORIES)
        self.model = None
        self.num_frames = 24
        self.input_size = 224
        self.meta: dict = {}
        self._device = "cpu"
        self._index_to_category: Dict[int, str] = {}
        self.missing_categories: List[str] = [c for c in CATEGORIES if c != "정상"]

        if path.is_file():
            self._load(path)
        elif explicit:
            raise FileNotFoundError(f"I3D 가중치 파일을 찾을 수 없습니다: {path}")
        else:
            logger.warning("I3D 가중치 없음(%s) - 행동분류 비활성 상태", path)

    def _load(self, path: Path):
        device_name = _resolve_device(I3D_DEVICE)
        loaded = _load_checkpoint(path, device_name)
        self.model = loaded["model"]
        self._device = loaded["device"]
        self._index_to_category = loaded["index_to_category"]
        self.num_frames = loaded["num_frames"]
        self.input_size = loaded["size"]
        self.meta = {k: loaded[k] for k in ("arch", "val_acc", "epoch", "device")}
        covered = set(self._index_to_category.values())
        self.missing_categories = [c for c in CATEGORIES if c not in covered and c != "정상"]
        logger.info("I3D 로드: %s arch=%s val_acc=%s device=%s / 모델이 학습하지 않은 카테고리: %s",
                    path, self.meta["arch"], self.meta["val_acc"], self.meta["device"],
                    self.missing_categories)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # I3D에서 출력되는 7개 클래스의 점수가
    # 올바른 형식인지 확인하는 함수
    @classmethod
    def validate_scores(cls, scores: Dict[str, float]) -> Dict[str, float]:

        # 점수 데이터가 딕셔너리 형태인지 확인
        if not isinstance(scores, dict):
            raise ValueError("I3D scores는 dict여야 합니다.")

        # 정상, 전도, 파손, 방화, 유기, 절도, 폭행 중
        # 빠진 클래스가 있는지 확인
        missing = [c for c in CATEGORIES if c not in scores]

        # 정의되지 않은 추가 클래스가 들어왔는지 확인
        extra = [k for k in scores if k not in CATEGORIES]

        # 7개 클래스가 정확하게 존재해야 함
        if missing or extra:
            raise ValueError(
                f"I3D 7개 클래스가 정확히 필요합니다. "
                f"missing={missing}, extra={extra}"
            )

        normalized = {}

        # 각 클래스의 점수를 확인
        for c in CATEGORIES:

            # 입력된 점수를 숫자로 변환
            value = float(scores[c])

            # 점수는 0~1 사이여야 함
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{c} score는 0~1 범위여야 합니다: {value}"
                )

            # 검증된 점수를 저장
            normalized[c] = value

        return normalized

    # ------------------------------------------------------------------ 전처리 / 추론

    def _preprocess(self, clip_frames: List[np.ndarray]):
        """BGR 프레임 리스트 -> (1, C, T, H, W) float32 텐서. 학습 전처리와 동일."""
        import cv2
        import torch

        n = len(clip_frames)
        # 학습 클립은 항상 num_frames 장이었다. 길이가 다르면 균일 샘플링(부족하면 반복)한다.
        idx = np.round(np.linspace(0, n - 1, self.num_frames)).astype(int)

        size = self.input_size
        arr = np.empty((self.num_frames, size, size, 3), dtype=np.float32)
        for t, i in enumerate(idx):
            frame = clip_frames[int(i)]
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("clip_frames 는 (H, W, 3) BGR 이미지 리스트여야 합니다.")
            img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr[t] = (img.astype(np.float32) / 255.0 - _MEAN) / _STD

        clip = np.ascontiguousarray(arr.transpose(3, 0, 1, 2))   # (C, T, H, W)
        return torch.from_numpy(clip).unsqueeze(0)

    # 실제 I3D 모델이 동작하는 위치
    def predict(
        self,
        clip_frames: List[np.ndarray]
    ) -> Tuple[str, float, Dict[str, float]]:
        """클립을 분류해 (카테고리, confidence, 7개 카테고리 점수)를 돌려준다."""
        import torch

        # 입력 영상 프레임이 없는 경우 오류 발생
        if not clip_frames:
            raise ValueError(
                "I3D 입력 clip_frames가 비어 있습니다."
            )

        if not self.is_ready:
            raise I3DNotConfiguredError(
                f"I3D 가중치가 로드되지 않았습니다: {self.weight_path}"
            )

        x = self._preprocess(clip_frames).to(self._device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy().astype(np.float64)

        # 모델이 학습하지 않은 카테고리(폭행)는 0.0. 정상 포함 7개가 항상 채워진다.
        scores = {c: 0.0 for c in CATEGORIES}
        for idx, p in enumerate(probs):
            scores[self._index_to_category[idx]] = float(min(max(p, 0.0), 1.0))

        return self.decide(scores)

    # 모델이 반환한 7개 점수를
    # 프로젝트에서 사용할 표준 결과로 변환하는 함수
    def result_from_scores(
        self,
        scores: Dict[str, float]
    ):

        # 입력된 7개 점수의 형식을 먼저 검증
        scores = self.validate_scores(scores)

        # 7개 클래스 중 가장 높은 점수를 가진 클래스를 선택
        category = max(scores, key=scores.get)

        # 선택된 클래스의 점수를 confidence로 사용
        return category, scores[category], scores

    def decide(self, scores: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """원점수를 사건 처리용 정상/비정상/판정 보류로 바꾼다."""
        raw_category, raw_confidence, scores = self.result_from_scores(scores)
        if raw_category == "정상":
            return raw_category, raw_confidence, scores
        if raw_confidence < CATEGORY_THRESHOLDS[raw_category]:
            return "판정 보류", raw_confidence, scores
        if raw_confidence - scores["정상"] < NORMAL_ANOMALY_MARGIN:
            return "판정 보류", raw_confidence, scores
        return raw_category, raw_confidence, scores

    # FR-EVT-06에서 사용하는 임계값 확인 함수
    def exceeds_threshold(
        self,
        category: str,
        confidence: float
    ) -> bool:

        # 정상은 이벤트 후보로 만들지 않는다.
        if category in ("정상", "판정 보류"):
            return False

        # 해당 이벤트 종류의 설정된 임계값과
        # 현재 confidence를 비교한다.
        return float(confidence) >= CATEGORY_THRESHOLDS[category]
