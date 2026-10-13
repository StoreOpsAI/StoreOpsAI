"""FR-EVT-05: I3D 분류 인터페이스.

현재 단계에서는 실제 I3D 모델/가중치/학습을 넣지 않는다.
대신 7개 클래스의 점수 형식 검증과 FR-EVT-06으로 넘길 수 있는 인터페이스를 구현한다.
실제 I3D가 연결되면 predict() 내부만 교체하면 된다.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from config.config import CATEGORIES, CATEGORY_THRESHOLDS


# 실제 I3D 모델이 아직 연결되지 않았다는 것을 나타내는 예외 클래스
class I3DNotConfiguredError(RuntimeError):
    pass


class I3DClassifier:
    # I3D 분류기를 초기화하는 부분
    # 현재는 실제 모델을 불러오지 않고 구조만 만들어 둔다.
    def __init__(self, weight_path: str | None = None):
        self.weight_path = weight_path
        self.categories = list(CATEGORIES)

        # 실제 I3D 모델이 연결되면 이 변수에 모델을 저장한다.
        self.model = None


    # I3D에서 출력되는 7개 클래스의 점수가
    # 올바른 형식인지 확인하는 함수
    @classmethod
    def validate_scores(cls, scores: Dict[str, float]) -> Dict[str, float]:

        # 점수 데이터가 딕셔너리 형태인지 확인
        if not isinstance(scores, dict):
            raise ValueError("I3D scores는 dict여야 합니다.")

        # 정상, 전도, 절도, 유기, 방화, 파손, 폭행 중
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


    # 실제 I3D 모델이 동작하는 위치
    def predict(
        self,
        clip_frames: List[np.ndarray]
    ) -> Tuple[str, float, Dict[str, float]]:

        # 입력 영상 프레임이 없는 경우 오류 발생
        if not clip_frames:
            raise ValueError(
                "I3D 입력 clip_frames가 비어 있습니다."
            )

        # 현재 프로젝트에서는 실제 I3D 모델을 연결하지 않았기 때문에
        # 여기서 실행을 중단하고 미설정 상태임을 알려준다.
        raise I3DNotConfiguredError(
            "FR-EVT-05 기능 인터페이스만 구현된 상태입니다. "
            "실제 I3D 모델/학습은 아직 연결하지 않았습니다."
        )


    # 나중에 I3D 모델이 반환한 7개 점수를
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


    # FR-EVT-06에서 사용하는 임계값 확인 함수
    def exceeds_threshold(
        self,
        category: str,
        confidence: float
    ) -> bool:

        # 정상은 이벤트 후보로 만들지 않는다.
        if category == "정상":
            return False

        # 해당 이벤트 종류의 설정된 임계값과
        # 현재 confidence를 비교한다.
        return float(confidence) >= CATEGORY_THRESHOLDS[category]
