"""영상 fps에 상관없이 정확히 3fps 표본(스펙 01 §0-3)으로 읽기 위한 공통 보조.

AI Hub 영상은 대부분 3fps지만 구매행동 반품은 10fps라서 프레임을 전부 읽으면 모든 시간(초) 계산이 틀어진다.
표본 k는 시각 k/PERCEPTION_FPS 초 이후 처음 나오는 원본 프레임이다(정수 간격으로 건너뛰면 10fps에서 3.33fps가 되어 시간이 어긋난다).
"""
from __future__ import annotations

import math

from config.config import PERCEPTION_FPS


def is_sample(index: int, native_fps: float) -> bool:
    """원본 프레임 index가 표본이면 True."""
    rate = PERCEPTION_FPS / (native_fps or PERCEPTION_FPS)
    return index == 0 or math.floor(index * rate + 1e-9) != math.floor((index - 1) * rate + 1e-9)


def native_index(sample: int, native_fps: float) -> int:
    """표본 번호 -> 원본 프레임 위치."""
    return math.ceil(sample * (native_fps or PERCEPTION_FPS) / PERCEPTION_FPS - 1e-9)


if __name__ == "__main__":
    assert all(is_sample(i, 3.0) for i in range(30))                       # 3fps는 전부
    picked = [i for i in range(600) if is_sample(i, 10.0)]
    assert len(picked) == 180 and picked[:4] == [0, 4, 7, 10], (len(picked), picked[:4])  # 60초 10fps -> 180표본
    assert all(native_index(k, 10.0) == picked[k] for k in range(180))     # 표본 번호와 원본 위치가 일치
    print("ok")
