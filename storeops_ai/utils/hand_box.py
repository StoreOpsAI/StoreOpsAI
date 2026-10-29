"""OpenPose 손 상자: 몸 관절(어깨·팔꿈치·손목)로 손 영역을 정한다.

근거: Simon 외, "Hand Keypoint Detection in Single Images using Multiview Bootstrapping", CVPR 2017 (OpenPose handDetector.cpp에서 확인).
중심 = 손목 + 0.33 x (손목 - 팔꿈치), 한 변 = 1.5 x max(손목-팔꿈치, 0.9 x 팔꿈치-어깨). 연구 쪽 storeops_ai_fr_evt/pipelines/relation_v2.hand_box와 같은 식.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

SIDES = ((5, 7, 9), (6, 8, 10))   # COCO (어깨, 팔꿈치, 손목) 왼·오른


def hand_box(xy: Sequence[Sequence[float]], cf: Sequence[float], side: Tuple[int, int, int], min_conf: float) -> Optional[Tuple[float, float, float]]:
    """(중심 x, 중심 y, 반 변). 세 관절 중 하나라도 신뢰도 미달이면 None."""
    s, e, w = side
    if min(cf[s], cf[e], cf[w]) < min_conf:
        return None
    fore = math.hypot(xy[w][0] - xy[e][0], xy[w][1] - xy[e][1])
    upper = math.hypot(xy[e][0] - xy[s][0], xy[e][1] - xy[s][1])
    half = 0.75 * max(fore, 0.9 * upper)
    return xy[w][0] + 0.33 * (xy[w][0] - xy[e][0]), xy[w][1] + 0.33 * (xy[w][1] - xy[e][1]), half


def reach_boxes(hands: Sequence[Tuple[int, float, float, float]], people: Sequence[Sequence[float]]) -> list:
    """주인 후보 상자 (번호, x1, y1, x2, y2): 손 상자 + 사람 박스.
    Lin 외 TIFS 2015는 물건이 멈추기 직전 그 자리의 사람 전경을 주인으로 본다. 손만 보면 들고 나가는 절도에서 놓치므로 몸 박스도 넣는다."""
    return [(tid, x - h, y - h, x + h, y + h) for tid, x, y, h in hands] + [tuple(p[:5]) for p in people]


def box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """두 박스 (x1, y1, x2, y2) 사이 빈 거리(닿거나 겹치면 0)."""
    return math.hypot(max(b[0] - a[2], 0.0, a[0] - b[2]), max(b[1] - a[3], 0.0, a[1] - b[3]))


def _self_check() -> None:
    xy = [[0, 0]] * 17
    xy = [list(p) for p in xy]
    xy[5], xy[7], xy[9] = [100, 100], [100, 160], [100, 200]
    cf = [0.0] * 17
    cf[5] = cf[7] = cf[9] = 0.9
    cx, cy, half = hand_box(xy, cf, SIDES[0], 0.3)
    assert abs(cx - 100) < 1e-6 and abs(cy - 213.2) < 1e-6 and abs(half - 40.5) < 1e-6   # 연구 쪽 점검과 같은 값
    assert hand_box(xy, cf, SIDES[1], 0.3) is None
    assert reach_boxes([(1, 10, 10, 2)], [(2, 0, 0, 100, 200)]) == [(1, 8, 8, 12, 12), (2, 0, 0, 100, 200)]
    assert box_gap((0, 0, 10, 10), (10, 5, 20, 20)) == 0 and box_gap((0, 0, 10, 10), (13, 14, 20, 20)) == 5
    print("ok")


if __name__ == "__main__":
    _self_check()
