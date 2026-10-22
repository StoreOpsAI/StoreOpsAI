"""전도 확인: 같은 자리에 누운 사람이 일정 시간 이상 이어지는지 본다(사용자 지정 규칙, 2026-10-19).

- 누움: 어깨 중점->엉덩이 중점 선(몸통)이 수직에서 torso_deg 이상 기울었다. 발목이 확실히 보이면 엉덩이->발목 선(다리)도
  같은 기준을 넘어야 한다(허리를 깊이 숙인 사람은 몸통은 눕지만 다리가 서 있어 걸러진다). 쪼그려 앉기·무릎 꿇기는 몸통이 서 있다.
  키가 줄어드는 것만으로 판단하지 않는다(CLAUDE.md 전도 기준).
- 같은 자리: 추적 번호 대신 위치로 잇는다(번호가 바뀌어도 된다). 직전 누운 자리에서 박스 긴 변의 same_spot 배 안이면 이어진 것.
- 관절이 잠깐 안 잡혀도 gap_sec 안이면 유지로 본다.
- 각도·간격 값은 논문에 없는 설정값이다(근거 약함, 재검토 대상).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

SHOULDERS, HIPS, ANKLES = (5, 6), (11, 12), (15, 16)


def _axis_deg(a: np.ndarray, b: np.ndarray) -> float:
    """a->b 선이 수직에서 기운 각도(0=수직, 90=수평)."""
    d = b - a
    return float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-9)))


def is_lying(kpt: np.ndarray, torso_deg: float, kpt_conf: float) -> Optional[bool]:
    """kpt (17,3) = x, y, 신뢰도. 어깨·엉덩이가 확실하지 않으면 None(판단 안 함)."""
    if (kpt[list(SHOULDERS + HIPS), 2] < kpt_conf).any():
        return None
    sh, hip = kpt[list(SHOULDERS), :2].mean(0), kpt[list(HIPS), :2].mean(0)
    if _axis_deg(sh, hip) < torso_deg:
        return False
    if (kpt[list(ANKLES), 2] >= kpt_conf).all():
        return _axis_deg(hip, kpt[list(ANKLES), :2].mean(0)) >= torso_deg
    return True


class LyingGate:
    def __init__(self, hold_sec: float, torso_deg: float, kpt_conf: float, gap_sec: float, same_spot: float):
        self.hold_sec, self.torso_deg, self.kpt_conf = hold_sec, torso_deg, kpt_conf
        self.gap_sec, self.same_spot = gap_sec, same_spot
        self.since: Optional[float] = None          # 지금 자리에서 누움이 시작된 시각
        self.last: Optional[float] = None           # 마지막으로 누움이 보인 시각
        self.spot: Optional[Tuple[float, float, float]] = None   # (중심 x, y, 박스 긴 변)

    def update(self, t: float, persons: Sequence[Tuple[np.ndarray, Sequence[float]]]) -> None:
        """persons: 이 프레임의 (관절 17x3, 박스 x1,y1,x2,y2)."""
        lying = []
        for kpt, box in persons:
            if is_lying(np.asarray(kpt, np.float32), self.torso_deg, self.kpt_conf):
                lying.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1])))
        if self.since is not None and self.last is not None and t - self.last > self.gap_sec:
            self.since = self.spot = None                                         # 끊긴 지 오래면 새로 시작
        if not lying:
            return
        if self.spot is not None:
            near = min(lying, key=lambda p: (p[0] - self.spot[0]) ** 2 + (p[1] - self.spot[1]) ** 2)
            if ((near[0] - self.spot[0]) ** 2 + (near[1] - self.spot[1]) ** 2) ** 0.5 <= self.same_spot * max(self.spot[2], near[2]):
                self.spot, self.last = near, t
                return
        self.spot, self.since, self.last = max(lying, key=lambda p: p[2]), t, t   # 새 자리(가장 큰 사람)

    def held(self, t: float) -> bool:
        return self.since is not None and self.last is not None and t - self.last <= self.gap_sec and self.last - self.since >= self.hold_sec


def _self_check() -> None:
    def person(lying: bool, x: float = 500.0, bend: bool = False, ankles: bool = True):
        k = np.zeros((17, 3), np.float32)
        if lying:
            pts = {5: (x, 400), 6: (x, 410), 11: (x + 100, 405), 12: (x + 100, 415), 15: (x + 200, 405), 16: (x + 200, 415)}
            box = (x - 20, 380, x + 220, 440)
        else:
            legs = (15, 16) if not bend else (15, 16)
            pts = {5: (x, 200), 6: (x + 20, 200), 11: (x, 300), 12: (x + 20, 300), 15: (x, 450), 16: (x + 20, 450)}
            if bend:   # 허리를 깊이 숙임: 몸통은 수평, 다리는 수직
                pts.update({5: (x - 100, 300), 6: (x - 100, 310)})
            box = (x - 120, 180, x + 40, 460)
        for i, (px, py) in pts.items():
            if ankles or i not in (15, 16):
                k[i] = (px, py, 0.9)
        return k, box

    assert is_lying(person(True)[0], 60, 0.5) and not is_lying(person(False)[0], 60, 0.5)
    assert not is_lying(person(False, bend=True)[0], 60, 0.5)                       # 깊이 숙임은 다리가 서 있어 누움이 아니다
    assert is_lying(person(True, ankles=False)[0], 60, 0.5)                         # 발목이 안 보이면 몸통만 본다
    k = person(True)[0]; k[5, 2] = 0.1
    assert is_lying(k, 60, 0.5) is None                                             # 관절이 확실하지 않으면 판단하지 않는다
    g = LyingGate(3.0, 60, 0.5, 1.0, 0.5)
    for i in range(20):                                                             # 0~1.9초 서 있음
        g.update(i * 0.1, [person(False)])
    for i in range(20, 60):                                                         # 2.0~5.9초 누움
        g.update(i * 0.1, [person(True, x=500 + (i % 3))])
        if i * 0.1 < 4.9:
            assert not g.held(i * 0.1), i
    assert g.held(5.9)
    g.update(7.5, [])
    assert not g.held(7.5)                                                          # 1초 넘게 안 보이면 풀린다
    g2 = LyingGate(3.0, 60, 0.5, 1.0, 0.5)
    for i in range(40):                                                             # 누운 자리가 계속 크게 바뀌면 이어지지 않는다
        g2.update(i * 0.1, [person(True, x=100 + 400 * (i % 2))])
    assert not g2.held(3.9)
    print("ok")


if __name__ == "__main__":
    _self_check()
