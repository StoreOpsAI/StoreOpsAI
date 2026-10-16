"""행동 판정에 쓸 사람 고르기: 전신이 보이는 사람만, 많이 겹친 박스는 한 명으로, 큰 순서로 상위 K명.

추적 번호는 동작을 자르기 위한 보조 수단이라 번호가 합쳐져도 괜찮다(프로젝트 목표: 행동 인식, 인물 재식별 아님).
"""
from __future__ import annotations

from typing import Sequence

Detection = tuple[int, int, int, int, int, float]   # (추적 번호, x1, y1, x2, y2, 신뢰도)


def _area(d: Detection) -> int:
    return max(0, d[3] - d[1]) * max(0, d[4] - d[2])


def _iou(a: Detection, b: Detection) -> float:
    w = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[4], b[4]) - max(a[2], b[2])
    inter = max(0, w) * max(0, h)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def select_people(dets: Sequence[Detection], frame_shape: Sequence[int], edge_margin: float,
                  merge_iou: float, top_k: int, min_conf: float = 0.0) -> list[Detection]:
    h, w = frame_shape[:2]
    dets = [d for d in dets if d[5] >= min_conf]
    mx, my = w * edge_margin, h * edge_margin
    full = [d for d in dets if d[1] > mx and d[2] > my and d[3] < w - mx and d[4] < h - my]   # 화면 가장자리에 안 닿음 = 전신
    parent = list(range(len(full)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            if _iou(full[i], full[j]) >= merge_iou:
                parent[find(i)] = find(j)
    groups: dict[int, list[Detection]] = {}
    for i, d in enumerate(full):
        groups.setdefault(find(i), []).append(d)
    merged = []
    for members in groups.values():
        biggest = max(members, key=_area)
        merged.append((min(m[0] for m in members), *biggest[1:5], max(m[5] for m in members)))
    return sorted(merged, key=_area, reverse=True)[:top_k]


def _self_check() -> None:
    shape = (1080, 1920, 3)
    a, b = (1, 100, 100, 300, 500, 0.9), (2, 105, 110, 305, 505, 0.8)    # 거의 같은 자리 = 한 명
    c = (3, 800, 200, 1000, 700, 0.9)
    cut = (4, 0, 300, 150, 900, 0.95)                                    # 왼쪽 가장자리에 닿음 = 전신 아님
    out = select_people([a, b, c, cut], shape, 0.01, 0.7, 3)
    assert sorted(d[0] for d in out) == [1, 3], out
    many = [(i, 100 + i * 250, 100, 300 + i * 250, 600, 0.9) for i in range(6)]
    assert len(select_people(many, shape, 0.01, 0.7, 3)) == 3
    print("ok")


if __name__ == "__main__":
    _self_check()
