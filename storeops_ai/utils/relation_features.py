"""R 입력: 표본 프레임마다 사람-물건 관계 20채널(장면 14 + 손목 6). 연구 쪽 relation_windows.frame_channels와
wrist_relation.wrist_channels를 그대로 옮긴 것이다(값이 학습과 같아야 한다. 같음은 연구 함수와 맞대 확인함, 2026-10-19).

- 사람 박스: 사람 점수 0.6 이상 전부. 손목: 박스가 가장 큰 사람의 왼·오른 손목(신뢰도 < POSE_MIN_CONF면 거리 CLIP).
- 물건: 물건 탐지기 박스(점수 하한 0.10, 학습과 같음)와 점수.
- 장면 채널 일부(사람 속도, 사람 사라짐 등)는 직전 표본 프레임 상태가 필요해 RelationTracker가 이어서 계산한다.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

CHANNELS = ("n_persons", "n_objects", "max_obj_conf", "near_dist", "near_inside", "n_orphans", "orphan_conf", "person_speed",
            "near_obj_speed", "co_move", "obj_size_ratio", "person_gone", "person_new", "valid")
FAR, CLIP, POSE_MIN_CONF = 1.0, 5.0, 0.5
WRISTS = (9, 10)
DIM = len(CHANNELS) + 3 * len(WRISTS)


def _centers(box: np.ndarray) -> np.ndarray:
    return np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], axis=1)


def _point_box_distance(pt, box) -> float:
    dx = max(box[0] - pt[0], 0.0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0.0, pt[1] - box[3])
    return float(np.hypot(dx, dy))


def frame_channels(pb: np.ndarray, ob: np.ndarray, oc: np.ndarray, prev: Optional[dict]) -> Tuple[np.ndarray, dict]:
    v = np.zeros(len(CHANNELS), np.float32)
    v[0], v[1], v[13] = len(pb), len(ob), 1.0
    v[2] = float(oc.max()) if len(oc) else 0.0
    height = float(np.median(pb[:, 3] - pb[:, 1])) if len(pb) else 0.0
    pc = _centers(pb) if len(pb) else np.zeros((0, 2))
    state = {"pc": pc, "oc": _centers(ob) if len(ob) else np.zeros((0, 2)), "h": height, "n": len(pb)}
    if prev is not None:
        v[11], v[12] = float(len(pb) < prev["n"]), float(len(pb) > prev["n"])
        if len(pc) and len(prev["pc"]) and height > 0:
            v[7] = float(np.median([np.min(np.hypot(*(prev["pc"] - c).T)) for c in pc]) / height)
    near_obj = None
    if len(pb) and len(ob) and height > 0:
        dist = np.array([[_point_box_distance(c, b) for b in pb] for c in state["oc"]]) / height
        nearest = dist.min(1)
        k = int(nearest.argmin())
        v[3], v[4] = nearest[k], float(nearest[k] == 0.0)
        far = nearest > FAR
        v[5], v[6] = float(far.sum()), float(oc[far].max()) if far.any() else 0.0
        v[10] = float((ob[k, 3] - ob[k, 1]) / height)
        near_obj = k
    elif len(ob):
        v[5], v[6], v[3] = float(len(ob)), float(oc.max()), CLIP
    if near_obj is not None and prev is not None and len(prev["oc"]) and height > 0:
        d = np.hypot(*(prev["oc"] - state["oc"][near_obj]).T)
        j = int(d.argmin())
        if d[j] <= height:
            v[8] = d[j] / height
            pj = int(np.argmin(np.hypot(*(pc - state["oc"][near_obj]).T)))
            pprev = np.hypot(*(prev["pc"] - pc[pj]).T) if len(prev["pc"]) else np.array([0.0])
            v[9] = abs(v[8] - float(pprev.min()) / height)
    state["near"] = near_obj
    return np.clip(v, -CLIP, CLIP), state


def wrist_channels(kpt: np.ndarray, box: Sequence[float], ob: np.ndarray, oc: np.ndarray) -> np.ndarray:
    v = np.zeros(3 * len(WRISTS), np.float32)
    h = max(float(box[3] - box[1]), 1.0)
    for i, w in enumerate(WRISTS):
        v[3 * i] = CLIP
        if kpt[w, 2] < POSE_MIN_CONF or not len(ob):
            continue
        d = np.array([_point_box_distance(kpt[w, :2], b) for b in ob]) / h
        k = int(d.argmin())
        v[3 * i:3 * i + 3] = min(d[k], CLIP), oc[k], float(d[k] == 0.0)
    return v


class RelationTracker:
    """표본 프레임마다 step()을 한 번 부른다(3fps). 직전 표본 상태를 이어 장면 채널을 계산한다."""

    def __init__(self) -> None:
        self._prev: Optional[dict] = None

    def step(self, person_boxes: np.ndarray, largest: Optional[Tuple[np.ndarray, Sequence[float]]],
             obj_boxes: np.ndarray, obj_confs: np.ndarray) -> np.ndarray:
        out = np.zeros(DIM, np.float32)
        pb = np.asarray(person_boxes, np.float32).reshape(-1, 4)
        ob, oc = np.asarray(obj_boxes, np.float32).reshape(-1, 4), np.asarray(obj_confs, np.float32).reshape(-1)
        out[:len(CHANNELS)], self._prev = frame_channels(pb, ob, oc, self._prev)
        if largest is not None:
            out[len(CHANNELS):] = wrist_channels(np.asarray(largest[0], np.float32), largest[1], ob, oc)
        return out


def _self_check() -> None:
    kpt = np.zeros((17, 3), np.float32); kpt[9] = [50, 50, 0.9]; kpt[10] = [0, 0, 0.1]
    v = wrist_channels(kpt, [0, 0, 40, 100], np.array([[40, 40, 60, 60], [200, 0, 210, 10]], np.float32), np.array([0.7, 0.9], np.float32))
    assert v[0] == 0 and abs(v[1] - 0.7) < 1e-6 and v[2] == 1 and v[3] == CLIP and v[4] == 0   # 연구 쪽 점검과 같은 값
    t = RelationTracker()
    a = t.step(np.array([[0, 0, 40, 100]]), (kpt, [0, 0, 40, 100]), np.array([[40, 40, 60, 60]]), np.array([0.7]))
    b = t.step(np.zeros((0, 4)), None, np.array([[40, 40, 60, 60]]), np.array([0.7]))
    assert a.shape == (DIM,) == (20,) and a[0] == 1 and abs(a[3] - 0.1) < 1e-6 and a[14] == 0 and b[11] == 1 and b[5] == 1 and b[14] == 0   # 사람이 떠나면 사라짐·남겨진 물건
    print("ok")


if __name__ == "__main__":
    _self_check()
