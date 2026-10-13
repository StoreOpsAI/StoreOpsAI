"""파인튜닝 물건 탐지기 박스와 사람 포즈(손목)를 합쳐, 사람별 시간 창의 '사람-물건 연결' 특징을 만든다.

물건 박스를 프레임 사이에 이어 물건 추적을 만들고, 손목(COCO 9·10번)과 물건 중심의 거리(사람 박스 높이 비율)로
쥠/놓음/정지/사람이 떠남을 본다. 출처(들고 온 물건 / 매장에서 집은 물건 / 원래 있던 물건) 단서도 같이 낸다:
  carried_at_entry = 사람이 나타난 직후 손목에 물건이 있었나, first_assoc_delay = 처음 물건을 쥐기까지 걸린 시간,
  pre_existing = 쥔 물건이 사람이 나타나기 전부터 있었나(매장에 있던 물건 후보).
특징은 모델 출력에서 계산한 입력일 뿐이고 라벨이 아니다. 출처 정답은 사람 검수 라벨이 따로 필요하다.
"""
from __future__ import annotations

import numpy as np

from config.config import (ABANDON_WIN_FRAMES, POSE_MIN_CONF, REL_DET_CONF, REL_ENTRY_FRAMES, REL_HOLD_DIST, REL_LINK_IOU,
                           REL_RELEASE_DIST, REL_STATIC_FRAMES, REL_STATIC_MOVE)
from models.new_object_detector import _iou

WRISTS = (9, 10)
FEATURES = ("held_frac", "held_first", "held_last", "n_obj_max", "min_dist_mean", "release", "static_len", "end_dist",
            "held_conf", "since_entry", "carried_at_entry", "first_assoc_delay", "pre_existing", "exit_gap", "near_frac")


def link_objects(dets_per_frame):
    """프레임별 [(박스, 점수)] -> 물건 추적 목록 [{frame: (박스, 점수)}]. 점수 하한 이상만, 최대 2프레임 끊김까지 잇는다."""
    tracks, active = [], []
    for f, dets in enumerate(dets_per_frame):
        dets = [d for d in dets if d[1] >= REL_DET_CONF]
        pairs = sorted(((_iou(t["last"], b), ti, di) for ti, t in enumerate(active) if f - t["frame"] <= 3
                        for di, (b, _) in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for iou, ti, di in pairs:
            if iou >= REL_LINK_IOU and ti not in used_t and di not in used_d:
                used_t.add(ti), used_d.add(di)
                active[ti].update(last=dets[di][0], frame=f)
                active[ti]["obj"][f] = dets[di]
        for di, d in enumerate(dets):
            if di not in used_d:
                t = {"last": d[0], "frame": f, "obj": {f: d}}
                active.append(t), tracks.append(t["obj"])
    return tracks


def _wrist_dist(seen_f, center):
    box, kpt = seen_f
    ok = [kpt[i, :2] for i in WRISTS if kpt[i, 2] >= POSE_MIN_CONF]
    if not ok:
        return float("inf")
    return float(min(np.hypot(*(w - center)) for w in ok) / max(box[3] - box[1], 1.0))


def _center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def frame_state(seen, objects):
    """프레임마다 {frame: (최소 거리, 쥔 물건 수, 쥔 물건 점수들, 쥔 물건 번호들)}."""
    state = {}
    for f, sf in seen.items():
        dists = [(_wrist_dist(sf, _center(o[f][0])), oi, o[f][1]) for oi, o in enumerate(objects) if f in o]
        held = [d for d in dists if d[0] <= REL_HOLD_DIST]
        state[f] = (min((d[0] for d in dists), default=float("inf")), len(held), [d[2] for d in held], [d[1] for d in held])
    return state


def _release(seen, objects, window, first, last):
    """창 앞 2/3에서 쥐었다가 이후 놓아(거리 > REL_RELEASE_DIST) 가만히 있는 물건이 있나 -> (release, static_len, end_dist)."""
    cut = first + (last - first) * 2 // 3
    best = (0.0, 0.0, 0.0)
    for obj in objects:
        frames = sorted(f for f in obj if f in seen and first <= f <= last)
        held_early = [f for f in frames if f <= cut and _wrist_dist(seen[f], _center(obj[f][0])) <= REL_HOLD_DIST]
        if len(held_early) < 2:
            continue
        after = [f for f in frames if f > held_early[-1] and _wrist_dist(seen[f], _center(obj[f][0])) > REL_RELEASE_DIST]
        if not after:
            continue
        h = max(seen[after[0]][0][3] - seen[after[0]][0][1], 1.0)
        moves = [np.hypot(*(_center(obj[f][0]) - _center(obj[after[0]][0]))) / h for f in after]
        if max(moves) <= REL_STATIC_MOVE * 3 and len(after) >= REL_STATIC_FRAMES:
            end_d = min(_wrist_dist(seen[after[-1]], _center(obj[after[-1]][0])), 4.0)
            best = max(best, (1.0, len(after) / ABANDON_WIN_FRAMES, end_d))
    return best


def window_features(seen, objects, start, track_first, track_last):
    """한 사람 추적(seen)의 창 [start, start+W) 특징 15개(FEATURES 순서)."""
    w = ABANDON_WIN_FRAMES
    state = frame_state(seen, objects)
    frames = [f for f in range(start, start + w) if f in state]
    held = np.array([state[f][1] > 0 for f in frames], float)
    third = max(len(frames) // 3, 1)
    mins = np.array([min(state[f][0], 2.0) for f in frames])
    release, static_len, end_dist = _release(seen, objects, range(start, start + w), start, start + w - 1)
    confs = [c for f in frames for c in state[f][2]]
    entry = [f for f in sorted(seen)[:REL_ENTRY_FRAMES] if f in state]
    held_all = sorted(f for f in state if state[f][1] > 0)
    first_held = held_all[0] if held_all else None
    pre = 0.0
    if first_held is not None:
        oi = state[first_held][3][0]
        pre = float(min(objects[oi]) < track_first)
    return np.array([
        held.mean(), held[:third].mean(), held[-third:].mean(), max((sum(1 for o in objects if f in o and state[f][0] <= 1.0) for f in frames), default=0),
        mins.mean(), release, static_len, end_dist, np.mean(confs) if confs else 0.0,
        min(max(start - track_first, 0) / w, 3.0), np.mean([state[f][1] > 0 for f in entry]) if entry else 0.0,
        min((first_held - track_first) / w, 6.0) if first_held is not None else 6.0, pre,
        min(max(track_last - (start + w - 1), 0) / w, 3.0), float(np.mean(mins < 1.0)),
    ], np.float32)


def _self_check():
    """사람이 물건을 쥐고(프레임 0~13) 놓은 뒤 물건은 그대로, 사람은 멀어짐 -> release=1. 안 쥐면 0."""
    kpt = lambda x: np.array([[x, 50, 0.0]] * 9 + [[x, 50, 0.9], [x, 50, 0.9]] + [[x, 50, 0.0]] * 6, np.float32)
    box = lambda x: np.array([x - 20, 0, x + 20, 100], np.float32)
    seen = {f: (box(100 + 30 * max(f - 13, 0)), kpt(100 + 30 * max(f - 13, 0))) for f in range(24)}
    obj_box = lambda f: np.array([95, 45, 105, 55], np.float32) if f >= 0 else None
    dets = [[(obj_box(f), 0.9)] for f in range(24)]
    objects = link_objects(dets)
    feats = dict(zip(FEATURES, window_features(seen, objects, 0, 0, 23)))
    assert len(objects) == 1 and feats["release"] == 1.0 and feats["held_first"] == 1.0 and feats["held_last"] == 0.0, feats
    far = {f: (box(500), kpt(500)) for f in range(24)}
    assert dict(zip(FEATURES, window_features(far, objects, 0, 0, 23)))["release"] == 0.0


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
