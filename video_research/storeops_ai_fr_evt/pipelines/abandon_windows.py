"""유기 영상의 사람 추적과 사람별 시간 창, 창 라벨을 만든다 (abandon_features가 특징을 뽑을 때 쓴다).

라벨(창 단위): 1 = 사건 구간과 창의 절반 이상 겹치고, 그 겹친 프레임에서 라벨 물건에 가장 가까운 사람의 창
              -1 = 사건 ±ABANDON_EVENT_MARGIN 근처의 같은 사람 창(전환 구간, 학습·평가에서 제외; AGENTS.md 규칙 4)
               0 = 그 밖의 모든 창(다른 사람 포함)
'가장 가까운 사람'은 라벨 정합성용이다. 추적 번호가 바뀌어도 창마다 다시 정하므로 치명적이지 않다.
"""
from __future__ import annotations

import numpy as np

from config.config import (ABANDON_EVENT_MARGIN, ABANDON_WIN_FRAMES, ABANDON_WIN_MIN_PRESENT, ABANDON_WIN_STRIDE,
                           POSE_IMGSZ, TRACKER_CONFIG)

PERSON_CONF = 0.25


def track_frames(model, frames):
    """tid -> {frame: (box(4), kpt(17,3))}. 영상마다 추적기를 새로 시작한다."""
    model.predictor = None
    tracks: dict[int, dict[int, tuple]] = {}
    for n, image in enumerate(frames):
        result = model.track(image, persist=True, tracker=TRACKER_CONFIG, imgsz=POSE_IMGSZ, conf=PERSON_CONF,
                             half=True, classes=[0], verbose=False)[0]
        if result.boxes.id is None:
            continue
        for tid, box, kpt in zip(result.boxes.id.int().tolist(), result.boxes.xyxy.cpu().numpy(),
                                 result.keypoints.data.cpu().numpy()):
            tracks.setdefault(tid, {})[n] = (box, kpt)
    return tracks


def make_windows(tracks, n_frames):
    """[(tid, start)]: 길이 ABANDON_WIN_FRAMES 창에서 그 사람이 MIN_PRESENT 비율 이상 보이는 것."""
    out = []
    for tid, seen in tracks.items():
        for start in range(0, n_frames - ABANDON_WIN_FRAMES + 1, ABANDON_WIN_STRIDE):
            present = sum(1 for f in range(start, start + ABANDON_WIN_FRAMES) if f in seen)
            if present >= ABANDON_WIN_MIN_PRESENT * ABANDON_WIN_FRAMES:
                out.append((tid, start))
    return out


def _object_center(obj_boxes, frame):
    """frame에 가장 가까운 프레임의 물건 박스 중심들."""
    nearest = min(obj_boxes, key=lambda f: abs(f - frame))
    return [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in obj_boxes[nearest]]


def _distance(seen, frames, obj_boxes):
    """창 안 겹친 프레임에서 사람 박스 중심과 가장 가까운 물건 중심의 평균 거리(보이지 않으면 inf)."""
    dists = []
    for f in frames:
        if f in seen:
            box = seen[f][0]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            dists.append(min(np.hypot(cx - ox, cy - oy) for ox, oy in _object_center(obj_boxes, f)))
    return float(np.mean(dists)) if dists else float("inf")


def label_windows(windows, tracks, events, obj_boxes):
    """창마다 1 / -1 / 0. events = [{start, end}], obj_boxes = {frame: [박스,...]} (라벨 물건 트랙)."""
    labels = np.zeros(len(windows), np.int8)
    if not obj_boxes:
        return labels
    for event in events:
        s, e = event["start"], event["end"] + 1
        by_start: dict[int, list[int]] = {}
        for i, (tid, start) in enumerate(windows):
            if start < e + ABANDON_EVENT_MARGIN and start + ABANDON_WIN_FRAMES > s - ABANDON_EVENT_MARGIN:
                by_start.setdefault(start, []).append(i)
        for start, idxs in by_start.items():
            inside = range(max(start, s), min(start + ABANDON_WIN_FRAMES, e))
            near = min(idxs, key=lambda i: _distance(tracks[windows[i][0]], range(start, start + ABANDON_WIN_FRAMES),
                                                    obj_boxes) if not inside else
                       _distance(tracks[windows[i][0]], inside, obj_boxes))
            if len(inside) >= ABANDON_WIN_FRAMES / 2:
                labels[near] = 1
            elif labels[near] == 0:
                labels[near] = -1
    return labels


def _self_check():
    """가까운 사람(1번)만 양성, 같은 창의 다른 사람(2번)은 음성, 사건 전 가장자리 창은 -1."""
    box = lambda x: (np.array([x, 0, x + 10, 10], np.float32), np.zeros((17, 3), np.float32))
    tracks = {1: {f: box(100) for f in range(0, 100)}, 2: {f: box(900) for f in range(0, 100)}}
    windows = make_windows(tracks, 100)
    labels = label_windows(windows, tracks, [{"start": 40, "end": 70}], {f: [[95, 0, 115, 10]] for f in range(40, 71)})
    got = {(w, int(l)) for w, l in zip(windows, labels) if l != 0}
    assert (1, 36) in {(w[0], w[1]) for w, l in zip(windows, labels) if l == 1}, got
    assert all(w[0] == 1 for w, l in zip(windows, labels) if l != 0), got
    assert any(l == -1 for l in labels) and not any(l == 1 for w, l in zip(windows, labels) if w[0] == 2)


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
