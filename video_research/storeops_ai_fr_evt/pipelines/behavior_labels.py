"""AI Hub 71550 이상행동 라벨(CVAT xml)에서 사건 구간과 행위자 위치를 읽고, 사람별 시간 창에 라벨을 붙인다.

행동 공통 구조: <접두사>_start / <접두사>_end 트랙(첫 보이는 프레임이 시작·끝), 행위자는 'Spine naval' 점 트랙(프레임별, ID 속성).
창 라벨은 abandon_windows와 같은 규칙이다: 1 = 사건과 절반 이상 겹치고 그 구간에서 행위자 점에 가장 가까운 사람의 창,
-1 = 사건 ±ABANDON_EVENT_MARGIN 근처의 같은 사람 창(전환 구간, 학습·평가 제외), 0 = 그 밖의 창(다른 사람 포함).
행위자가 둘 이상(폭행)이면 행위자마다 가장 가까운 사람을 양성으로 둔다. 가까운 사람 정하기는 라벨 정합성용이다.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from config.config import ABANDON_EVENT_MARGIN, ABANDON_WIN_FRAMES

PREFIX = {"전도": "fall", "유기": "abandon", "파손": "broken", "방화": "fire", "절도": "theft", "폭행": "fight", "반품": "normal", "매장이동": "normal"}  # 반품·매장이동은 정상 풀(사건 없음, 모든 창 라벨 0)
ACTOR_JOINT = "Spine naval"


def parse_label(xml_path: Path, behavior: str):
    """(events [{start, end}], actors {ID: {frame: (x, y)}})."""
    prefix = PREFIX[behavior]
    starts, ends, actors = [], [], {}
    for track in ET.parse(xml_path).getroot().iter("track"):
        label = track.get("label")
        if label in (f"{prefix}_start", f"{prefix}_end"):
            seen = [int(b.get("frame")) for b in track.iter("box") if b.get("outside") == "0"]
            if seen:
                (starts if label.endswith("start") else ends).append(min(seen))
        elif label == ACTOR_JOINT:
            for point in track.iter("points"):
                if point.get("outside") != "0":
                    continue
                actor = next((a.text for a in point.iter("attribute") if a.get("name") == "ID"), "0")
                x, y = (float(v) for v in point.get("points").split(",")[:2])
                actors.setdefault(actor, {})[int(point.get("frame"))] = (x, y)
    if len(starts) != len(ends) or not starts:
        raise ValueError(f"{prefix}_start/{prefix}_end 개수 불일치: {xml_path.name} ({len(starts)}, {len(ends)})")
    return [{"start": s, "end": e} for s, e in zip(sorted(starts), sorted(ends))], actors


def _distance(seen, frames, points):
    """창의 지정 프레임에서 사람 박스 중심과 행위자 점(가장 가까운 프레임의 점)의 평균 거리, 못 보면 inf."""
    keys = np.array(sorted(points))
    dists = []
    for f in frames:
        if f in seen:
            box = seen[f][0]
            px, py = points[int(keys[np.abs(keys - f).argmin()])]
            dists.append(np.hypot((box[0] + box[2]) / 2 - px, (box[1] + box[3]) / 2 - py))
    return float(np.mean(dists)) if dists else float("inf")


def label_windows_actors(windows, tracks, events, actors):
    labels = np.zeros(len(windows), np.int8)
    if not actors:
        return labels
    w = ABANDON_WIN_FRAMES
    for event in events:
        s, e = event["start"], event["end"] + 1
        by_start: dict[int, list[int]] = {}
        for i, (tid, start) in enumerate(windows):
            if start < e + ABANDON_EVENT_MARGIN and start + w > s - ABANDON_EVENT_MARGIN:
                by_start.setdefault(start, []).append(i)
        for start, idxs in by_start.items():
            inside = range(max(start, s), min(start + w, e))
            frames = inside if len(inside) else range(start, start + w)
            for points in actors.values():
                dist = {i: _distance(tracks[windows[i][0]], frames, points) for i in idxs}
                near = min(dist, key=dist.get)
                if dist[near] == float("inf"):
                    continue
                if len(inside) >= w / 2:
                    labels[near] = 1
                elif labels[near] == 0:
                    labels[near] = -1
    return labels


def _self_check():
    """행위자 둘(1번·2번)이 각각 가까운 사람에게 양성, 먼 3번 사람은 음성."""
    from pipelines.abandon_windows import make_windows

    box = lambda x: (np.array([x, 0, x + 10, 10], np.float32), np.zeros((17, 3), np.float32))
    tracks = {t: {f: box(x) for f in range(100)} for t, x in ((1, 100), (2, 300), (3, 900))}
    actors = {"0": {f: (105, 5) for f in range(40, 71)}, "1": {f: (305, 5) for f in range(40, 71)}}
    windows = make_windows(tracks, 100)
    labels = label_windows_actors(windows, tracks, [{"start": 40, "end": 70}], actors)
    pos = {w[0] for w, l in zip(windows, labels) if l == 1}
    assert pos == {1, 2}, pos
    assert all(l == 0 for w, l in zip(windows, labels) if w[0] == 3)


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
