"""전도 이벤트 단위 지표(스펙 02 §13): 이벤트 정밀도·재현율·F1, 오경보/시간, 탐지 지연(초).

- 이벤트 = 전도 클립 하나(적중 1건). 적중 = 라벨 전도 박스와 겹치는 경보(전환 구간 시작 이후), 오경보 = judge_alarms와 같은 정의.
- 정밀도 = 적중 / (적중 + 오경보), 오경보/시간 = 오경보 / 정상 시간(전환 구간 앞 정상 + 정상 클립 전체).
- 탐지 지연 = 첫 적중 경보 프레임 - 라벨 전도 시작 프레임 (초, 3fps). 음수면 라벨 시작보다 먼저 경보.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from config.config import LYING_POSITIVE_IOU, LYING_TRANSITION_FRAMES
from models.fall_detector import lying_alarms
from models.new_object_detector import _iou

FPS = 3.0


def clip_alarms(clip, scores, thr, hold, link_iou):
    by_frame = defaultdict(list)
    for f, box, s in zip(clip["frame"], clip["box"], scores):
        if s >= thr:
            by_frame[int(f)].append(tuple(map(float, box)))
    return lying_alarms([(f, by_frame.get(f, [])) for f in range(clip["frames"])], FPS, hold_sec=hold, link_iou=link_iou)


def judge(clip, alarms):
    """(첫 적중 경보 프레임 또는 None, 오경보 수). 전환 구간 안의 다른 사람 경보는 세지 않는다(eval_lying.judge_alarms와 같음)."""
    if not clip["falls"]:
        return None, len(alarms)
    first = min(f["start"] for f in clip["falls"])
    hit_frame, false = None, 0
    for frame, box in alarms:
        on_fallen = any(frame >= f["start"] - LYING_TRANSITION_FRAMES and _iou(box, f["box_end"]) >= LYING_POSITIVE_IOU for f in clip["falls"])
        if on_fallen:
            hit_frame = frame if hit_frame is None else min(hit_frame, frame)
        elif frame < first - LYING_TRANSITION_FRAMES or frame >= first:
            false += 1
    return hit_frame, false


def event_stats(clips, scores, thr, hold, link_iou):
    hits = events = false = 0
    minutes, delays = 0.0, []
    for c in clips:
        hit_frame, fa = judge(c, clip_alarms(c, scores[c["clip"]], thr, hold, link_iou))
        false += fa
        if c["falls"]:
            events += 1
            if hit_frame is not None:
                hits += 1
                delays.append((hit_frame - min(f["start"] for f in c["falls"])) / FPS)
            normal_frames = max(min(f["start"] for f in c["falls"]) - LYING_TRANSITION_FRAMES, 0)
        else:
            normal_frames = c["frames"]
        minutes += normal_frames / FPS / 60
    precision = hits / max(hits + false, 1)
    recall = hits / max(events, 1)
    return {"recall": recall, "precision": precision, "f1": 2 * precision * recall / max(precision + recall, 1e-9), "hits": hits, "fall_clips": events,
            "false_alarms": false, "normal_minutes": round(minutes, 1), "false_alarm_per_hour": false / minutes * 60,
            "delay_median_sec": float(np.median(delays)) if delays else float("nan")}


if __name__ == "__main__":
    box = (100.0, 100.0, 300.0, 160.0)
    fall = {"clip": "a", "frames": 60, "falls": [{"start": 30, "end": 59, "box_start": box, "box_end": box}],
            "frame": np.arange(60), "box": np.array([box] * 60, np.float32)}
    normal = {"clip": "b", "frames": 60, "falls": [], "frame": np.arange(60), "box": np.array([box] * 60, np.float32)}
    s = {"a": np.where(np.arange(60) >= 30, 0.99, 0.01), "b": np.full(60, 0.01)}
    r = event_stats([fall, normal], s, 0.5, 1.0, 0.3)
    assert r["recall"] == 1.0 and r["false_alarms"] == 0 and r["precision"] == 1.0, r
    assert abs(r["delay_median_sec"] - 1.0) < 1e-6, r  # 라벨 시작 30에서 3표본 뒤(1초)에 경보
    s["b"] = np.full(60, 0.99)  # 정상 클립이 계속 누움으로 판정 → 오경보 1건
    assert event_stats([fall, normal], s, 0.5, 1.0, 0.3)["false_alarms"] == 1
    print("ok")
