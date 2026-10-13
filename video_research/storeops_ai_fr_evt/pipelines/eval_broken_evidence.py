"""파손 근거 측정: 던져서 바닥에 흩어진 물건을 세 가지 근거로 잡을 수 있는지 본다 (train 영상만).

근거(창 WINDOW 초 단위):
  ① added:   이 창에서 처음 생긴 새 물건 수 (NewObjectDetector 'added', 생긴 시각 = first_frame)
  ② motion:  사람 박스 밖에서 직전 프레임과 크게 달라진 화소 비율의 최댓값 (날아가거나 흩어지는 물건)
  ③ removed: 이 창에서 처음 생긴 치워진 자리 수 (선반에서 빠진 물건)
양성: 파손 사건 구간 [start, end]과 겹치는 창. 사건은 겹치는 창 중 하나라도 켜지면 잡은 것으로 센다.
음성: 파손 영상의 사건 전후 여유(MARGIN_BEFORE/AFTER) 밖 창 + 유기·절도·방화 train, 반품 영상 전체.
공식 validation 영상은 쓰지 않는다. 여기서 본 분포로 기준값 후보만 고르고, 확정은 세션 단위 개발 세트에서 한다.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from config.config import NEW_OBJECT_DIFF
from models.new_object_detector import SCALE, NewObjectDetector
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.eval_new_object import load_person_boxes
from pipelines.prepare_behaviors import list_videos

MARGIN_BEFORE_SEC = 5
MARGIN_AFTER_SEC = 20  # 사건 뒤 잔해가 가라앉는 시간. 전환 구간을 음성에 섞지 않는다


def frame_evidence(video_path, person_boxes):
    """(프레임별 motion 비율, added first_frame 목록, removed first_frame 목록, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
    detector = NewObjectDetector(fps)
    motion, added, removed, previous = [], [], [], None
    for boxes in person_boxes:
        ok, frame = cap.read()
        if not ok:
            break
        for item in detector.update(frame, boxes):
            (added if item["kind"] == "added" else removed).append(item["first_frame"])
        gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY),
                                (5, 5), 0).astype(np.float32)
        outside = np.ones(gray.shape, bool)
        for x1, y1, x2, y2 in boxes:
            w, h = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            outside[max(0, int((y1 - h) * SCALE)):int((y2 + h) * SCALE),
                    max(0, int((x1 - w) * SCALE)):int((x2 + w) * SCALE)] = False
        changed = np.abs(gray - previous) > NEW_OBJECT_DIFF if previous is not None else np.zeros(gray.shape, bool)
        motion.append(float((changed & outside).sum()) / gray.size)
        previous = gray
    cap.release()
    return motion, added, removed, fps


def windows(motion, added, removed, fps, window_sec):
    """[(시작 프레임, 끝 프레임, added 수, motion 최대, removed 수)]."""
    length = max(1, round(window_sec * fps))
    rows = []
    for start in range(0, len(motion), length):
        end = min(start + length, len(motion)) - 1
        rows.append((start, end, sum(start <= f <= end for f in added), max(motion[start:end + 1]),
                     sum(start <= f <= end for f in removed)))
    return rows


def split_windows(rows, events, fps):
    """(사건별 겹치는 창 목록, 음성 창 목록)."""
    positives = [[r for r in rows if r[0] <= e["end_frame"] and r[1] >= e["start_frame"]] for e in events]
    spans = [(e["start_frame"] - MARGIN_BEFORE_SEC * fps, e["end_frame"] + MARGIN_AFTER_SEC * fps) for e in events]
    negatives = [r for r in rows if not any(r[0] <= b and r[1] >= a for a, b in spans)]
    return positives, negatives


def fires(row, min_added, min_motion, need):
    """세 근거 중 need개 이상 켜지면 참."""
    return (row[2] >= min_added) + (row[3] >= min_motion) + (row[4] >= 1) >= need


def list_sources():
    """{출처: [(mp4, 파손 사건 목록)]}. 파손만 사건이 있고 나머지는 전부 음성."""
    sources = {}
    for name in ("파손", "유기", "절도", "방화"):
        items = [(mp4, xml) for xml, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]
        sources[name] = [(mp4, parse_annotation(xml, "broken") if name == "파손" else []) for mp4, xml in items]
    sources["반품"] = [(mp4, []) for mp4 in sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))]
    return sources


def collect(window_sec, cache_dir):
    """{출처: {"events": [[창...]...], "negatives": [창...], "minutes": 분}}. 프레임 근거는 캐시한다."""
    result = {}
    for source, videos in list_sources().items():
        events_out, negatives_out, seconds = [], [], 0.0
        for video, events in videos:
            cache = cache_dir / f"{Path(video).stem}.json"
            if cache.is_file():
                motion, added, removed, fps = json.loads(cache.read_text(encoding="utf-8"))
            else:
                persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons")
                motion, added, removed, fps = frame_evidence(video, persons)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps([motion, added, removed, fps]), encoding="utf-8")
            positives, negatives = split_windows(windows(motion, added, removed, fps, window_sec), events, fps)
            events_out += positives
            negatives_out += negatives
            seconds += sum(row[1] - row[0] + 1 for row in negatives) / fps
        result[source] = {"events": events_out, "negatives": negatives_out, "minutes": seconds / 60}
    return result


def _peaks(events, index):
    return [max((row[index] for row in rows), default=0) for rows in events]


def _percentiles(values):
    return np.percentile(values, [10, 50, 90]).round(4).tolist() if values else []


def main():
    parser = argparse.ArgumentParser(description="파손 근거(새 물건 수·움직임·치워진 자리) 측정, train 영상만")
    parser.add_argument("--window", type=float, default=5.0, help="창 길이(초)")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "broken_evidence")
    args = parser.parse_args()
    data = collect(args.window, args.output_dir / "frames")

    events = data["파손"]["events"]
    distributions = {"파손_사건_peak": {"added": _percentiles(_peaks(events, 2)), "motion": _percentiles(_peaks(events, 3)),
                                      "removed": _percentiles(_peaks(events, 4))}}
    for source, item in data.items():
        rows = item["negatives"]
        distributions[f"{source}_음성창"] = {"added": _percentiles([r[2] for r in rows]),
                                          "motion": _percentiles([r[3] for r in rows]),
                                          "removed": _percentiles([r[4] for r in rows])}

    motion_grid = np.percentile([r[3] for item in data.values() for r in item["negatives"]], [90, 95, 99]).tolist()
    grid = []
    for min_added, min_motion, need in product((1, 2, 3), motion_grid, (1, 2, 3)):
        recall = sum(any(fires(r, min_added, min_motion, need) for r in rows) for rows in events) / len(events)
        fp = {source: round(sum(fires(r, min_added, min_motion, need) for r in item["negatives"])
                            / item["minutes"], 3) if item["minutes"] else 0.0 for source, item in data.items()}
        grid.append({"min_added": min_added, "min_motion": round(min_motion, 4), "need": need,
                     "recall": round(recall, 3), "fp_per_minute": fp})

    summary = {"window_sec": args.window, "events": len(events),
               "minutes": {s: round(item["minutes"], 1) for s, item in data.items()},
               "distributions_p10_p50_p90": distributions, "grid": grid}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "grid"}, ensure_ascii=False, indent=2))
    for row in sorted(grid, key=lambda g: (-g["recall"], sum(g["fp_per_minute"].values())))[:12]:
        print(row)


if __name__ == "__main__":
    main()
