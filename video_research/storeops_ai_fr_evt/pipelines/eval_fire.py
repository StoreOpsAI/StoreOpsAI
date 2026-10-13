"""방화 근거: 불 검출(FIRE_BASE_WEIGHTS 등 YOLO 가중치) 단독과 사람+불 동시 검출을 방화·다른 영상에서 잰다.

- 불 경보: FIRE_SAMPLE_FPS로 표본을 뽑아 FIRE_PERSIST_SEC 이상 연속으로 불이 검출되면 한 번(끊기면 다시 셈)
- 사람+불: 불 경보 순간 불 박스 근처(FIRE_NEAR_RATIO)에 사람 박스가 있을 때
- 적중: 방화 사건 시작 ~ 종료 + FIRE_MATCH_AFTER_SEC 안의 경보. 오경보는 방화 영상의 사건 시작 전과 다른 영상 전체에서만 센다(score_alarms)
- 검출은 점수 0.02 이상을 모두 캐시해 두고 점수 하한별로 다시 센다(하한은 train에서 고르므로 낙관적). 공식 validation 미사용
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from config.config import FIRE_BASE_WEIGHTS, FIRE_CLASSES, FIRE_NEAR_RATIO, FIRE_PERSIST_SEC, FIRE_SAMPLE_FPS
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.eval_new_object import load_person_boxes
from pipelines.prepare_behaviors import list_videos

FIRE_MATCH_AFTER_SEC = 10
CONFS = (0.05, 0.1, 0.2, 0.3)


def load_fire_detections(video, model, cache_dir, imgsz=640):
    """(fps, 표본 간격, {프레임: [(x1, y1, x2, y2, 점수)]}). FIRE_CLASSES에 드는 클래스만 남긴다."""
    cache = cache_dir / f"{Path(video).stem}.json"
    if cache.is_file():
        fps, step, found = json.loads(cache.read_text(encoding="utf-8"))
        return fps, step, {int(f): boxes for f, boxes in found.items()}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
    step = max(1, round(fps / FIRE_SAMPLE_FPS))
    found, frame_no = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % step == 0:
            result = model.predict(frame, conf=0.02, imgsz=imgsz, verbose=False)[0]
            found[frame_no] = [[*map(int, box), round(score, 3)]
                               for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                                          result.boxes.cls.int().tolist())
                               if result.names[cls] in FIRE_CLASSES]
        frame_no += 1
    cap.release()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([fps, step, found]), encoding="utf-8")
    return fps, step, found


def fire_alarms(found, fps, step, conf, persons=None):
    """경보 프레임 목록. persons(프레임별 사람 박스)를 주면 불 근처에 사람이 있을 때만."""
    need = max(1, round(FIRE_PERSIST_SEC * fps / step))
    alarms, run = [], 0
    for frame_no in sorted(found):
        fires = [box for box in found[frame_no] if box[4] >= conf]
        if persons is not None:
            people = persons[frame_no] if frame_no < len(persons) else []
            fires = [box for box in fires if any(_near(box, person) for person in people)]
        run = run + 1 if fires else 0
        if run == need:
            alarms.append(frame_no)
    return alarms


def score_alarms(alarms, events, fps):
    """(맞힌 사건 수, 사건 수, 오경보 수). 오경보는 불이 없다고 확실한 구간(첫 사건 시작 전, 사건 없는 영상은 전체)만 센다.

    사건 종료 뒤에도 불은 계속 탈 수 있고 그 구간은 라벨이 없으므로 적중·오경보 어느 쪽으로도 세지 않는다.
    """
    spans = [(e["start_frame"], e["end_frame"] + FIRE_MATCH_AFTER_SEC * fps) for e in events]
    first = min((a for a, _ in spans), default=float("inf"))
    hits = sum(any(a <= f <= b for f in alarms) for a, b in spans)
    return hits, len(spans), sum(f < first for f in alarms)


def _near(fire, person):
    x1, y1, x2, y2 = person
    dx, dy = (x2 - x1) * FIRE_NEAR_RATIO, (y2 - y1) * FIRE_NEAR_RATIO
    cx, cy = (fire[0] + fire[2]) / 2, (fire[1] + fire[3]) / 2
    return x1 - dx <= cx <= x2 + dx and y1 - dy <= cy <= y2 + dy


def main():
    parser = argparse.ArgumentParser(description="방화 불·연기 검출 평가")
    parser.add_argument("--cache-dir", type=Path, default=APP_ROOT / "output" / "fire_eval")
    parser.add_argument("--weights", type=Path, default=APP_ROOT / FIRE_BASE_WEIGHTS)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    fire_videos = [(mp4, parse_annotation(xml, "fire"))
                   for xml, mp4, split in list_videos(WORKSPACE_ROOT / "방화", "방화")[0] if split == "train"]
    others = [mp4 for name in ("절도", "파손", "유기")
              for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]
    others += sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))

    summary = {}
    for mode in ("fire", "person_fire"):
        for conf in CONFS:
            hit = total = false = 0
            minutes = 0.0
            for video, events in [(v, e) for v, e in fire_videos] + [(v, []) for v in others]:
                fps, step, found = load_fire_detections(video, model, args.cache_dir)
                persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons") \
                    if mode == "person_fire" else None
                alarms = fire_alarms(found, fps, step, conf, persons)
                h, t, fa = score_alarms(alarms, events, fps)
                hit, total, false = hit + h, total + t, false + fa
                first = min((e["start_frame"] for e in events), default=max(found, default=0) + step)
                minutes += first / fps / 60  # 오경보를 센 구간 길이만
            summary[f"{mode}@{conf}"] = {"recall": round(hit / total, 2) if total else 0.0, "events": total,
                                         "false_alarms": false, "fp_per_min": round(false / minutes, 3)}
            print(f"{mode}@{conf}", summary[f"{mode}@{conf}"])
    (args.cache_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
