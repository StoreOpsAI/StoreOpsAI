"""파손(물건을 들고 바닥으로 던짐) 근거: YOLO-pose 추정 관절의 최대 손목 속도가 라벨 관절(eval_broken_keypoints)처럼 갈리는지 본다.

- 파손 train 영상의 사건 구간마다, 그 구간 안 모든 사람 중 최대 손목 속도 하나(사건 단위)
- 반품 영상(정상)은 추적 번호마다 영상 전체의 최대 손목 속도 하나(사람 단위), 기준을 넘는 사람 수를 분당으로도 센다
- 손목 속도 = 3fps 연속 표본 사이 손목 이동량 / 몸통 길이(어깨 중점-골반 중점). 관절 신뢰도 POSE_MIN_CONF 미만은 버린다
- 공식 validation은 쓰지 않는다
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from config.config import POSE_MIN_CONF, POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.prepare_behaviors import list_videos

SAMPLE_FPS = 3  # 라벨 관절 확인과 같은 간격
LS, RS, LW, RW, LH, RH = 5, 6, 9, 10, 11, 12  # COCO 관절 번호


def load_pose_tracks(video, cache_dir):
    """{추적 번호: {프레임: 17x3 관절}} (3fps 표본). 캐시한다."""
    cache = cache_dir / f"{Path(video).stem}.json"
    if cache.is_file():
        return {tid: {int(f): np.array(k) for f, k in frames.items()}
                for tid, frames in json.loads(cache.read_text(encoding="utf-8")).items()}
    from ultralytics import YOLO

    model = YOLO(str(APP_ROOT / POSE_WEIGHTS))
    cap = cv2.VideoCapture(str(video))
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or SAMPLE_FPS) / SAMPLE_FPS))
    tracks, frame_no = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % step == 0:
            result = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
            if result.boxes.id is not None:
                for tid, kps in zip(result.boxes.id.int().tolist(), result.keypoints.data.tolist()):
                    tracks.setdefault(str(tid), {})[frame_no] = kps
        frame_no += 1
    cap.release()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(tracks), encoding="utf-8")
    return {tid: {f: np.array(k) for f, k in frames.items()} for tid, frames in tracks.items()}


def wrist_speeds(frames, start=0, end=None):
    """(프레임, 손목 속도) 목록. start~end 안의 연속 표본만 쓴다."""
    order = [f for f in sorted(frames) if start <= f and (end is None or f <= end)]
    good = lambda k, i: k[i, 2] >= POSE_MIN_CONF
    step = min((b - a for a, b in zip(order, order[1:])), default=1)  # 표본 간격(프레임)
    speeds = []
    for a, b in zip(order, order[1:]):
        ka, kb = frames[a], frames[b]
        if b - a > step:  # 추적이 끊긴 곳은 속도를 재지 않는다
            continue
        if not all(good(k, i) for k in (ka, kb) for i in (LS, RS, LH, RH)):
            continue
        torso = np.linalg.norm((kb[LS, :2] + kb[RS, :2]) / 2 - (kb[LH, :2] + kb[RH, :2]) / 2) + 1e-6
        moves = [np.linalg.norm(kb[w, :2] - ka[w, :2]) for w in (LW, RW) if good(ka, w) and good(kb, w)]
        if moves:
            speeds.append((b, max(moves) / torso))
    return speeds


def main():
    parser = argparse.ArgumentParser(description="파손 손목 속도 (YOLO-pose)")
    parser.add_argument("--cache-dir", type=Path, default=APP_ROOT / "output" / "broken_pose")
    args = parser.parse_args()

    events = []
    for xml, mp4, split in list_videos(WORKSPACE_ROOT / "파손", "파손")[0]:
        if split != "train":
            continue
        tracks = load_pose_tracks(mp4, args.cache_dir)
        for event in parse_annotation(xml, "broken"):
            values = [s for frames in tracks.values()
                      for _, s in wrist_speeds(frames, event["start_frame"], event["end_frame"])]
            events.append(max(values, default=0.0))
    people, minutes = [], 0.0
    for mp4 in sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4")):
        tracks = load_pose_tracks(mp4, args.cache_dir)
        cap = cv2.VideoCapture(str(mp4))
        minutes += cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or SAMPLE_FPS) / 60
        cap.release()
        people += [max((s for _, s in wrist_speeds(frames)), default=0.0) for frames in tracks.values()]

    broken, normal = np.array(events), np.array(people)
    summary = {"broken_events": len(broken), "return_people": len(normal), "return_minutes": round(minutes, 1),
               "broken_p10_50_90": np.percentile(broken, [10, 50, 90]).round(2).tolist(),
               "return_p10_50_90": np.percentile(normal, [10, 50, 90]).round(2).tolist(), "thresholds": {}}
    for t in (0.73, 0.97, 1.2, 1.5, 2.0):
        summary["thresholds"][t] = {"broken_recall": round(float(np.mean(broken >= t)), 2),
                                    "return_people_ratio": round(float(np.mean(normal >= t)), 2),
                                    "return_per_min": round(float(np.sum(normal >= t)) / minutes, 2)}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
