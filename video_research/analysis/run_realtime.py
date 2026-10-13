"""전도 실시간 실행기(팀원 배포용). 웹캠 번호, RTSP 주소, 또는 영상 파일을 받아 3fps로 표본을 뽑아 전도 경보를 낸다.

사용 예:
  python run_realtime.py --source 0                          # 웹캠 0번
  python run_realtime.py --source rtsp://user:pw@host/stream # IP 카메라
  python run_realtime.py --source clip.mp4 --log-scores scores.jsonl   # 영상 파일(시험용)

--show: 화면 왼쪽에 FALL 게이지(누움 점수, 노란 선=경보 임계값)와 HOLD 게이지(같은 자리 유지 시간 진행률), 사람 박스에 점수(%)를 그린다.
경보는 한 줄 JSON으로 출력한다: {"event":"fall","time":..., "sample":..., "box":[x1,y1,x2,y2], "score":...}
--log-scores를 주면 경보가 아닌 표본까지 모든 사람의 점수를 기록한다(임계값 조정·오경보 분석용).
경로는 모두 이 파일 위치 기준 상대 경로다(config.json). 과거 프레임만 쓰며 큐를 쌓지 않는다(밀리면 오래된 프레임은 버린다).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lying_core import LyingMLP, LyingTracker, pose_features

HERE = Path(__file__).resolve().parent


def load_config():
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def open_source(source):
    """(캡처, 파일 여부). 숫자면 웹캠 번호."""
    is_file = Path(source).is_file()
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"입력을 열 수 없습니다: {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 실시간 입력은 버퍼를 쌓지 않는다
    return cap, is_file


def read_frames(cap, is_file, fps):
    """(프레임, 표본 여부, 표본 번호, 시각(초)). 파일은 프레임 간격으로, 실시간은 벽시계 시간으로 fps에 맞춰 표본을 고른다."""
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, round(src_fps / fps)) if is_file else 1
    frame_no = sample_no = 0
    next_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        now = time.time()
        if is_file:
            take, stamp = frame_no % step == 0, frame_no / src_fps
        else:
            take, stamp = now >= next_time, now
            if take:
                next_time = now + 1.0 / fps
        frame_no += 1
        yield frame, take, sample_no, stamp
        sample_no += take


def draw_overlay(frame, boxes, scores, threshold, progress, alarm_until, now):
    """왼쪽 세로 게이지(누움 점수 0~100%, 임계값 선), 그 옆 유지 시간 게이지, 사람 박스(점수 색), 경보 배너. 글자는 영문(cv2 글꼴 한계)."""
    out = frame.copy()
    h, w = out.shape[:2]
    top, bottom, left, bar = int(h * 0.12), int(h * 0.88), 24, 36
    top_score = float(scores.max()) if len(scores) else 0.0
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        color = (0, 0, 255) if s >= threshold else (0, 200, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(out, f"{s * 100:.0f}%", (int(x1), max(int(y1) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    for x, value, label, color in ((left, top_score, "FALL", (0, 0, 255) if top_score >= threshold else (0, 200, 0)),
                                   (left + 150, progress, "HOLD", (0, 165, 255))):
        cv2.rectangle(out, (x, top), (x + bar, bottom), (40, 40, 40), -1)
        fill = int(bottom - (bottom - top) * value)
        cv2.rectangle(out, (x, fill), (x + bar, bottom), color, -1)
        cv2.rectangle(out, (x, top), (x + bar, bottom), (255, 255, 255), 2)
        cv2.putText(out, f"{label} {value * 100:.0f}%", (x - 4, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    ty = int(bottom - (bottom - top) * threshold)
    cv2.line(out, (left - 8, ty), (left + bar + 8, ty), (0, 255, 255), 3)  # 경보 임계값
    cv2.putText(out, "NORMAL", (left - 4, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if now < alarm_until:
        cv2.rectangle(out, (0, 0), (w, int(h * 0.08)), (0, 0, 255), -1)
        cv2.putText(out, "FALL ALARM", (w // 2 - 130, int(h * 0.055)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description="전도 실시간 실행기")
    parser.add_argument("--source", required=True, help="웹캠 번호 | RTSP 주소 | 영상 파일")
    parser.add_argument("--log-scores", default="", help="모든 사람의 점수를 이 JSONL 파일에 기록")
    parser.add_argument("--show", action="store_true", help="화면에 박스·왼쪽 게이지(누움 점수, 유지 시간)·경보 배너를 띄운다(q로 종료)")
    parser.add_argument("--save-overlay", default="", help="시험용: 마지막 표본 화면을 이 이미지 파일로 저장")
    parser.add_argument("--max-samples", type=int, default=0, help="시험용: 이 표본 수에서 종료")
    args = parser.parse_args()
    cfg = load_config()
    from ultralytics import YOLO

    pose = YOLO(str(HERE / cfg["pose_weights"]))
    mlp = LyingMLP(HERE / cfg["lying_weights"])
    tracker = LyingTracker(cfg["fps"], cfg["hold_sec"], cfg["same_spot_iou"], cfg["gap_samples"])
    cap, is_file = open_source(args.source)
    log = open(args.log_scores, "w", encoding="utf-8") if args.log_scores else None
    boxes, scores, alarm_until = np.zeros((0, 4)), np.zeros(0), 0.0
    try:
        for frame, take, sample_no, stamp in read_frames(cap, is_file, cfg["fps"]):
            if take:
                result = pose.predict(frame, imgsz=cfg["pose_imgsz"], conf=cfg["person_conf"], verbose=False)[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                kpts = result.keypoints.data.cpu().numpy()
                scores = mlp(np.array([pose_features(b, k, cfg["kpt_conf"]) for b, k in zip(boxes, kpts)]).reshape(len(boxes), -1)) if len(boxes) else np.zeros(0)
                if log:
                    for box, score in zip(boxes, scores):
                        log.write(json.dumps({"sample": sample_no, "time": round(stamp, 2), "box": [round(float(v), 1) for v in box], "score": round(float(score), 4)}) + "\n")
                lying = [tuple(map(float, b)) for b, s in zip(boxes, scores) if s >= cfg["score_threshold"]]
                for alarm_sample, box in tracker.update(sample_no, lying):
                    score = float(max(s for b, s in zip(boxes, scores) if tuple(map(float, b)) == box))
                    alarm_until = time.time() + 5.0
                    print(json.dumps({"event": "fall", "time": round(stamp, 2), "sample": alarm_sample, "box": [round(v, 1) for v in box], "score": round(score, 4)}), flush=True)
                if args.max_samples and sample_no + 1 >= args.max_samples:
                    break
            if args.show or args.save_overlay:
                view = draw_overlay(frame, boxes, scores, cfg["score_threshold"], tracker.progress(), alarm_until, time.time())
                if args.save_overlay and take:
                    cv2.imwrite(args.save_overlay, view)  # 시험용: 마지막 표본 화면을 파일로 남긴다
                if args.show:
                    cv2.imshow("fall detection (q: quit)", cv2.resize(view, None, fx=0.5, fy=0.5) if view.shape[1] > 1600 else view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
