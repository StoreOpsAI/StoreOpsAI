"""
StoreOps AI - 학습된 S3D 행동분류 모델 웹캠 테스트

사용법 (프로젝트 루트에서):
    python webcam_i3d.py

기본:
    - training/runs/ours_a_final(화면 전체)와 ours_c1_final(사람 크롭)을 ours_stack/fusion.json으로 동적 결합해 사용
    - 웹캠 전체 화면을 학습과 동일한 방식으로 입력
    - 3 FPS로 프레임을 버퍼링
    - 최근 4초(12표본)를 24프레임으로 균일 샘플링
    - 2초마다 추론
    - 카테고리별 임계값(fusion.json에서 정한 값, 환경변수로 덮어쓰기 가능)을 넘으면 EVENT 표시

종료: q 또는 ESC
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# 프로젝트 루트에서 실행할 때 config/models import가 가능하도록 보장
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import (
    CATEGORY_THRESHOLDS, CROP_LARGEST_PERSON, I3D_WEIGHT_PATH, I3D_WINDOW_SEC, YOLO_MODEL_PATH, TRACKER_CONFIG, CONF_THRESHOLD, PERSON_CLASS_ID,
)
from models.stack_head import largest_person_boxes, union_crop
from models.yolo_detector import YOLODetector
from models.i3d_classifier import I3DClassifier, I3DNotConfiguredError


RELEASE_SCORE = 0.5   # 켜진 경보를 끄는 점수(확률). 개발 기록에서 사건 밖 최대 점수가 약 0.45였다는 것에서 정함(근거 약함)
DISPLAY_NAMES = {
    "정상": "NORMAL",
    "쓰러짐": "FALL",
    "쓰레기 투기": "LITTERING",
    "절도": "THEFT",
}


def parse_args():
    p = argparse.ArgumentParser(description="학습된 S3D/I3D 계열 모델 웹캠 테스트")
    p.add_argument("--camera", type=int, default=0, help="웹캠 번호. 기본 0")
    p.add_argument(
        "--weight",
        default=I3D_WEIGHT_PATH,
        help="학습된 best.pt 경로",
    )
    p.add_argument("--window-sec", type=float, default=I3D_WINDOW_SEC,
                   help="한 번 분류할 최근 영상 길이. 기본은 서비스·학습과 같은 4초")
    p.add_argument("--interval-sec", type=float, default=2.0,
                   help="추론 간격. 기본 2초")
    p.add_argument("--buffer-fps", type=float, default=3.0,
                   help="분류 버퍼에 저장할 FPS. 학습 데이터와 동일하게 기본 3")
    p.add_argument("--threshold", type=float, default=None,
                   help="모든 이상행동에 같은 EVENT 임계값을 쓸 때만 지정. 기본은 config의 카테고리별 임계값")
    p.add_argument("--width", type=int, default=1920, help="학습 영상과 같은 1920x1080(카메라가 지원하지 않으면 실제 해상도로 열린다)")
    p.add_argument("--height", type=int, default=1080)
    return p.parse_args()


def draw_score_panel(frame, scores, category, confidence, threshold, ready, meta, held=False):
    h, w = frame.shape[:2]

    # 반투명 검정 패널
    panel_w = min(430, w - 20)
    panel_h = 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    y = 38
    cv2.putText(
        frame, "StoreOps AI - S3D WebCam Test",
        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2
    )
    y += 28

    status = "MODEL READY" if ready else "MODEL ERROR"
    cv2.putText(frame, status, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if ready else (0, 0, 255), 2)
    y += 27

    if meta:
        info = f"{meta.get('arch','?')} | val_acc={meta.get('val_acc','?')}"
        cv2.putText(frame, info, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    y += 28

    if category:
        thr = threshold if threshold is not None else CATEGORY_THRESHOLDS.get(category, 1.0)
        event = held or (category not in ("정상", "판정 보류") and confidence > thr)
        text = f"{DISPLAY_NAMES.get(category, category)}  {confidence:.1%}"
        cv2.putText(
            frame, text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.82,
            (0, 0, 255) if event else (0, 255, 255), 2
        )
        y += 30
        cv2.putText(
            frame,
            "EVENT DETECTED" if event else "NO EVENT",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (0, 0, 255) if event else (0, 255, 0),
            2,
        )
        y += 27

        for c in ["정상", "쓰러짐", "쓰레기 투기", "절도"]:
            s = float(scores.get(c, 0.0))
            label = f"{DISPLAY_NAMES[c]:8s} {s:6.1%}"
            cv2.putText(frame, label, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
            y += 20
    else:
        cv2.putText(frame, "Waiting for enough frames...",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2)

    return frame


def main():
    args = parse_args()

    weight = Path(args.weight)
    if not weight.is_file():
        print(f"[ERROR] 모델 파일이 없습니다: {weight}")
        print("기본 모델: training/runs/ours_a_final/best.pt (+ ours_c1_final, ours_stack/fusion.json)")
        return 1

    print("=" * 70)
    print("StoreOps AI - 학습된 행동분류 모델 웹캠 테스트")
    print("=" * 70)
    print(f"model      : {weight}")
    print(f"camera     : {args.camera}")
    print(f"window     : {args.window_sec:.1f} sec")
    print(f"infer      : every {args.interval_sec:.1f} sec")
    print(f"buffer fps : {args.buffer_fps:.1f}")
    print("threshold  : " + (f"{args.threshold:.2f}" if args.threshold is not None else str(CATEGORY_THRESHOLDS)))
    print("종료       : Q 또는 ESC")
    print()

    try:
        classifier = I3DClassifier(str(weight))
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {type(e).__name__}: {e}")
        return 1

    if not classifier.is_ready:
        print("[ERROR] I3D/S3D 모델이 활성화되지 않았습니다.")
        return 1

    print(f"[OK] 모델 로드 완료: {classifier.meta}")
    print(f"[OK] 학습되지 않은 서비스 카테고리: {classifier.missing_categories}")
    print()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # 일부 카메라/PC에서는 DSHOW가 안 맞을 수 있으므로 기본 backend 재시도
        cap.release()
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[ERROR] 웹캠을 열 수 없습니다: camera={args.camera}")
        print("다른 번호를 시도해보세요. 예: python webcam_i3d.py --camera 1")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 실제 저장 간격은 wall-clock 기준. 카메라 FPS가 달라도 3fps 정도로 맞춘다.
    sample_interval = 1.0 / max(args.buffer_fps, 0.1)
    max_frames = int(round(args.window_sec * args.buffer_fps))   # 학습 창과 같은 표본 수(4초 x 3fps = 12). 분류기가 모델 입력 길이(24)로 균일 재표본한다

    frame_buffer = deque(maxlen=max_frames)
    box_buffer = deque(maxlen=max_frames)   # 표본 프레임마다 사람 박스(C1 크롭용)
    # 시연 기록: 모델이 실제로 본 3fps 표본 프레임을 mp4로, 추론마다 점수를 CSV로 남긴다(원인 분석·재현용). 같은 mp4를 서비스에 다시 돌리면 같은 입력이 된다.
    rec_dir = ROOT / "output" / "demo_recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer, t_start = None, time.time()
    score_file = open(rec_dir / f"{stamp}_scores.csv", "w", newline="", encoding="utf-8")
    score_csv = csv.writer(score_file)
    score_csv.writerow(["경과초", "판정", "신뢰도", "정상", "쓰러짐", "쓰레기 투기", "절도"])
    print(f"기록: {rec_dir / (stamp + '.mp4')} / {rec_dir / (stamp + '_scores.csv')}")
    detector = YOLODetector(YOLO_MODEL_PATH, TRACKER_CONFIG, CONF_THRESHOLD, PERSON_CLASS_ID) if classifier.has_stack else None
    print("A+C1 스택:", "사용" if detector else "미사용(A 단독)")
    buffer_times = deque(maxlen=max_frames)

    last_sample_time = 0.0
    last_submit_time = 0.0
    latest_result = None
    active_cat = None   # 히스테리시스(표시용): 임계값을 넘어 켜진 행동은 점수가 RELEASE_SCORE 아래로 내려갈 때까지 유지한다
    latest_result_time = 0.0
    future = None

    # 추론 중에도 웹캠 화면이 멈추지 않게 별도 worker에서 실행
    executor = ThreadPoolExecutor(max_workers=1)

    fps_t0 = time.time()
    fps_count = 0
    display_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] 웹캠 프레임을 읽지 못했습니다.")
                break

            now = time.time()

            # 3fps 버퍼링
            if now - last_sample_time >= sample_interval:
                frame_buffer.append(frame.copy())
                if writer is None:
                    writer = cv2.VideoWriter(str(rec_dir / f"{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.buffer_fps, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                if detector is not None:
                    r = detector.track(frame)
                    box_buffer.append([(int(i), b) for b, i in zip(r.boxes.xyxy.cpu().numpy().tolist(), (r.boxes.id.cpu().numpy() if r.boxes.id is not None else [])) if i >= 0])
                buffer_times.append(now)
                last_sample_time = now

            # 완료된 추론 결과 반영
            if future is not None and future.done():
                try:
                    latest_result = future.result()
                    sc = latest_result[2]
                    score_csv.writerow([f"{now - t_start:.1f}", latest_result[0], f"{latest_result[1]:.4f}"] + [f"{sc.get(k, 0.0):.4f}" for k in ("정상", "쓰러짐", "쓰레기 투기", "절도")])
                    score_file.flush()
                    print(f"[점수] {latest_result[0]} {latest_result[1]:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in latest_result[2].items() if k != "정상"), flush=True)   # 콘솔 기록(화면에는 영향 없음): 환경별 점수 확인용
                    latest_result_time = now
                except Exception as e:
                    print(f"[I3D ERROR] {type(e).__name__}: {e}")
                future = None

            # 충분한 시간의 프레임이 쌓였을 때만 추론
            enough_window = (
                len(frame_buffer) >= max_frames
                and len(buffer_times) >= 2
                and (buffer_times[-1] - buffer_times[0]) >= min(
                    args.window_sec * 0.85, args.window_sec - 0.1
                )
            )

            if (
                enough_window
                and future is None
                and now - last_submit_time >= args.interval_sec
            ):
                # 현재 버퍼를 복사해서 worker에 전달
                clip = list(frame_buffer)
                if detector is not None:
                    boxes = largest_person_boxes(list(box_buffer)) if CROP_LARGEST_PERSON else [b for bs in box_buffer for _, b in bs]   # 기본: 학습 때처럼 모든 사람 박스의 합집합
                    future = executor.submit(classifier.predict_pair, clip, [union_crop(f, boxes) for f in clip])
                else:
                    future = executor.submit(classifier.predict, clip)
                last_submit_time = now

            # 화면 표시
            if latest_result is not None:
                category, confidence, scores = latest_result
                thr_c = CATEGORY_THRESHOLDS.get(category, 1.0) if args.threshold is None else args.threshold
                if category not in ("정상", "판정 보류") and confidence > thr_c:
                    active_cat = category
                elif active_cat is not None and scores.get(active_cat, 0.0) >= RELEASE_SCORE:
                    category, confidence = active_cat, scores[active_cat]      # 켜진 경보 유지
                else:
                    active_cat = None
                held = active_cat is not None
            else:
                category, confidence, scores, held = None, 0.0, {}, False

            frame = draw_score_panel(
                frame, scores, category, confidence,
                args.threshold, classifier.is_ready, classifier.meta, held
            )

            # 하단 안내
            if len(buffer_times) >= 2:
                buffered_sec = buffer_times[-1] - buffer_times[0]
            else:
                buffered_sec = 0.0

            cv2.putText(
                frame,
                f"buffer: {buffered_sec:.1f}/{args.window_sec:.1f}s  "
                f"next infer: {max(0.0, args.interval_sec - (now-last_submit_time)):.1f}s",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1
            )
            cv2.putText(
                frame,
                "Q / ESC: quit",
                (frame.shape[1] - 170, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

            # 화면 FPS
            fps_count += 1
            if now - fps_t0 >= 1.0:
                display_fps = fps_count / (now - fps_t0)
                fps_count = 0
                fps_t0 = now

            cv2.putText(
                frame, f"camera fps: {display_fps:.1f}",
                (frame.shape[1] - 170, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

            cv2.imshow("StoreOps AI - WebCam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        score_file.close()
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()

    print("\n웹캠 테스트 종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
