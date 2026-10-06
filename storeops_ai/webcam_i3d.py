"""
StoreOps AI - 학습된 S3D 행동분류 모델 웹캠 테스트

사용법 (프로젝트 루트에서):
    python webcam_i3d.py

기본:
    - training/runs/mc_stage2/best.pt 사용
    - 웹캠 전체 화면을 학습과 동일한 방식으로 입력
    - 3 FPS로 프레임을 버퍼링
    - 최근 8초를 24프레임으로 균일 샘플링
    - 2초마다 추론
    - 0.60 이상이면 EVENT 표시

종료: q 또는 ESC
"""

from __future__ import annotations

import argparse
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

from models.i3d_classifier import I3DClassifier, I3DNotConfiguredError


DISPLAY_NAMES = {
    "정상": "NORMAL",
    "쓰러짐": "FALL",
    "파손": "BROKEN",
    "쓰레기 투기": "LITTERING",
    "싸움": "FIGHT",
}


def parse_args():
    p = argparse.ArgumentParser(description="학습된 S3D/I3D 계열 모델 웹캠 테스트")
    p.add_argument("--camera", type=int, default=0, help="웹캠 번호. 기본 0")
    p.add_argument(
        "--weight",
        default=str(ROOT / "training" / "runs" / "mc_stage2" / "best.pt"),
        help="학습된 best.pt 경로",
    )
    p.add_argument("--window-sec", type=float, default=8.0,
                   help="한 번 분류할 최근 영상 길이. 기본 8초")
    p.add_argument("--interval-sec", type=float, default=2.0,
                   help="추론 간격. 기본 2초")
    p.add_argument("--buffer-fps", type=float, default=3.0,
                   help="분류 버퍼에 저장할 FPS. 학습 데이터와 동일하게 기본 3")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="이상행동 EVENT 표시 임계값. 기본 0.60")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def draw_score_panel(frame, scores, category, confidence, threshold, ready, meta):
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
        event = category not in ("정상", "판정 보류") and confidence > threshold
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

        for c in ["정상", "쓰러짐", "싸움", "파손", "쓰레기 투기"]:
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
        print("기본 모델: training/runs/mc_stage2/best.pt")
        return 1

    print("=" * 70)
    print("StoreOps AI - 학습된 행동분류 모델 웹캠 테스트")
    print("=" * 70)
    print(f"model      : {weight}")
    print(f"camera     : {args.camera}")
    print(f"window     : {args.window_sec:.1f} sec")
    print(f"infer      : every {args.interval_sec:.1f} sec")
    print(f"buffer fps : {args.buffer_fps:.1f}")
    print(f"threshold  : {args.threshold:.2f}")
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
    max_frames = max(classifier.num_frames, int(round(args.window_sec * args.buffer_fps)) + 2)

    frame_buffer = deque(maxlen=max_frames)
    buffer_times = deque(maxlen=max_frames)

    last_sample_time = 0.0
    last_submit_time = 0.0
    latest_result = None
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
                buffer_times.append(now)
                last_sample_time = now

            # 완료된 추론 결과 반영
            if future is not None and future.done():
                try:
                    latest_result = future.result()
                    latest_result_time = now
                except Exception as e:
                    print(f"[I3D ERROR] {type(e).__name__}: {e}")
                future = None

            # 충분한 시간의 프레임이 쌓였을 때만 추론
            enough_window = (
                len(frame_buffer) >= classifier.num_frames
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
                future = executor.submit(classifier.predict, clip)
                last_submit_time = now

            # 화면 표시
            if latest_result is not None:
                category, confidence, scores = latest_result
            else:
                category, confidence, scores = None, 0.0, {}

            frame = draw_score_panel(
                frame, scores, category, confidence,
                args.threshold, classifier.is_ready, classifier.meta
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
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()

    print("\n웹캠 테스트 종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
