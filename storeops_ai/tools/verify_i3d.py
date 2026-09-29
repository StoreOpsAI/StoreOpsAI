"""학습된 행동분류 모델만 단독으로 확인하는 도구 (YOLO/추적/사건 발급 없이).

파이프라인 전체를 돌리기 전에 '모델이 로드되고 점수가 나오는지'부터 볼 때 쓴다.
프로젝트 루트에서 실행:

    # 1) 체크포인트 정보만 확인
    python -m tools.verify_i3d

    # 2) 영상 한 개를 통째로 1회 분류
    python -m tools.verify_i3d --video sample.mp4

    # 3) 슬라이딩 윈도우로 구간별 분류 (예: 30초 창을 5초씩 이동)
    python -m tools.verify_i3d --video sample.mp4 --window-sec 30 --stride-sec 5

    # 4) extract_multiclass.py 가 만든 클립 폴더(frame_000001.jpg ...)를 그대로 분류 (학습 형식과 동일한 입력)
    python -m tools.verify_i3d --frames-dir path/to/clip_folder
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2

from config.config import I3D_BUFFER_FPS, I3D_WEIGHT_PATH
from models.i3d_classifier import I3DClassifier


def _fmt(scores):
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return "  ".join(f"{k} {v:.3f}" for k, v in ranked if v >= 0.005)


def _read_video(path, buffer_fps):
    """분류용으로 buffer_fps 간격 프레임만 읽는다 (파이프라인의 full 모드와 같은 방식). (시각, BGR 프레임) 목록."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(fps / buffer_fps)))
    items, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            items.append((n / fps, frame))
        n += 1
    cap.release()
    return items, n / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=None, help=f"체크포인트 경로 (기본: {I3D_WEIGHT_PATH})")
    ap.add_argument("--video", help="분류할 영상 파일")
    ap.add_argument("--frames-dir", help="frame_*.jpg 가 들어있는 클립 폴더")
    ap.add_argument("--window-sec", type=float, default=None, help="구간 길이(초). 생략하면 영상 전체")
    ap.add_argument("--stride-sec", type=float, default=None, help="구간 이동 간격(초). 생략하면 window-sec")
    ap.add_argument("--buffer-fps", type=float, default=I3D_BUFFER_FPS, help="분류용 프레임 샘플링 속도")
    args = ap.parse_args()

    clf = I3DClassifier(args.weight)
    if not clf.is_ready:
        sys.exit("모델이 로드되지 않았습니다. --weight 경로를 확인하세요.")
    print(f"모델: {clf.weight_path}")
    print(f"  arch={clf.meta['arch']}  학습 시 val_acc={clf.meta['val_acc']}  epoch={clf.meta['epoch']}  "
          f"device={clf.meta['device']}")
    print(f"  입력: {clf.num_frames}프레임 x {clf.input_size}x{clf.input_size}")
    print(f"  출력 카테고리: {sorted(set(clf._index_to_category.values()))}")
    print(f"  학습하지 않은 카테고리(점수 0 고정): {clf.missing_categories}")

    if args.frames_dir:
        files = sorted(glob.glob(str(Path(args.frames_dir) / "frame_*.jpg")))
        if not files:
            sys.exit(f"frame_*.jpg 가 없습니다: {args.frames_dir}")
        frames = [cv2.imread(f) for f in files]
        cat, conf, scores = clf.predict(frames)
        print(f"\n{Path(args.frames_dir).name} ({len(frames)}프레임) -> {cat} {conf:.3f}")
        print("  " + _fmt(scores))
        return

    if not args.video:
        return

    items, total_sec = _read_video(args.video, args.buffer_fps)
    print(f"\n영상: {args.video}  길이 {total_sec:.1f}s  분류용 프레임 {len(items)}장 ({args.buffer_fps:g}fps)")
    win = args.window_sec or total_sec
    stride = args.stride_sec or win
    t0 = 0.0
    while True:
        t1 = min(t0 + win, total_sec)
        chunk = [f for t, f in items if t0 <= t <= t1]
        if len(chunk) >= 2:
            cat, conf, scores = clf.predict(chunk)
            print(f"[{t0:6.1f}s ~ {t1:6.1f}s] {cat} {conf:.3f} | {_fmt(scores)}")
        if t0 + win >= total_sec:
            break
        t0 += stride


if __name__ == "__main__":
    main()
