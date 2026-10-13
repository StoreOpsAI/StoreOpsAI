"""C′안 입력 캐시: 창(4초, 12표본, 간격 2초)마다 사람 박스 합집합을 넓혀 자른 224x224 프레임을 JPEG로 저장한다(원본 해상도에서 자름).

- 크롭: 창 안 모든 사람 박스의 합집합을 CROP_PAD(사방 폭·높이의 50%)만큼 넓힌 정사각형. 한 변은 화면 높이의 MIN_SIDE_RATIO 이상(작은 사람을 과도하게 확대하지 않음).
  화면 밖으로 나가면 안쪽으로 밀어 넣는다. 창 안에 사람이 한 명도 없으면 화면 전체를 쓴다(서비스도 같은 규칙).
- 배경 의존을 줄이기 위해 사람 중심으로만 보여 준다. 사람 박스는 서비스와 같은 설정의 검출·추적(persons/)에서 온다.
- OVERLAY=True면 물건 탐지 박스(objdet/)를 원본 프레임에 밝은 테두리로 그린 뒤 자른다(C2 변형). 기본은 크롭만(C1).
출력: C:/data/features/crops_<c1|c2>/<영상>.pkl = {start: [jpeg x12], "region": {start: (x, y, side)}}
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.fullscene_windows import STRIDE, WIN, video_path
from pipelines.global_split import SPLIT_PATH
from pipelines.video_sampling import is_sample

ROOT = Path(BEHAVIOR_FEATURE_ROOT)
SIZE, CROP_PAD, MIN_SIDE_RATIO, QUALITY = 224, 0.5, 0.25, 90
OVERLAY_COLOR = (0, 255, 255)  # BGR 노랑 테두리


def window_region(boxes: np.ndarray, shape: tuple[int, int]) -> tuple[float, float, float]:
    """사람 박스 합집합을 넓힌 정사각형 (x, y, side). 박스가 없으면 화면 전체(짧은 변 기준 정사각형이 아니라 전체 프레임을 쓰므로 side<=0으로 표시)."""
    h, w = shape
    if len(boxes) == 0:
        return 0.0, 0.0, -1.0
    x1, y1, x2, y2 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    side = max((x2 - x1) * (1 + 2 * CROP_PAD), (y2 - y1) * (1 + 2 * CROP_PAD), MIN_SIDE_RATIO * h)
    side = min(side, float(min(h, w)))
    cx = float(np.clip((x1 + x2) / 2, side / 2, w - side / 2))
    cy = float(np.clip((y1 + y2) / 2, side / 2, h - side / 2))
    return cx - side / 2, cy - side / 2, side


def crop_frame(frame: np.ndarray, region: tuple[float, float, float]) -> np.ndarray:
    x, y, side = region
    patch = frame if side <= 0 else frame[int(y):int(y + side), int(x):int(x + side)]
    return cv2.resize(patch, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def draw_objects(frame: np.ndarray, boxes: np.ndarray, conf: np.ndarray, min_conf: float) -> np.ndarray:
    out = frame.copy()
    for b, c in zip(boxes, conf):
        if c >= min_conf:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), OVERLAY_COLOR, 3)
    return out


def build_clip(video: Path, persons: dict, objects: dict | None, min_conf: float) -> dict:
    cap = cv2.VideoCapture(str(video))
    fps, raw, frames = cap.get(cv2.CAP_PROP_FPS), 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if is_sample(raw - 1, fps):
            frames.append(frame)
    cap.release()
    n = len(frames)
    shape = frames[0].shape[:2]
    p_by = {int(f): persons["box"][persons["frame"] == f] for f in np.unique(persons["frame"])}
    o_by = {int(f): (objects["box"][objects["frame"] == f], objects["conf"][objects["frame"] == f]) for f in np.unique(objects["frame"])} if objects else {}
    out: dict = {"region": {}}
    for s in range(0, n - WIN + 1, STRIDE):
        boxes = np.concatenate([p_by[f] for f in range(s, s + WIN) if f in p_by] or [np.zeros((0, 4), np.float32)])
        region = window_region(boxes, shape)
        jp = []
        for f in range(s, s + WIN):
            img = draw_objects(frames[f], *o_by[f], min_conf) if objects and f in o_by else frames[f]
            ok, buf = cv2.imencode(".jpg", crop_frame(img, region), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            jp.append(buf.tobytes())
        out[s], out["region"][s] = jp, region
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--overlay", action="store_true", help="물건 탐지 박스를 그려 넣는다(C2)")
    parser.add_argument("--min-conf", type=float, default=0.25, help="C2에서 그릴 물건 점수 하한(TEMPORARY_DEFAULT, 오경보 후보 수 조절)")
    parser.add_argument("--with-validation", action="store_true")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    out_dir = ROOT / ("crops_c2" if args.overlay else "crops_c1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))
            if (r["official_split"] == "train" and r["q_fall"] != "True") or (args.with_validation and r["official_split"] == "validation")]
    for k, r in enumerate(sorted(rows, key=lambda r: r["clip"])[i::n], 1):
        target, pf = out_dir / f"{r['clip']}.pkl", ROOT / "persons" / f"{r['clip']}.npz"
        if target.is_file() or not pf.is_file():
            continue
        video = next((p for p in (video_path(r["clip"], r["behavior"], s) for s in (r["official_split"], "train", "validation")) if p.is_file()), None)
        if video is None:
            continue
        of = ROOT / "objdet" / f"{r['clip']}.npz"
        objects = dict(np.load(of)) if args.overlay and of.is_file() else None
        data = build_clip(video, dict(np.load(pf)), objects, args.min_conf)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(data))
        tmp.replace(target)
        if k % 50 == 0:
            print(f"[{k}] {r['clip']}", flush=True)
    print("CROP_CACHE_DONE", flush=True)


def _self_check() -> None:
    boxes = np.array([[100, 200, 160, 400], [300, 220, 360, 420]], np.float32)
    x, y, side = window_region(boxes, (1080, 1920))
    assert side >= 0.25 * 1080 - 1 and x >= 0 and y >= 0 and x + side <= 1920 and y + side <= 1080       # 화면 안, 최소 크기
    assert x <= 100 and x + side >= 360                                                                   # 사람 둘 모두 포함
    assert window_region(np.zeros((0, 4), np.float32), (1080, 1920))[2] < 0                              # 사람이 없으면 화면 전체
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert crop_frame(frame, (x, y, side)).shape == (SIZE, SIZE, 3) and crop_frame(frame, (0, 0, -1)).shape == (SIZE, SIZE, 3)
    drawn = draw_objects(frame, np.array([[10, 10, 50, 50]], np.float32), np.array([0.9], np.float32), 0.25)
    assert drawn.sum() > 0 and frame.sum() == 0                                                           # 원본은 그대로, 복사본에만 그림
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
