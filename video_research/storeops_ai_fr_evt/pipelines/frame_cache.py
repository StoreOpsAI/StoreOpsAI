"""S3D 미세조정용 프레임 캐시: 영상을 3fps 표본으로 읽어 224x224로 줄인 프레임을 JPEG로 저장한다(영상 디코딩 반복을 피함).

저장: C:/data/features/frames224/<영상>.pkl = {"jpeg": [bytes, ...], "fps": 3.0}. 대상은 전 행동 공통 분할표의 학습 가능 클립(전도 헤드 기준, 격리 제외).
서비스 전처리와 같은 리사이즈(INTER_LINEAR). JPEG 품질은 학습·추론이 같은 압축을 거치게 고정한다(QUALITY).
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import cv2

from config.config import AIHUB_ORGANIZED_ROOT, BEHAVIOR_FEATURE_ROOT
from pipelines.fullscene_windows import SIZE, video_path
from pipelines.global_split import SPLIT_PATH
from pipelines.video_sampling import is_sample

OUT = Path(BEHAVIOR_FEATURE_ROOT) / "frames224"
QUALITY = 90


def cache_clip(video: Path, target: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    fps, raw, jpegs = cap.get(cv2.CAP_PROP_FPS), 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if is_sample(raw - 1, fps):
            ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            jpegs.append(buf.tobytes())
    cap.release()
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"jpeg": jpegs, "fps": 3.0}))
    tmp.replace(target)
    return len(jpegs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--with-validation", action="store_true")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")) if (r["official_split"] == "train" and r["q_fall"] != "True") or (args.with_validation and r["official_split"] == "validation")]
    OUT.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(sorted(rows, key=lambda r: r["clip"])[i::n], 1):
        target = OUT / f"{r['clip']}.pkl"
        if target.is_file():
            continue
        split = r["official_split"]
        video = next((p for p in (video_path(r["clip"], r["behavior"], s) for s in (split, "train", "validation")) if p.is_file()), None)
        if video is not None:
            cache_clip(video, target)
        if k % 50 == 0:
            print(f"[{k}] {r['clip']}", flush=True)
    print("FRAME_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
