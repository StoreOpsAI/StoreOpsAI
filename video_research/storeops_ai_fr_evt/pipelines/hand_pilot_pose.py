"""손 크롭 시험 1단계: 시험용 부분집합(유기 약 160, 매장이동 전체, 반품 30)의 포즈를 뽑는다. 절도는 이미 있는 캐시(output/pose_cache/절도/train)를 쓴다.

선택: 격리되지 않은 학습 영상을 날짜 fold·이름 순으로 정렬해 고르게 뽑는다(결정적). 출력은 extract_pose_cache와 같은 형식: output/pose_cache/<행동>/train/<영상>.npz.
공식 validation은 쓰지 않는다.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from config.config import AIHUB_ORGANIZED_ROOT, POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.global_split import SPLIT_PATH
from pipelines.lying_pose_extract import extract_clip

PICK = {"유기": 160, "매장이동": 999, "반품": 30}
POSE_ALL = __import__("os").getenv("POSE_ALL") == "1"
if POSE_ALL:   # 본 학습용: 격리되지 않은 모든 학습 영상
    PICK = {b: 10**6 for b in ("유기", "절도", "전도", "매장이동", "파손", "방화")}   # 반품 영상은 쓰지 않는다(사용자 지시 2026-10-07)
ROOT = APP_ROOT / "output" / "pose_cache"


def pick_clips() -> list[dict]:
    all_rows = list(csv.DictReader(open(SPLIT_PATH, encoding="utf-8")))
    rows = [r for r in all_rows if r["official_split"] == "train" and r["q_fall"] != "True"]
    if POSE_ALL:   # 공식 validation도 관절만 뽑는다(평가·튜닝에는 쓰지 않음, 격리 표시와 무관하게 전부)
        rows += [r for r in all_rows if r["official_split"] == "validation"]
    out = []
    for b, n in PICK.items():
        rs = sorted((r for r in rows if r["behavior"] == b), key=lambda r: (int(r["fold"]), r["clip"]))
        out += [rs[int(i)] for i in np.unique(np.linspace(0, len(rs) - 1, min(n, len(rs))).round().astype(int))]
    return out


def main() -> None:
    import os

    from ultralytics import YOLO

    shard, n_shard = (int(x) for x in os.getenv("POSE_SHARD", "0/1").split("/"))

    model = YOLO(str(APP_ROOT / POSE_WEIGHTS))
    clips = pick_clips()[shard::n_shard]
    start, done = time.time(), 0
    for i, r in enumerate(clips, 1):
        target = ROOT / r["behavior"] / r["official_split"] / f"{r['clip']}.npz"
        video = next((v for v in (Path(AIHUB_ORGANIZED_ROOT) / r["behavior"] / sp / "videos" / f"{r['clip']}.mp4" for sp in ("train", "validation")) if v.is_file()), Path("missing"))   # 반품은 일부가 validation 폴더에 있다
        if target.is_file() or not video.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **extract_clip(model, {"video": str(video)}))
        done += 1
        if done % 10 == 0:
            print(f"[{i}/{len(clips)}] 새로 {done}개 {time.time() - start:.0f}s", flush=True)
    print("HAND_POSE_DONE", len(clips), "개 대상", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
