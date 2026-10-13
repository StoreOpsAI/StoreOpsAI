"""행동 영상(organized/<행동>/<split>/videos)에서 사람 박스·관절(YOLO11s-pose)을 뽑아 output/pose_cache/<행동>/<split>/<영상>.npz로 저장한다.

저장 형식은 전도 포즈 캐시(lying_pose_extract.extract_clip)와 같다: frame, box, box_conf, kpt(17x3), n_frames.
meta.json에 추출기 해시·입력 크기·후보 신뢰도·git 커밋·fps를 남긴다. 이미 있는 캐시는 건너뛴다.
--verify: 영상마다 캐시가 있고 n_frames가 라벨 xml의 프레임 수와 일치하는지 검사해 불일치 목록을 출력한다(삭제는 별도로 사람이 확인).
공식 validation은 특징만 뽑고 평가·튜닝에 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from config.config import AIHUB_ORGANIZED_ROOT, POSE_IMGSZ, POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_pose_extract import PERSON_CONF, extract_clip

ROOT = APP_ROOT / "output" / "pose_cache"


def xml_frames(label: Path) -> int:
    return int(ET.parse(label).getroot().find("meta/task/size").text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    base = Path(AIHUB_ORGANIZED_ROOT) / args.behavior / args.split
    videos = sorted((base / "videos").glob("*.mp4"))
    out = ROOT / args.behavior / args.split
    out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        bad = []
        for v in videos:
            cache, label = out / f"{v.stem}.npz", base / "labels" / f"{v.stem}.xml"
            if not cache.is_file() or (label.is_file() and int(np.load(cache)["n_frames"]) != xml_frames(label)):
                bad.append(v.stem)
        print(f"{args.behavior} {args.split}: 영상 {len(videos)}개, 검증 실패 {len(bad)}개", bad[:5])
        return
    from ultralytics import YOLO

    weights = APP_ROOT / POSE_WEIGHTS
    meta = {"extractor": POSE_WEIGHTS, "extractor_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "imgsz": POSE_IMGSZ,
            "candidate_person_conf": PERSON_CONF, "fps": 3.0, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=APP_ROOT).stdout.strip()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    model = YOLO(str(weights))
    start = time.time()
    for i, v in enumerate(videos, 1):
        cache = out / f"{v.stem}.npz"
        if not cache.is_file():
            np.savez_compressed(cache, **extract_clip(model, {"video": str(v)}))
            print(f"[{i}/{len(videos)}] {v.stem} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
