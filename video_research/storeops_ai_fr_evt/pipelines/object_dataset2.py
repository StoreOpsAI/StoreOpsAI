"""2분할 교차 적합용 작은 물건 탐지기 데이터셋: 유기(object_abandon) + 절도(object_article) 물건 박스를 한 클래스(object)로 합친다.

- 분할: 날짜 fold {0,1,2}=묶음 A, {3,4}=묶음 B. 모델 trainA는 A로 학습해 B에 추론하고, trainB는 B로 학습해 A에 추론한다(자기 클립을 학습한 모델이 그 클립에 추론하지 않는다).
- 학습 데이터에서 날짜의 10%(해시)는 체크포인트 선택용 val로 뗀다(같은 날짜는 train과 val에 걸치지 않는다).
- 음성 프레임: 같은 묶음의 전도·반품·매장이동 영상에서 물건 박스가 없는 프레임을 NEG_FRAMES개 뽑아 빈 라벨로 넣는다(오경보 억제).
- 공식 validation·격리 클립은 쓰지 않는다(유기·절도는 q_abandon·q_theft 둘 다 False인 클립). 프레임은 OBJECT_FT_FRAME_STRIDE 간격, 긴 변 OBJECT_FT_IMG_SIDE.
출력: C:/data/object_ft2/<A|B>/{images,labels}/{train,val}, data.yaml, split.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

from config.config import AIHUB_ORGANIZED_ROOT, OBJECT_FT_FRAME_STRIDE, OBJECT_FT_IMG_SIDE
from pipelines.global_split import SPLIT_PATH

ROOT = Path(os.getenv("OBJECT_FT_DS_ROOT", "C:/data/object_ft2"))
GROUPS = {"A": {0, 1, 2}, "B": {3, 4}, "H": {0, 1, 2, 3, 4}}   # H: 8:2 보류 구성(모든 fold에서 학습, 평가용 날짜는 fold 3·4에서 뗀다 = 기존 모델 A가 본 적 없는 날짜)
TEST_FOLDS, TEST_PERCENT = {3, 4}, 20
LABEL = {"유기": "object_abandon", "절도": "object_article"}
NEG_SOURCES = ("전도", "반품", "매장이동")
NEG_FRAMES = 1200
VAL_PERCENT = 10


def read_boxes(xml: Path, label: str) -> dict[int, list[list[float]]]:
    boxes: dict[int, list[list[float]]] = defaultdict(list)
    for track in ET.parse(xml).getroot().iter("track"):
        if track.get("label") == label:
            for box in track.iter("box"):
                if box.get("outside") == "0":
                    boxes[int(box.get("frame"))].append([float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")])
    return boxes


def val_dates(per_date: dict[str, int], percent: int = VAL_PERCENT, salt: str = "objval_") -> set[str]:
    """날짜를 해시 순서로 훑어 이미지가 전체의 VAL_PERCENT%가 될 때까지 val 날짜로 정한다(같은 날짜는 train과 val에 걸치지 않는다)."""
    total, chosen, acc = sum(per_date.values()), set(), 0
    for d in sorted(per_date, key=lambda d: hashlib.md5((salt + d).encode()).hexdigest()):
        if acc >= total * percent / 100:
            break
        chosen.add(d); acc += per_date[d]
    return chosen


def video_of(behavior: str, clip: str) -> Path:
    for split in ("train", "validation"):
        p = Path(AIHUB_ORGANIZED_ROOT) / behavior / split / "videos" / f"{clip}.mp4"
        if p.is_file():
            return p
    raise FileNotFoundError(clip)


def export_frames(video: Path, frames: list[int], boxes: dict[int, list[list[float]]], name: str, img_dir: Path, lbl_dir: Path) -> int:
    cap, done = cv2.VideoCapture(str(video)), 0
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = OBJECT_FT_IMG_SIDE / max(h, w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])  # imwrite는 한글 경로에 저장하지 못한다
        (img_dir / f"{name}_{f:03d}.jpg").write_bytes(buf.tobytes())
        lines = [f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}" for b in boxes.get(f, [])]
        (lbl_dir / f"{name}_{f:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
        done += 1
    cap.release()
    return done


def build(model: str, dry: bool) -> dict:
    """model A: A 묶음 클립으로 학습(추론 대상은 B). model B: 반대."""
    train_folds = GROUPS[model]
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")) if r["official_split"] == "train" and int(r["fold"]) in train_folds]
    pos = [r for r in rows if r["behavior"] in LABEL and r["q_abandon"] != "True" and r["q_theft"] != "True"]
    negs = [r for r in rows if r["behavior"] in NEG_SOURCES and r["q_fall"] != "True"]
    counts = {"train": 0, "val": 0, "test": 0}
    plan: list[tuple[dict, list[int], dict]] = []
    for r in pos:
        boxes = read_boxes(Path(AIHUB_ORGANIZED_ROOT) / r["behavior"] / "train" / "labels" / f"{r['clip']}.xml", LABEL[r["behavior"]])
        frames = sorted(f for f in boxes if f % OBJECT_FT_FRAME_STRIDE == 0)
        if frames:
            plan.append((r, frames, boxes))
    rng = random.Random(0)
    rng.shuffle(negs)
    per = max(1, NEG_FRAMES // max(len(negs), 1) + 1)
    for r in negs[:NEG_FRAMES]:
        n_frames = 180 if r["behavior"] != "매장이동" else 900
        plan.append((r, sorted(rng.sample(range(0, n_frames, OBJECT_FT_FRAME_STRIDE), per)), {}))
    per_date: dict[str, int] = defaultdict(int)
    for r, frames, _ in plan:
        per_date[r["date"]] += len(frames)
    tdates: set[str] = set()
    if model == "H":   # 평가용(test): fold 3·4 날짜의 TEST_PERCENT%(해시 순서). 날짜 단위라 같은 날짜가 학습과 평가에 걸치지 않는다
        fold34 = {r["date"] for r, _, _ in plan if int(r["fold"]) in TEST_FOLDS}
        tdates = val_dates({d: n for d, n in per_date.items() if d in fold34}, TEST_PERCENT, "objtest_")
    vdates = val_dates({d: n for d, n in per_date.items() if d not in tdates})
    plan = [(r, "test" if r["date"] in tdates else "val" if r["date"] in vdates else "train", frames, boxes) for r, frames, boxes in plan]
    for r, role, frames, _ in plan:
        counts[role] += len(frames)
    if dry:
        return {"model": model, "clips": len(plan), **counts}
    root = ROOT / model
    for role in counts:
        (root / "images" / role).mkdir(parents=True, exist_ok=True)
        (root / "labels" / role).mkdir(parents=True, exist_ok=True)
    for i, (r, role, frames, boxes) in enumerate(plan, 1):
        export_frames(video_of(r["behavior"], r["clip"]), frames, boxes, r["clip"], root / "images" / role, root / "labels" / role)
        if i % 100 == 0:
            print(f"[{model}] {i}/{len(plan)}", flush=True)
    yaml = [f"path: {root.as_posix()}", "train: images/train", "val: images/val"] + (["test: images/test"] if model == "H" else []) + ["names:", "  0: object"]
    (root / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (root / "split.json").write_text(json.dumps({"train_folds": sorted(train_folds), "predict_folds": sorted(set(range(5)) - train_folds), "images": counts, "test_dates": sorted(tdates), "val_dates": sorted(vdates)}, indent=1), encoding="utf-8")
    return {"model": model, "clips": len(plan), **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B", "H"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(build(args.model, args.dry_run), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
