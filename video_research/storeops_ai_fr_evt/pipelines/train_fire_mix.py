"""불 검출기 3차 학습: 2차(방화 표기 + 어려운 음성)에 공개 불 데이터와 합성 불을 더해 과적합·망각을 줄인다.

- D-Fire(CC0 1.0, de Venâncio 외 NCA 2022) 불 사진 DFIRE_FIRE장 + 불 없는 사진 DFIRE_NONE장: 다양한 불 모양 복습
- 합성(Ghiasi 외 "Simple Copy-Paste", CVPR 2021 응용): D-Fire 불을 평가 fold 밖 매장 영상의 불 없는 프레임에
  작은 크기(SYNTH_HEIGHT px)로 붙인다. D-Fire 라벨은 박스뿐이라 SAM(Kirillov 외 ICCV 2023, MobileSAM)에 박스를 주어
  불 모양 마스크를 얻고 마스크 픽셀만 붙인다(박스째 붙이면 주변 배경까지 네모로 붙어 '네모 = 불'을 배울 위험)
- 어려운 음성은 train_fire 2차 실행의 채굴 캐시(fire_cv2/fold*/stage1/mine)를 그대로 쓴다(같은 선택 규칙)
- 평가는 train_fire와 같다(공통 분할 fold k만). 개수·크기는 사전 값, 공식 validation 미사용
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT
from pipelines.make_splits import FOLDS, load_splits
from pipelines.prepare_behaviors import list_videos
from pipelines.train_fire import FIRE_CLASS_ID, _total, evaluate, load_samples, mine_negatives, train_fold

DFIRE_DIR = WORKSPACE_ROOT / "datasets" / "dfire" / "data"
DFIRE_FIRE, DFIRE_NONE = 800, 200
SYNTH_HEIGHT = (16, 96)  # 붙이는 불 높이(px, 1080p 기준). 방화 영상 불 크기 범위
MAX_PATCHES = 300


def load_dfire(rng, with_patches=True):
    """([(이미지, [(클래스, cx, cy, w, h)])], [(불 조각, 마스크)]). 불 사진·불 없는 사진을 무작위로 고른다.
    with_patches=False면 합성용 조각(SAM)을 만들지 않는다."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(DFIRE_DIR / "*.parquet")))
    index = [(f, i, label) for f in files
             for i, label in enumerate(pq.read_table(f, columns=["label"]).column("label").to_pylist())]
    has_fire = lambda label: any(l.split()[0] == "1" for l in label.splitlines() if l.strip())
    picked = rng.sample([x for x in index if has_fire(x[2])], DFIRE_FIRE) + \
        rng.sample([x for x in index if not x[2].strip()], DFIRE_NONE)
    from ultralytics import SAM

    sam = SAM(str(APP_ROOT / "mobile_sam.pt")) if with_patches else None
    items, patches = [], []
    for f in files:
        rows = {i: label for g, i, label in picked if g == f}
        if not rows:
            continue
        images = pq.read_table(f, columns=["image"]).column("image").to_pylist()
        for i, label in rows.items():
            image = cv2.imdecode(np.frombuffer(images[i]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            boxes = [tuple(map(float, l.split())) for l in label.splitlines() if l.strip()]
            items.append((image, boxes))
            h, w = image.shape[:2]
            fire = [(max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h)), int((cx + bw / 2) * w), int((cy + bh / 2) * h))
                    for c, cx, cy, bw, bh in boxes if c == FIRE_CLASS_ID]
            fire = [b for b in fire if b[2] - b[0] >= 8 and b[3] - b[1] >= 8]
            if sam is None or not fire or len(patches) >= MAX_PATCHES:
                continue
            masks = sam(image, bboxes=fire, verbose=False)[0].masks.data.cpu().numpy()
            for (x1, y1, x2, y2), mask in zip(fire, masks):
                m = mask[y1:y2, x1:x2].astype(np.float32)
                if m.mean() > 0.05:  # 마스크가 거의 비면 버린다
                    patches.append((image[y1:y2, x1:x2].copy(), m))
    return items, patches


def paste(background, patch, rng):
    """(불 조각, 마스크)를 화면 아래쪽 60%(바닥) 임의 위치에 마스크 픽셀만 붙이고, 붙은 마스크의 YOLO 박스를 돌려준다."""
    image, mask = patch
    h, w = background.shape[:2]
    height = rng.randint(*SYNTH_HEIGHT)
    size = (max(4, int(image.shape[1] * height / image.shape[0])), height)
    image, mask = cv2.resize(image, size), cv2.resize(mask, size)
    ph, pw = image.shape[:2]
    x, y = rng.randint(0, w - pw - 1), rng.randint(int(h * 0.4), h - ph - 1)
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.0)[..., None]  # 마스크 경계만 살짝 부드럽게
    region = background[y:y + ph, x:x + pw].astype(np.float32)
    background[y:y + ph, x:x + pw] = (alpha * image + (1 - alpha) * region).astype(np.uint8)
    ys, xs = np.nonzero(mask > 0.5)
    x1, y1, x2, y2 = x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1
    return (FIRE_CLASS_ID, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def synth_items(videos, patches, rng, count):
    items = []
    for _ in range(count):
        video = rng.choice(videos)
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, rng.randrange(max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))))
        ok, frame = cap.read()
        cap.release()
        if ok:
            items.append((frame, [paste(frame, rng.choice(patches), rng) for _ in range(rng.randint(1, 2))]))
    return items


def write_images(items, folder):
    """(이미지, [(클래스, cx, cy, w, h)]) 목록을 YOLO 이미지·라벨로 쓴다."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    paths = []
    for n, (image, boxes) in enumerate(items):
        path = folder / "images" / f"extra_{n:05d}.jpg"
        cv2.imencode(".jpg", image)[1].tofile(str(path))
        (folder / "labels" / f"extra_{n:05d}.txt").write_text(
            "\n".join(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes), encoding="utf-8")
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description="불 검출기 3차: D-Fire + 합성 + 방화 표기 + 어려운 음성 (공통 5-fold)")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "fire_cv3")
    parser.add_argument("--mine-dir", type=Path, default=APP_ROOT / "output" / "fire_cv2")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--synth", type=int, default=0,
                        help="합성 이미지 수. 기본 0: D-Fire 불이 과노출 흰 덩어리가 많아 효과를 따로 비교한 뒤 쓴다")
    args = parser.parse_args()

    rng = random.Random(0)
    splits = load_splits()
    fold_of = lambda video: splits[Path(video).name]["fold"]
    videos = load_samples()
    others = [mp4 for name in ("절도", "파손", "유기")
              for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]
    others += sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))
    dfire, patches = load_dfire(rng, with_patches=args.synth > 0)
    print(f"D-Fire {len(dfire)}장, 불 조각 {len(patches)}개")

    summary = {"stage3": []}
    for k in range(FOLDS):
        folder = args.output_dir / f"fold{k}"
        train = [v for v in videos if fold_of(v["video"]) != k]
        dev = [v for v in videos if fold_of(v["video"]) == k]
        train_others = [v for v in others if fold_of(v) != k]
        dev_others = [v for v in others if fold_of(v) == k]
        negatives = mine_negatives(None, train_others, args.mine_dir / f"fold{k}" / "stage1" / "mine", seed=k)
        extra = write_images(dfire + synth_items(train_others, patches, random.Random(k), args.synth), folder / "extra")
        model = train_fold(train + negatives, folder, args.epochs, extra_paths=extra)
        summary["stage3"].append({"fold": k, **evaluate(model, "stage3", train, dev, dev_others, folder)})
        print(json.dumps(summary["stage3"][-1], ensure_ascii=False))
        (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["total"] = {"fire": _total(summary["stage3"], "dev_fire_videos"),
                        "other": _total(summary["stage3"], "dev_other_videos")}
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["total"], ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
