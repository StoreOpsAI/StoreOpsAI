"""물건 탐지기 평가(크기별 재현율 + 이미지당 오경보): 평가용 날짜(test)의 정답 물건 박스 기준.

사용: python -m pipelines.eval_object_size --weights <best.pt> --imgsz 1920 [--data C:/data/object_ft3/H] [--split test]
정답: 정답 박스와 IoU>=0.5인 탐지가 있으면 적중. 점수 하한 CONFS마다 보고. 박스 크기 = sqrt(넓이)를 1920 가로 기준 px로 환산해 구간을 나눈다(해상도가 달라도 같은 기준).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

CONFS = (0.10, 0.25)
BINS = ((0, 30), (30, 52), (52, 90), (90, 1e9))   # 1920 기준 한 변(px) = 1280 기준 20·35·60에 해당


def iou(a, b) -> float:
    iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--data", default="C:/data/object_ft3/H")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    imgs = sorted(glob.glob(f"{args.data}/images/{args.split}/*.jpg"))
    gt_hits = {c: [] for c in CONFS}
    fp = {c: 0 for c in CONFS}
    for p in imgs:
        res = model.predict(p, imgsz=args.imgsz, conf=min(CONFS), verbose=False)[0]
        h, w = res.orig_shape
        boxes, conf = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
        gts = []
        for line in Path(p.replace("images", "labels").replace(".jpg", ".txt")).read_text().split("\n"):
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                gts.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        for c in CONFS:
            keep = boxes[conf >= c]
            for g in gts:
                side = np.sqrt((g[2] - g[0]) * (g[3] - g[1])) * 1920 / w
                gt_hits[c].append((side, any(iou(g, q) >= 0.5 for q in keep)))
            fp[c] += sum(1 for q in keep if not any(iou(g, q) >= 0.5 for g in gts))
    print(f"가중치 {args.weights} imgsz {args.imgsz} | {args.split} 이미지 {len(imgs)}장")
    for c in CONFS:
        rows = np.array(gt_hits[c], dtype=float)
        parts = ", ".join(f"{lo}~{hi if hi < 1e8 else '∞'}px {int(((rows[:, 0] >= lo) & (rows[:, 0] < hi)).sum())}개 {rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi), 1].mean():.2f}" for lo, hi in BINS if ((rows[:, 0] >= lo) & (rows[:, 0] < hi)).any())
        print(f"  점수>={c}: 전체 재현율 {rows[:, 1].mean():.3f} ({len(rows)}개) | 크기별 [{parts}] | 정답 아닌 탐지 {fp[c]}개 ({fp[c] / max(len(imgs), 1):.2f}/이미지)")


if __name__ == "__main__":
    main()
