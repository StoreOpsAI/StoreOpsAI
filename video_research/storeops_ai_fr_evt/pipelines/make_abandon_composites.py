"""정상 구매행동 영상에 유기 물건을 붙인 합성 영상을 만든다 (유기 재현율 측정용).

- 물건: 유기 train 라벨(object_abandon)의 마지막 박스를 GrabCut으로 오려 쓴다. 공식 validation은 쓰지 않는다.
- 바탕: 구매행동 정상 영상. 같은 바탕 영상을 붙이지 않은 원본 그대로 돌리면 오경보 측정이 된다.
- 장면 A(person): 어떤 사람의 발 위치에 그 순간부터 물건이 남는다. 사람 박스 안은 그리지 않아 사람이 가리다 떠나면 드러난다.
- 장면 B(no_person): 다른 시각에 사람이 섰던 바닥 위치에, 근처에 아무도 없는 순간 물건이 나타난다.
물건 크기는 그 위치의 사람 키에 비례시킨다. 결과: <output-dir>/*.mp4 + labels.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_object_boxes
from pipelines.eval_new_object import load_person_boxes
from pipelines.prepare_behaviors import list_videos

MIN_TAIL_SEC = 20  # 물건이 나타난 뒤 영상이 최소 이만큼 남아야 확정·확인까지 갈 수 있다
HEIGHT_RATIO = (0.15, 0.35)  # 물건 높이 / 사람 키


def cut_objects():
    """유기 train 영상마다 물건 하나: (BGR 패치, 0~1 알파, 출처 이름)."""
    items, _ = list_videos(WORKSPACE_ROOT / "유기", "유기")
    objects = []
    for xml, mp4, split in items:
        if split != "train":
            continue
        boxes = parse_object_boxes(xml)
        if not boxes:
            continue
        frame_no = max(boxes)
        cap = cv2.VideoCapture(str(mp4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        cap.release()
        if ok:
            objects.append((*_grabcut(frame, boxes[frame_no][0]), Path(mp4).stem))
    return objects


def _grabcut(frame, box):
    x1, y1, x2, y2 = box
    pad = max(10, (x2 - x1) // 4, (y2 - y1) // 4)
    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.grabCut(crop, mask, (x1 - cx1, y1 - cy1, x2 - x1, y2 - y1), None, None, 5, cv2.GC_INIT_WITH_RECT)
    alpha = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:  # 오려내기 실패 → 박스 전체
        return frame[y1:y2, x1:x2].copy(), np.ones((y2 - y1, x2 - x1), np.float32)
    patch, alpha = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1], alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return patch, cv2.GaussianBlur(alpha, (5, 5), 0)  # 가장자리를 살짝 풀어 붙인 티를 줄인다


def plan_placement(persons, fps, scenario, rng):
    """(나타나는 프레임, 발 위치 (x, 바닥 y), 사람 키) 또는 None."""
    last = len(persons) - int(MIN_TAIL_SEC * fps)
    feet = [(f, box) for f, boxes in enumerate(persons[:max(0, last)]) for box in boxes]
    if not feet:
        return None
    frame_no, (x1, y1, x2, y2) = rng.choice(feet)
    spot = ((x1 + x2) // 2, y2)
    if scenario == "person":
        return frame_no, spot, y2 - y1
    empty = [f for f in range(last) if not any(bx1 - (bx2 - bx1) <= spot[0] <= bx2 + (bx2 - bx1)
                                                 for bx1, _, bx2, _ in persons[f])]
    empty = [f for f in empty if abs(f - frame_no) > 5 * fps]  # 그 자리에 사람이 섰던 순간과는 떨어진 시각
    return (rng.choice(empty), spot, y2 - y1) if empty else None


def paste(frame, patch, alpha, box, person_boxes):
    x1, y1, x2, y2 = box
    a =alpha[..., None].copy()
    for px1, py1, px2, py2 in person_boxes:  # 사람이 앞을 가린다
        a[max(0, py1 - y1):max(0, py2 - y1), max(0, px1 - x1):max(0, px2 - x1)] = 0
    region = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (a * patch + (1 - a) * region).astype(np.uint8)


def make_one(video, obj, scenario, rng, output_dir):
    patch, alpha, source = obj
    persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    plan = plan_placement(persons, fps, scenario, rng)
    if plan is None:
        cap.release()
        return None
    appear, (fx, fy), person_h = plan
    scale = rng.uniform(*HEIGHT_RATIO) * person_h / patch.shape[0]
    patch, alpha = cv2.resize(patch, None, fx=scale, fy=scale), cv2.resize(alpha, None, fx=scale, fy=scale)
    ph, pw = patch.shape[:2]
    box = (fx - pw // 2, fy - ph, fx - pw // 2 + pw, fy)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(ph, pw) < 8 or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
        cap.release()
        return None  # 너무 작거나 화면 밖으로 나가는 배치는 라벨만 남고 물건이 없게 되므로 버린다
    name = f"{Path(video).stem}__{scenario}.mp4"
    writer = cv2.VideoWriter(str(output_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no >= appear:
            paste(frame, patch, alpha, box, persons[frame_no] if frame_no < len(persons) else [])
        writer.write(frame)
        frame_no += 1
    cap.release()
    writer.release()
    return {"video": name, "base": Path(video).name, "scenario": scenario, "object": source,
            "box": list(map(int, box)), "appear_frame": int(appear), "fps": fps}


def main():
    parser = argparse.ArgumentParser(description="정상 영상 + 유기 물건 합성")
    parser.add_argument("--base-dir", type=Path, default=WORKSPACE_ROOT / "구매행동" / "videos")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "abandon_composites")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    objects = cut_objects()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for video in sorted(args.base_dir.glob("*.mp4")):
        for scenario in ("person", "no_person"):
            label = make_one(video, rng.choice(objects), scenario, rng, args.output_dir)
            if label is None:
                print(f"건너뜀(놓을 자리 없음): {video.name} {scenario}")
                continue
            labels.append(label)
            print(label["video"], label["box"], label["appear_frame"])
    (args.output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"합성 {len(labels)}개, 물건 {len(objects)}종 → {args.output_dir}")


if __name__ == "__main__":
    main()
