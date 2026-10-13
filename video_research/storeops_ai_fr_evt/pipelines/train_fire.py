"""불 검출기 파인튜닝 + 어려운 음성 채굴(OHEM, Shrivastava 외 CVPR 2016 응용).

D-Fire 사전학습 YOLO(FIRE_BASE_WEIGHTS)를 방화 train 수동 불 박스(data/fire_boxes.json)로 학습하고,
다른 행동·반품 영상에서 1차 모델이 불로 착각한 프레임을 '불 아님'으로 넣어 2차 학습한다.

누수·과적합 방지:
- 분할은 모든 모듈 공통(data/splits.json, 촬영 날짜 그룹 5-fold). fold k 평가 때 fold k 영상은 학습·채굴에 안 씀
- 학습 이미지 = 학습 fold의 표기 프레임 + 사건 시작 1초 전까지의 프레임(사람 시간 라벨상 불 없음). '?' 프레임은 제외
- 증강은 ultralytics 기본값으로 학습 쪽에만. val=False, 마지막 가중치 사용(dev로 에폭·가중치를 고르지 않음)
- 점수 하한 FIRE_CONF, 연속 FIRE_PERSIST_SEC, 채굴 개수는 사전에 정한 값. 공식 validation 미사용
평가(fold k): 방화 사건 재현율·사건 시작 전 오경보, 다른 행동·반품 영상 분당 오경보, 표기 프레임 적중률(train-dev 차이).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

from config.config import FIRE_BASE_WEIGHTS, FIRE_CLASSES, FIRE_CONF, FIRE_IMGSZ
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.eval_fire import fire_alarms, load_fire_detections, score_alarms
from pipelines.make_splits import FOLDS, load_splits
from pipelines.prepare_behaviors import list_videos

NEG_PER_VIDEO = 5
HARD_PER_VIDEO = 5   # 다른 영상 하나에서 모을 착각 프레임 최대 수(점수 높은 순)
EASY_PER_VIDEO = 1   # 다양성을 위한 무작위 불 아님 프레임 수
FIRE_CLASS_ID = 1  # FIRE_BASE_WEIGHTS의 fire 클래스 번호


def cell_box(cells, x0, y0):
    """'C5:H9' (칸 = 20px) → 원본 좌표 박스."""
    a, b = cells.split(":")
    return (x0 + (ord(a[0]) - 65) * 20, y0 + int(a[1:]) * 20, x0 + (ord(b[0]) - 64) * 20, y0 + (int(b[1:]) + 1) * 20)


def load_samples():
    """영상별 {'video', 'events', 'frames': [(프레임, 박스 목록)]}. 박스 목록이 비면 불 없음."""
    labels = json.loads((APP_ROOT / "data" / "fire_boxes.json").read_text(encoding="utf-8"))
    videos = []
    for xml, mp4, split in list_videos(WORKSPACE_ROOT / "방화", "방화")[0]:
        if split != "train":
            continue
        events = parse_annotation(xml, "fire")
        label = labels[Path(mp4).name]
        x0, y0 = label["crop"]
        frames = [(int(f), [cell_box(v, x0, y0)] if v else []) for f, v in label["frames"].items() if v != "?"]
        fps = cv2.VideoCapture(str(mp4)).get(cv2.CAP_PROP_FPS) or 3.0
        last_clear = events[0]["start_frame"] - int(fps)  # 사건 시작 1초 전까지는 불 없음
        labeled = {f for f, _ in frames}
        negatives = [round(last_clear * k / NEG_PER_VIDEO) for k in range(NEG_PER_VIDEO)] if last_clear > 0 else []
        frames += [(f, []) for f in negatives if f not in labeled]
        videos.append({"video": mp4, "events": events, "frames": frames})
    return videos


def mine_negatives(model, videos, cache_dir, seed):
    """[{'video', 'frames': [(프레임, [])]}]: 착각 프레임(점수 높은 순) + 무작위 프레임. 불이 없는 영상만 넣는다."""
    rng = random.Random(seed)
    items = []
    for video in videos:
        fps, step, found = load_fire_detections(video, model, cache_dir, imgsz=FIRE_IMGSZ)
        scored = sorted(((max((b[4] for b in boxes), default=0.0), f) for f, boxes in found.items()), reverse=True)
        hard = [f for score, f in scored[:HARD_PER_VIDEO] if score >= FIRE_CONF]
        easy = rng.sample([f for f in found if f not in hard], min(EASY_PER_VIDEO, max(0, len(found) - len(hard))))
        items.append({"video": video, "frames": [(f, []) for f in hard + easy], "hard": len(hard)})
    return items


def write_dataset(videos, folder):
    """YOLO 이미지·라벨을 쓰고 이미지 경로 목록을 돌려준다."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    paths = []
    for item in videos:
        cap = cv2.VideoCapture(str(item["video"]))
        for frame_no, boxes in item["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            name = f"{Path(item['video']).stem}_{frame_no:05d}"
            cv2.imencode(".jpg", frame)[1].tofile(str(folder / "images" / f"{name}.jpg"))
            (folder / "labels" / f"{name}.txt").write_text("\n".join(
                f"{FIRE_CLASS_ID} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                for x1, y1, x2, y2 in boxes), encoding="utf-8")
            paths.append(str(folder / "images" / f"{name}.jpg"))
        cap.release()
    return paths


def frame_accuracy(model, videos):
    """표기 프레임 적중: 불 프레임은 표기 박스 안에 중심이 있는 검출, 불 없음 프레임은 검출 없음."""
    fire_hit = fire_total = clear_ok = clear_total = 0
    for item in videos:
        cap = cv2.VideoCapture(str(item["video"]))
        for frame_no, boxes in item["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue
            result = model.predict(frame, conf=FIRE_CONF, imgsz=FIRE_IMGSZ, verbose=False)[0]
            found = [b for b, c in zip(result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist())
                     if result.names[c] in FIRE_CLASSES]
            if boxes:
                fire_total += 1
                fire_hit += any(x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2
                                for b in found for x1, y1, x2, y2 in boxes)
            else:
                clear_total += 1
                clear_ok += not found
        cap.release()
    return {"fire_frame_recall": round(fire_hit / max(1, fire_total), 2), "fire_frames": fire_total,
            "clear_frame_specificity": round(clear_ok / max(1, clear_total), 2), "clear_frames": clear_total}


def video_scores(model, videos, cache_dir):
    hit = total = false = 0
    minutes = 0.0
    for video, events in videos:
        fps, step, found = load_fire_detections(video, model, cache_dir, imgsz=FIRE_IMGSZ)
        h, t, fa = score_alarms(fire_alarms(found, fps, step, FIRE_CONF), events, fps)
        hit, total, false = hit + h, total + t, false + fa
        minutes += min((e["start_frame"] for e in events), default=max(found, default=0) + step) / fps / 60
    return {"events": total, "hits": hit, "recall": round(hit / total, 2) if total else None, "false_alarms": false,
            "minutes": round(minutes, 1), "fp_per_min": round(false / minutes, 3) if minutes else None}


def train_fold(train_videos, folder, epochs, extra_paths=()):
    """extra_paths: 이미 YOLO 이미지·라벨로 써 둔 추가 학습 이미지(D-Fire·합성, train_fire_mix)."""
    from ultralytics import YOLO

    paths = write_dataset(train_videos, folder / "data") + list(extra_paths)
    (folder / "train.txt").write_text("\n".join(paths), encoding="utf-8")
    (folder / "data.yaml").write_text(
        f"train: {folder / 'train.txt'}\nval: {folder / 'train.txt'}\nnames:\n  0: smoke\n  1: fire\n", encoding="utf-8")
    YOLO(str(APP_ROOT / FIRE_BASE_WEIGHTS)).train(
        data=str(folder / "data.yaml"), epochs=epochs, imgsz=FIRE_IMGSZ, batch=8, val=False, plots=False, seed=0,
        workers=2, project=str(folder), name="run", exist_ok=True, verbose=False)
    return YOLO(str(folder / "run" / "weights" / "last.pt"))


def evaluate(model, name, train, dev, dev_others, folder):
    return {"model": name, "train_frames": frame_accuracy(model, train), "dev_frames": frame_accuracy(model, dev),
            "dev_fire_videos": video_scores(model, [(v["video"], v["events"]) for v in dev], folder / "det"),
            "dev_other_videos": video_scores(model, [(v, []) for v in dev_others], folder / "det")}


def _total(results, key):
    rows = [r[key] for r in results]
    events, minutes = sum(r["events"] for r in rows), sum(r["minutes"] for r in rows)
    return {"events": events, "recall": round(sum(r["hits"] for r in rows) / events, 2) if events else None,
            "false_alarms": sum(r["false_alarms"] for r in rows), "minutes": round(minutes, 1),
            "fp_per_min": round(sum(r["false_alarms"] for r in rows) / minutes, 3) if minutes else None}


def main():
    parser = argparse.ArgumentParser(description="불 검출기 파인튜닝 + 어려운 음성 (공통 분할 5-fold)")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "fire_cv2")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    splits = load_splits()
    fold_of = lambda video: splits[Path(video).name]["fold"]  # 분할에 없는 영상은 KeyError로 멈춘다
    videos = load_samples()
    others = [mp4 for name in ("절도", "파손", "유기")
              for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]
    others += sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))

    summary = {"stage1": [], "stage2": []}
    for k in range(FOLDS):
        folder = args.output_dir / f"fold{k}"
        train = [v for v in videos if fold_of(v["video"]) != k]
        dev = [v for v in videos if fold_of(v["video"]) == k]
        train_others = [v for v in others if fold_of(v) != k]
        dev_others = [v for v in others if fold_of(v) == k]
        groups = lambda items: {splits[Path(x).name]["group"] for x in items}
        assert not groups([v["video"] for v in train] + train_others) & groups([v["video"] for v in dev] + dev_others), "누수"

        stage1 = train_fold(train, folder / "stage1", args.epochs)
        summary["stage1"].append({"fold": k, **evaluate(stage1, "stage1", train, dev, dev_others, folder / "stage1")})
        negatives = mine_negatives(stage1, train_others, folder / "stage1" / "mine", seed=k)
        stage2 = train_fold(train + negatives, folder / "stage2", args.epochs)
        summary["stage2"].append({"fold": k, "hard_negatives": sum(n["hard"] for n in negatives),
                                  "negative_frames": sum(len(n["frames"]) for n in negatives),
                                  **evaluate(stage2, "stage2", train, dev, dev_others, folder / "stage2")})
        print(json.dumps({"fold": k, "stage1": summary["stage1"][-1], "stage2": summary["stage2"][-1]}, ensure_ascii=False))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["total"] = {stage: {"fire": _total(summary[stage], "dev_fire_videos"),
                                "other": _total(summary[stage], "dev_other_videos")} for stage in ("stage1", "stage2")}
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["total"], ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
