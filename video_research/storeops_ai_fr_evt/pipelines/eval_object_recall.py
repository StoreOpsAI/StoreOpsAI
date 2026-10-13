"""물건 탐지기가 유기 라벨 물건(object_abandon) 박스를 찾는지 재현율로 잰다 (방침 B 확정 전 측정).

train 첫 패스 영상만 쓴다(공식 validation 봉인). 라벨 박스는 물건이 처음 보이는 프레임의 것이다.
적중: 그 프레임에서 사람이 아닌 탐지 박스가 라벨 박스와 IoU >= OBJECT_RECALL_IOU.
크기별(화면 대비 넓이)·사건 내 순서별로 나눠 보고한다. 예: --detector coco | world
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from config.config import (OBJECT_RECALL_CONF, OBJECT_RECALL_IMGSZ, OBJECT_RECALL_IOU, OBJECT_VERIFY_CLASSES,
                           OBJECT_VERIFY_WEIGHTS, PERSON_CLASS_ID)
from models.new_object_detector import _iou
from pipelines.abandon_manifest import ABANDON_MANIFEST_PATH
from pipelines.aihub_abandonment import APP_ROOT

SIZE_BINS = ((0.0, 0.002, "작음<0.2%"), (0.002, 0.01, "중간0.2~1%"), (0.01, 1.0, "큼>=1%"))


def load_detector(name: str, weights: str = ""):
    if name == "ft":
        from ultralytics import YOLO
        return YOLO(weights), False
    if name == "coco":
        from ultralytics import YOLO
        return YOLO(str(APP_ROOT / "yolo11n.pt")), True
    if name == "world":
        from ultralytics import YOLOWorld
        model = YOLOWorld(str(APP_ROOT / OBJECT_VERIFY_WEIGHTS))
        model.set_classes(list(OBJECT_VERIFY_CLASSES))
        return model, False
    raise ValueError(f"알 수 없는 탐지기: {name}")


def detect_boxes(model, skip_person: bool, frame):
    result = model.predict(frame, conf=OBJECT_RECALL_CONF, imgsz=OBJECT_RECALL_IMGSZ, verbose=False)[0]
    keep = [i for i, c in enumerate(result.boxes.cls.int().tolist()) if not (skip_person and c == PERSON_CLASS_ID)]
    return [(result.boxes.xyxy[i].tolist(), float(result.boxes.conf[i])) for i in keep]


def read_frame(video: str, frame_no: int):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def size_bin(box, shape) -> str:
    ratio = (box[2] - box[0]) * (box[3] - box[1]) / (shape[0] * shape[1])
    return next(label for lo, hi, label in SIZE_BINS if lo <= ratio < hi)


def evaluate(clips, detector: str, weights: str = ""):
    model, skip_person = load_detector(detector, weights)
    rows = []
    for clip in clips:
        seen = defaultdict(int)
        for obj in clip["objects"]:
            frame = read_frame(clip["video"], obj["frame"])
            if frame is None:
                raise RuntimeError(f"프레임을 읽지 못함: {clip['clip']} {obj['frame']}")
            dets = detect_boxes(model, skip_person, frame)
            best = max((_iou(obj["box"], box) for box, _ in dets), default=0.0)
            event = next((i for i, e in enumerate(clip["events"]) if e["start"] <= obj["frame"] <= e["end"]), -1)
            order = seen[event]
            seen[event] += 1
            rows.append({"clip": clip["clip"], "session": clip["session"], "frame": obj["frame"], "order": order,
                         "size": size_bin(obj["box"], frame.shape), "best_iou": round(best, 3), "n_dets": len(dets),
                         "hit": best >= OBJECT_RECALL_IOU})
    return rows


def summarize(rows) -> dict:
    out = {"n": len(rows), "recall": sum(r["hit"] for r in rows) / max(len(rows), 1)}
    for key in ("size", "order"):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r[key])].append(r["hit"])
        out[key] = {k: {"n": len(v), "recall": round(sum(v) / len(v), 3)} for k, v in sorted(groups.items())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=("coco", "world", "ft"), required=True)
    parser.add_argument("--weights", default="", help="--detector ft일 때 파인튜닝 가중치")
    parser.add_argument("--site", default="", help="이 사이트(매장)의 영상 전체로 평가. 매장 단위 시험용")
    parser.add_argument("--fold", type=int, default=-1, help="이 fold의 영상 전체로 평가(첫 패스 대신). 시험 fold용")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개 영상만(스모크용)")
    args = parser.parse_args()
    manifest = json.loads(Path(ABANDON_MANIFEST_PATH).read_text(encoding="utf-8"))
    if args.site:
        clips = [c for c in manifest["train"] if c["site"] == args.site and not c["quarantine"]]
    elif args.fold >= 0:
        clips = [c for c in manifest["train"] if c["fold"] == args.fold and not c["quarantine"]]
    else:
        clips = [c for c in manifest["train"] if c["first_pass"] and not c["quarantine"]]
    clips = clips[:args.limit] if args.limit else clips
    rows = evaluate(clips, args.detector, args.weights)
    summary = summarize(rows)
    tag = f"{args.detector}_site{args.site}" if args.site else f"{args.detector}_fold{args.fold}" if args.fold >= 0 else args.detector
    out = Path("output") / f"object_recall_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"iou": OBJECT_RECALL_IOU, "summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
