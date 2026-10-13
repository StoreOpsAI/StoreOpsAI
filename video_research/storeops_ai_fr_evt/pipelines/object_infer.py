"""작은 물건 탐지기 교차 적합 추론: 영상이 속한 날짜 fold를 학습하지 않은 모델로 3fps 표본 프레임을 탐지해 저장한다.

fold {0,1,2} 영상 -> 모델 B(fold {3,4}로 학습), fold {3,4} 영상 -> 모델 A(fold {0,1,2}로 학습). object_dataset2.GROUPS와 같은 분할.
원본 해상도 프레임을 OBJECT_FT_IMG_SIDE(학습과 같은 1280)로 추론한다. 후보를 넉넉히 남기려고 conf 하한을 낮게(INFER_CONF) 둔다(최종 기준은 OOF에서 정한다).
출력: C:/data/features/objdet/<영상>.npz = frame(표본 번호), box(x1,y1,x2,y2 원본 좌표), conf, n_frames
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT, OBJECT_FT_IMG_SIDE
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.fullscene_windows import video_path
from pipelines.global_split import SPLIT_PATH
from pipelines.object_dataset2 import GROUPS
from pipelines.video_sampling import is_sample

SUFFIX = os.getenv("OBJDET_SUFFIX", "")  # "30"이면 30에폭 재학습 모델(cross{A,B}30)로 추론해 objdet30에 저장(기존 결과 보존)
OUT = Path(BEHAVIOR_FEATURE_ROOT) / f"objdet{SUFFIX}"
INFER_CONF = 0.10  # TEMPORARY_DEFAULT: 후보 수집용 하한(최종 경보 기준 아님). 관계 특징에서 점수를 연속값으로 쓴다
MAX_PER_FRAME = 30
BATCH = 8
PRIORITY = {"유기": 0, "절도": 0, "반품": 1, "매장이동": 1, "전도": 2}


def model_for(fold: int) -> str:
    """그 fold를 학습하지 않은 모델 이름(A/B)."""
    return "B" if fold in GROUPS["A"] else "A"


def read_samples(video: Path) -> list[np.ndarray]:
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
    return frames


def detect(model, frames: list[np.ndarray]) -> dict[str, np.ndarray]:
    frame_idx, boxes, confs = [], [], []
    for s in range(0, len(frames), BATCH):
        for k, r in enumerate(model.predict(frames[s:s + BATCH], imgsz=OBJECT_FT_IMG_SIDE, conf=INFER_CONF, half=True, verbose=False)):
            b, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            order = np.argsort(-c)[:MAX_PER_FRAME]
            frame_idx.extend([s + k] * len(order))
            boxes.extend(b[order])
            confs.extend(c[order])
    return {"frame": np.array(frame_idx, np.int32), "box": np.array(boxes, np.float32).reshape(-1, 4), "conf": np.array(confs, np.float32), "n_frames": np.int32(len(frames))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--with-validation", action="store_true")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    from ultralytics import YOLO

    models: dict[str, "YOLO"] = {}
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))
            if (r["official_split"] == "train" and r["q_fall"] != "True") or (args.with_validation and r["official_split"] == "validation")]
    OUT.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(sorted(rows, key=lambda r: (PRIORITY[r["behavior"]], r["clip"]))[i::n], 1):
        target = OUT / f"{r['clip']}.npz"
        if target.is_file():
            continue
        name = model_for(int(r["fold"]))
        if name not in models:
            models[name] = YOLO(str(APP_ROOT / "output" / "object_ft" / f"cross{name}{SUFFIX}" / "weights" / "best.pt"))
        video = next((p for p in (video_path(r["clip"], r["behavior"], s) for s in (r["official_split"], "train", "validation")) if p.is_file()), None)
        if video is not None:
            tmp = target.with_name(target.stem + ".tmp.npz")
            np.savez_compressed(tmp, **detect(models[name], read_samples(video)))
            tmp.replace(target)
        if k % 50 == 0:
            print(f"[{k}] {r['clip']}", flush=True)
    print("OBJECT_INFER_DONE", flush=True)


def _self_check() -> None:
    """가짜 모델로 detect의 인덱스·정렬·상한을 확인한다."""
    import types

    class Boxes:
        def __init__(self, n):
            self.xyxy = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: np.arange(n * 4, dtype=np.float32).reshape(n, 4)))
            self.conf = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: np.linspace(0.2, 0.9, n, dtype=np.float32)))

    class Fake:
        def predict(self, frames, **kw):
            return [types.SimpleNamespace(boxes=Boxes(0 if i == 1 else 40)) for i, _ in enumerate(frames)]

    out = detect(Fake(), [np.zeros((4, 4, 3), np.uint8)] * 3)
    assert out["n_frames"] == 3 and set(out["frame"].tolist()) == {0, 2}                        # 물건이 없는 프레임은 빠진다
    assert (out["frame"] == 0).sum() == MAX_PER_FRAME and out["conf"][0] == np.float32(0.9)  # 점수 높은 순, 상한 적용
    assert model_for(0) == "B" and model_for(2) == "B" and model_for(3) == "A" and model_for(4) == "A"
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
