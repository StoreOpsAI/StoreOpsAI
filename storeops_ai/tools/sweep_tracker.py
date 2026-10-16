"""한 사람만 나오는 영상에서 BoT-SORT 설정을 바꿔 가며 추적 번호가 몇 번 끊기는지 잰다.
번호가 N개면 끊김 N-1회(한 명뿐이라 라벨이 필요 없다). 사용: python tools/sweep_tracker.py 영상.mp4 [모델.pt] [conf]"""
import itertools
import sys
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[1] / "config" / "botsort_storeops.yaml"
GRID = {"proximity_thresh": [0.5, 0.7, 0.9], "match_thresh": [0.8, 0.9], "track_buffer": [30, 90, 150]}


def count_ids(model: YOLO, video: str, tracker: str, conf: float) -> tuple[int, int]:
    ids, frames = set(), 0
    for r in model.track(video, tracker=tracker, classes=[0], conf=conf, stream=True, verbose=False):
        frames += 1
        if r.boxes.id is not None:
            ids |= set(r.boxes.id.int().tolist())
    return len(ids), frames


def main(video: str, weights: str = "yolo11s-pose.pt", conf: float = 0.6) -> None:
    model, spec = YOLO(weights), yaml.safe_load(BASE.read_text(encoding="utf-8"))
    print("proximity match buffer | 번호 수 (끊김 = 번호 수 - 1)")
    for prox, match, buf in itertools.product(*GRID.values()):
        cfg = {**spec, "proximity_thresh": prox, "match_thresh": match, "track_buffer": buf}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        model.predictor = None   # 설정마다 추적기를 새로 만든다
        n, frames = count_ids(model, video, f.name, float(conf))
        print(f"{prox:>9} {match:>5} {buf:>6} | {n}  ({frames}프레임)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
