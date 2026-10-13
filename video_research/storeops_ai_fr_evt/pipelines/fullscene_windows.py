"""서비스와 같은 입력 형식의 S3D 특징: 화면 전체, 4초 창(3fps 12프레임), 2초 간격, 224x224.

서비스 전처리(models/i3d_classifier._preprocess)와 같다: 프레임을 224x224로 리사이즈(INTER_LINEAR) -> 창의 12프레임에서 24프레임 균일 샘플링
-> BGR->RGB, /255, Kinetics 정규화. 여기서는 사전학습 S3D를 얼린 채 1024차원 특징만 저장한다(가능성 점검용, 미세조정 전 단계).

라벨(전도): 창 안에서 라벨 누운 구간(fall_start~끝)과 겹친 프레임이 FALL_MIN_LYING_SEC 이상이면 1, 겹침이 0이고 전환 구간(LYING_TRANSITION_FRAMES) 앞이면 0,
그 사이(막 넘어지는 중·애매한 구간)는 -1(학습·평가 제외, AGENTS.md 규칙 4). 다른 행동·정상 풀의 창은 전부 0.
분할은 전 행동 공통 분할표(data/global_split.csv)의 날짜 fold와 격리(q_fall)를 따른다. 공식 validation은 특징만 뽑고 학습·튜닝에 쓰지 않는다.
출력: C:/data/features/fullscene/<영상>.npz (s3d, start, label, n_frames)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import AIHUB_ORGANIZED_ROOT, BEHAVIOR_FEATURE_ROOT, LYING_TRANSITION_FRAMES, PERCEPTION_FPS
from pipelines.global_split import SPLIT_PATH
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.video_sampling import is_sample

OUT = Path(BEHAVIOR_FEATURE_ROOT) / "fullscene"
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)  # Kinetics-400 정규화(서비스 i3d_classifier와 같은 상수)
STD = np.array([0.22803, 0.22145, 0.216989], np.float32)
SIZE, WIN, STRIDE, NUM_FRAMES = 224, 12, 6, 24  # 서비스: I3D_WINDOW_SEC=4(3fps 12프레임), 2초 간격, 학습 num_frames=24
FALL_MIN_LYING_SEC = 3.0  # 사용자 지정: 누워 있는 상태가 3초 이상 이어진 창을 양성으로 본다(OOF 대상 아님, 정의)


def read_small(video: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    fps, raw, out = cap.get(cv2.CAP_PROP_FPS), 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if is_sample(raw - 1, fps):
            out.append(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
    cap.release()
    return np.stack(out) if out else np.zeros((0, SIZE, SIZE, 3), np.uint8)


def window_label(start: int, falls: list[dict]) -> int:
    if not falls:
        return 0
    lying = max(0, min(start + WIN, max(f["end"] for f in falls) + 1) - max(start, min(f["start"] for f in falls)))
    if lying >= FALL_MIN_LYING_SEC * PERCEPTION_FPS:
        return 1
    if lying == 0 and start + WIN <= min(f["start"] for f in falls) - LYING_TRANSITION_FRAMES:
        return 0
    return -1


def embed(model, frames: np.ndarray, starts: list[int], device: str) -> np.ndarray:
    idx = np.round(np.linspace(0, WIN - 1, NUM_FRAMES)).astype(int)
    out = []
    for i in range(0, len(starts), 8):
        batch = []
        for s in starts[i:i + 8]:
            clip = frames[s:s + WIN][idx][..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
            batch.append(((clip - MEAN) / STD).transpose(3, 0, 1, 2))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(device)
        with torch.inference_mode():
            out.append(model(x).mean(dim=(2, 3, 4)).float().cpu().numpy())
    return np.concatenate(out)


def video_path(clip: str, behavior: str, split: str) -> Path:
    return Path(AIHUB_ORGANIZED_ROOT) / behavior / split / "videos" / f"{clip}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1", help="i/n: 영상 목록을 n등분해 i번째만 처리(GPU 프로세스 분할)")
    parser.add_argument("--with-validation", action="store_true", help="공식 validation도 특징만 뽑는다(학습·튜닝에는 쓰지 않음)")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    from torchvision.models.video import S3D_Weights, s3d

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = s3d(weights=S3D_Weights.DEFAULT).features.to(device).eval()
    falls = {c["clip"]: c["falls"] for c in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["train"] + json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["validation"]}
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")) if (args.with_validation or r["official_split"] == "train") and r["q_fall"] != "True" or (args.with_validation and r["official_split"] == "validation")]
    rows = sorted(rows, key=lambda r: r["clip"])[i::n]
    OUT.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(rows, 1):
        target = OUT / f"{r['clip']}.npz"
        video = video_path(r["clip"], r["behavior"], r["official_split"] if r["behavior"] != "반품" else ("train" if video_path(r["clip"], "반품", "train").is_file() else "validation"))
        if target.is_file() or not video.is_file():
            continue
        frames = read_small(video)
        starts = list(range(0, len(frames) - WIN + 1, STRIDE))
        if not starts:
            continue
        labels = np.array([window_label(s, falls.get(r["clip"], []) if r["behavior"] == "전도" else []) for s in starts], np.int8)
        np.savez_compressed(target, s3d=embed(model, frames, starts, device), start=np.array(starts), label=labels, n_frames=len(frames))
        if k % 20 == 0:
            print(f"[{k}/{len(rows)}] {r['clip']}", flush=True)
    print("FULLSCENE_DONE", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
