"""손 크롭(H) 입력 캐시: 창(4초, 12표본, 간격 2초)마다 손목을 따라가는 한 변 400px 크롭을 224x224 JPEG로 저장한다(원본 해상도에서 자름).

- 크롭 규칙은 hand_pilot_probe.frame_centers 그대로: 손목(COCO 9·10, 신뢰도>=0.3)이 보이는 가장 큰 사람의 손목 중점, 두 손목이 멀면 넓힘, 중심 EMA 평활, 손목이 없으면 직전 중심 유지(처음이면 화면 중앙).
- 입력: output/pose_cache/<행동>/train/<영상>.npz(hand_pilot_pose), fullscene 창 시작 위치. 영상마다 프레임을 한 번만 자르고 창들이 공유한다.
- 출력: C:/data/features/crops_h/<영상>.pkl = {시작: [jpeg x12]} (crops_c1과 같은 형식). 이어받기 가능.
공식 validation은 --with-validation일 때만 처리한다(평가 입력 생성용, 튜닝에 쓰지 않음).
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.fullscene_windows import SIZE, WIN, video_path
from pipelines.global_split import SPLIT_PATH
from pipelines.hand_pilot_probe import POSE, frame_centers
from pipelines.video_sampling import is_sample

ROOT = Path(BEHAVIOR_FEATURE_ROOT)
QUALITY = 90


def build_clip(video: Path, kpts: dict[int, np.ndarray], starts: list[int]) -> dict[int, list[bytes]]:
    cap = cv2.VideoCapture(str(video))
    fps, raw, t = cap.get(cv2.CAP_PROP_FPS), 0, 0
    n_all = max(starts) + WIN
    centers, jpg = None, {}
    while t < n_all:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if not is_sample(raw - 1, fps):
            continue
        if centers is None:
            centers = frame_centers(kpts, n_all, frame.shape[:2])
        cx, cy, side = centers[t]
        x, y = int(cx - side / 2), int(cy - side / 2)
        crop = cv2.resize(frame[y:y + side, x:x + side], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        jpg[t] = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])[1].tobytes()
        t += 1
    cap.release()
    return {s: [jpg.get(s + k, jpg[max(jpg)]) for k in range(WIN)] for s in starts if jpg}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--with-validation", action="store_true")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    out = ROOT / "crops_h"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")) if r["behavior"] != "반품"   # 반품 영상은 쓰지 않는다(사용자 지시)
            and ((r["official_split"] == "train" and r["q_fall"] != "True") or (args.with_validation and r["official_split"] == "validation"))]
    for k, r in enumerate(sorted(rows, key=lambda r: r["clip"])[i::n], 1):
        target = out / f"{r['clip']}.pkl"
        split = "validation" if r["official_split"] == "validation" else "train"
        pose, ff = POSE / r["behavior"] / split / f"{r['clip']}.npz", ROOT / "fullscene" / f"{r['clip']}.npz"
        if target.is_file() or not pose.is_file() or not ff.is_file():
            continue
        video = video_path(r["clip"], r["behavior"], split if r["behavior"] != "반품" else ("train" if video_path(r["clip"], "반품", "train").is_file() else "validation"))
        if not video.is_file():
            continue
        pz = np.load(pose)
        kpts: dict[int, list] = {}
        for fr, kp in zip(pz["frame"], pz["kpt"]):
            kpts.setdefault(int(fr), []).append(kp)
        data = build_clip(video, {f: np.stack(v) for f, v in kpts.items()}, np.load(ff)["start"].tolist())
        if data:
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(data))
            tmp.replace(target)
        if k % 50 == 0:
            print(f"[{k}] {r['clip']}", flush=True)
    print("CROPS_H_DONE", flush=True)


def _self_check() -> None:
    kp = {0: np.zeros((1, 17, 3), np.float32)}
    kp[0][0, 9] = [800.0, 500.0, 0.9]
    kp[0][0, 10] = [900.0, 540.0, 0.9]
    c = frame_centers(kp, 3, (1080, 1920))
    assert len(c) == 3 and c[0][2] >= 400 and 700 < c[0][0] < 1000                  # 손목 중점 부근, 최소 400px
    assert abs(c[2][0] - c[1][0]) < 1e-6                                              # 손목이 없는 프레임은 직전 중심 유지
    print("ok")


if __name__ == "__main__":
    _self_check() if len(sys.argv) > 1 and sys.argv[1] == "--self-check" else main()
