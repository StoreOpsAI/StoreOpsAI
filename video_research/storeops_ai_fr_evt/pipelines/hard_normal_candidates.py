"""어려운 정상 후보 추출: 반품 train 영상(누운 사람 없다고 가정되는 정상 풀)에서 전도 MLP가 누움이라고 의심하는 지점을 모아 판정용 사진 시트로 만든다.

- 모델: 전도 first_pass + 매장이동 정상으로 학습한 관절 MLP(최종 설정). 반품 영상은 학습에 쓰지 않았다.
- 후보 기준은 낮게(SURFACE_THR 이상, 같은 자리 SURFACE_HOLD_SEC 이상) 잡아 놓치지 않게 모으고, 실제 경보 판정은 OOF 탐색으로 정한 운영점으로 따로 한다.
- 출력: output/hard_normal/candidates.json (후보 id·영상·프레임·박스·점수), output/hard_normal/sheets/sheet_NN.jpg (후보 id 표기).
판정(누움/앉기·숙이기/기타)은 사진을 보고 data/hard_normal_review.json에 기록한다(검수자 표기 필수).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config.config import AIHUB_ORGANIZED_ROOT
from models.fall_detector import lying_alarms
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_lying import fit_method
from pipelines.lying_pose_dataset import load_dataset, pose_features
from pipelines.video_sampling import native_index

POSE_DIR = APP_ROOT / "output" / "lying_pose_return"
OUT = APP_ROOT / "output" / "hard_normal"
SURFACE_THR = 0.5  # 후보 수집용(경보 기준 아님)
SURFACE_HOLD_SEC = 1.0
FPS = 3.0
TILE, COLS, ROWS, MARGIN = 420, 4, 3, 0.35


def clip_alarms(score, data):
    """(프레임, 박스, 점수) 목록: 점수 >= SURFACE_THR인 사람만 fall_detector에 넣어 같은 자리 유지 경보를 낸다."""
    by_frame, scores = defaultdict(list), {}
    for f, box, s in zip(data["frame"], data["box"], score):
        if s >= SURFACE_THR:
            by_frame[int(f)].append(tuple(map(float, box)))
            scores[(int(f), tuple(map(float, box)))] = float(s)
    n = int(data["n_frames"])
    alarms = lying_alarms([(f, by_frame.get(f, [])) for f in range(n)], FPS, hold_sec=SURFACE_HOLD_SEC)
    return [(f, box, scores[(f, box)]) for f, box in alarms]


def crop(video, frame, box):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, native_index(frame, cap.get(cv2.CAP_PROP_FPS)))  # 표본 번호 -> 원본 프레임 위치
    ok, image = cap.read()
    cap.release()
    if not ok:
        return np.zeros((TILE, TILE, 3), np.uint8)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(x1 - MARGIN * w, 0)), int(min(x2 + MARGIN * w, image.shape[1]))
    y1, y2 = int(max(y1 - MARGIN * h, 0)), int(min(y2 + MARGIN * h, image.shape[0]))
    patch = image[y1:y2, x1:x2]
    scale = TILE / max(patch.shape[:2])
    patch = cv2.resize(patch, None, fx=scale, fy=scale)
    tile = np.zeros((TILE, TILE, 3), np.uint8)
    tile[:patch.shape[0], :patch.shape[1]] = patch
    return tile


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(exist_ok=True)
    score = fit_method(load_dataset(include_normal=True, include_aug=False), "mlp")
    video_dir = None  # 반품 train/validation 어느 쪽이든 찾는다
    cands = []
    for npz in sorted(POSE_DIR.glob("*.npz")):
        data = np.load(npz)
        feats = np.array([pose_features(b, k) for b, k in zip(data["box"], data["kpt"])]).reshape(len(data["box"]), -1)
        s = score({"feat": feats})
        for f, box, sc in clip_alarms(s, data):
            cands.append({"id": f"R{len(cands):04d}", "clip": npz.stem, "frame": f, "box": list(box), "score": round(sc, 4)})
    (OUT / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    per = COLS * ROWS
    for i in range(0, len(cands), per):
        tiles = []
        for c in cands[i:i + per]:
            t = crop(next(Path(AIHUB_ORGANIZED_ROOT).glob(f"반품/*/videos/{c['clip']}.mp4")), c["frame"], c["box"])
            cv2.putText(t, f"{c['id']} s={c['score']:.2f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(t)
        tiles += [np.zeros((TILE, TILE, 3), np.uint8)] * (per - len(tiles))
        sheet = np.vstack([np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
        ok, buf = cv2.imencode(".jpg", cv2.resize(sheet, None, fx=0.6, fy=0.6))  # imwrite는 한글 경로에 저장하지 못한다
        assert ok
        (OUT / "sheets" / f"sheet_{i // per:02d}.jpg").write_bytes(buf.tobytes())
    print(f"후보 {len(cands)}개, 시트 {-(-len(cands) // per)}장 -> {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
