"""관절 시퀀스 헤드(YOLO11s-pose 관절 -> 작은 GRU, 4클래스 softmax)를 날짜 5-fold로 학습해 OOF 로짓을 낸다. S3D와 후기 결합으로 비교하기 위한 스트림 P.

- 입력: output/pose_cache/<행동>/train/<영상>.npz (hand_pilot_pose가 만든 프레임별 사람 관절). 창(12표본)마다 프레임별로 박스가 가장 큰 사람 한 명의 관절을 쓴다.
- 프레임 특징 53차원: 관절 17개 x (박스 중심 기준 x, y를 박스 높이로 나눈 값, 신뢰도) + 박스 가로/세로 비 + 사람 있음 표시. 절대 위치·화면 크기는 넣지 않는다(카메라가 달라도 같은 값).
- 창 목록·라벨·fold는 finetune_s3d.build_index와 같다(반품 영상은 쓰지 않는다: 사용자 지시). label -1(애매)은 학습에서 뺀다. 증강(좌우 반전)은 학습에만 적용한다.
- 출력: output/pose_head/oof_fold{k}.npz (finetune_s3d와 같은 형식), 최종 모델 fold -1은 --final.
사용: python -m pipelines.pose_head [--fold k] [--final]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch import nn

from pipelines import finetune_s3d as F

POSE = F.APP_ROOT / "output" / "pose_cache"
OUT = F.APP_ROOT / "output" / "pose_head"
WIN, DIM, EPOCHS, BATCH, LR = F.WIN, 53, 20, 256, 1e-3
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # COCO 좌우 관절 쌍


def frame_feature(kpt: np.ndarray, box: np.ndarray) -> np.ndarray:
    w, h = max(float(box[2] - box[0]), 1.0), max(float(box[3] - box[1]), 1.0)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    xy = np.stack([(kpt[:, 0] - cx) / h, (kpt[:, 1] - cy) / h, kpt[:, 2]], axis=1)
    return np.concatenate([xy.reshape(-1), [w / h, 1.0]]).astype(np.float32)


def clip_frames(path) -> dict[int, np.ndarray]:
    """프레임 번호 -> 박스가 가장 큰 사람의 프레임 특징."""
    z = np.load(path)
    best: dict[int, tuple[float, np.ndarray]] = {}
    for fr, box, kpt in zip(z["frame"], z["box"], z["kpt"]):
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        if int(fr) not in best or area > best[int(fr)][0]:
            best[int(fr)] = (area, frame_feature(kpt, box))
    return {f: v[1] for f, v in best.items()}


def build_windows(rows: list[dict]) -> np.ndarray:
    cache: dict[str, dict[int, np.ndarray]] = {}
    x = np.zeros((len(rows), WIN, DIM), np.float32)
    for i, r in enumerate(rows):
        if r["clip"] not in cache:
            cache[r["clip"]] = clip_frames(POSE / r["behavior"] / "train" / f"{r['clip']}.npz")
        fr = cache[r["clip"]]
        for k in range(WIN):
            if r["start"] + k in fr:
                x[i, k] = fr[r["start"] + k]
    return x


def flip(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    j = torch.tensor(FLIP)
    y[..., :51] = x[..., :51].reshape(*x.shape[:-1], 17, 3)[..., j, :].reshape(*x.shape[:-1], 51)
    y[..., 0:51:3] = -y[..., 0:51:3]
    return y


class PoseGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(DIM, 128), nn.ReLU())
        self.gru = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.inp(x))
        return self.out(torch.cat([h.mean(1), h.max(1).values], dim=1))


def fit(x: np.ndarray, y: np.ndarray, device: str, seed: int = 0) -> PoseGRU:
    torch.manual_seed(seed)
    model = PoseGRU().to(device)
    freq = np.bincount(y, minlength=4).astype(np.float64)
    weight = torch.tensor((freq.sum() / (4 * np.maximum(freq, 1))) ** 0.5, dtype=torch.float32, device=device)   # 역빈도의 제곱근
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(xt), device=device).split(BATCH):
            xb = xt[idx]
            xb = torch.where(torch.rand(len(xb), 1, 1, device=device) < 0.5, flip(xb), xb)   # 좌우 반전 증강(학습만)
            loss = nn.functional.cross_entropy(model(xb), yt[idx], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def score(model: PoseGRU, x: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        return torch.cat([model(torch.from_numpy(b).to(device)).float().cpu() for b in np.array_split(x, max(1, len(x) // 2048))]).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [r for r in F.build_index() if r["behavior"] != "반품" and (POSE / r["behavior"] / "train" / f"{r['clip']}.npz").is_file()]
    x = build_windows(rows)
    y = np.array([r["cls"] for r in rows]); ok = np.array([r["label"] != -1 for r in rows]); fold = np.array([r["fold"] for r in rows])
    print(f"창 {len(rows)}개(학습 사용 {int(ok.sum())}), 영상 {len({r['clip'] for r in rows})}개", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    folds = [-1] if args.final else ([args.fold] if args.fold is not None else sorted(set(fold.tolist())))
    for k in folds:
        tr = ok & (fold != k)
        model = fit(x[tr], y[tr], device)
        if k == -1:
            torch.save({"model": model.state_dict()}, OUT / "pose_fold-1.pt"); continue
        te = np.flatnonzero(fold == k)
        np.savez_compressed(OUT / f"oof_fold{k}.npz", clip=np.array([rows[i]["clip"] for i in te]), start=np.array([rows[i]["start"] for i in te]),
                            label=np.array([rows[i]["label"] for i in te]), cls=y[te], behavior=np.array([rows[i]["behavior"] for i in te]), logits=score(model, x[te], device))
        print(f"fold {k} 보류 창 {len(te)}개 채점 완료", flush=True)


def _self_check() -> None:
    kpt = np.zeros((17, 3), np.float32); kpt[:, 2] = 1.0; kpt[1] = [10.0, 0.0, 1.0]
    f = frame_feature(kpt, np.array([-20.0, -50.0, 20.0, 50.0], np.float32))
    assert f.shape == (DIM,) and abs(f[3] - 0.1) < 1e-6 and abs(f[-2] - 0.4) < 1e-6 and f[-1] == 1.0   # 박스 높이 100으로 나눈 상대 좌표, 가로/세로 비
    v = np.zeros((1, 1, DIM), np.float32); v[0, 0, 15] = 0.2; v[0, 0, 17] = 0.9                         # 관절 5(왼어깨)의 x와 신뢰도
    g = flip(torch.from_numpy(v))
    assert abs(g[0, 0, 18] + 0.2) < 1e-6 and g[0, 0, 15] == 0 and abs(g[0, 0, 20] - 0.9) < 1e-6          # 관절 6(오른어깨)으로 옮겨지고 x 부호 반전
    assert PoseGRU()(torch.zeros(2, WIN, DIM)).shape == (2, 4)
    print("ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _self_check() if "--self-check" in sys.argv else main()
