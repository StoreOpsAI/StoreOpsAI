"""창 특징 헤드 OOF 비교(스펙 01 §15, 02 §3·§19): 구조형(포즈 시퀀스) / S3D만 / X-CLIP만 / 결합을 같은 날짜 fold에서 잰다.

- 데이터: 전 행동 공통 분할표(data/global_split.csv)의 학습 가능 클립만(헤드의 봉인 검증·격리 제외). fold = 촬영 날짜 그룹.
- 라벨: 대상 행동 클립의 창 라벨 1/0(전환 구간 -1은 제외). 다른 행동·정상 풀의 창은 전부 0(그 행동이 아님). 봉인 검증은 읽지 않는다.
- 분류기는 표준화 + 클래스 균형 로지스틱(설정 고정). 출력은 OOF 창 점수.
- 지표: AP, 그리고 오경보 예산별 재현율(임계값은 정상 창 점수 분포에서 정한다: 정상 창 시간당 오경보가 예산 이하가 되는 가장 낮은 점수).
  정상 창 시간 = 창 수 x 창 간격(ABANDON_WIN_STRIDE/3fps초). 사람 창 단위라 영상 시간당이 아니라 "사람 창 시간당"이다.
- 정상 풀 중 점원 그룹(촬영 날짜 10-16, 반품)의 오경보율을 따로 낸다.
출력: output/window_head/<head>_oof.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config.config import ABANDON_WIN_FRAMES, ABANDON_WIN_STRIDE, BEHAVIOR_FEATURE_ROOT, PERCEPTION_FPS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.global_split import SPLIT_PATH

HEAD_BEHAVIOR = {"fall": "전도", "abandon": "유기", "theft": "절도", "fire": "방화"}
FEATURE_DIRS = ("fall_train", "theft_train", "fire_train", "normal_train", "abandon_train", "abandon_v1")
BUDGETS_PER_HOUR = (0.42, 1.0, 2.0, 5.0)  # 카메라당 하루 10건(24시간) ≈ 0.42/시간을 기준으로 한 민감도 값
WINDOW_HOURS = ABANDON_WIN_STRIDE / PERCEPTION_FPS / 3600


def load(head: str) -> dict[str, np.ndarray]:
    rows = {r["clip"]: r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))}
    target, parts = HEAD_BEHAVIOR[head], {k: [] for k in ("s3d", "xclip", "pose", "y", "fold", "group", "behavior")}
    seen = set()
    for d in FEATURE_DIRS:
        for f in sorted((Path(BEHAVIOR_FEATURE_ROOT) / d).glob("*.npz")):
            r = rows.get(f.stem)
            if f.name.endswith(".tmp.npz") or r is None or f.stem in seen or r["official_split"] != "train" or r[f"q_{head}"] == "True":
                continue
            z = np.load(f)
            if int(z["n"]) == 0:
                continue
            seen.add(f.stem)
            label = z["label"].astype(int) if r["behavior"] == target else np.zeros(int(z["n"]), int)
            keep = label >= 0  # 대상 행동의 전환 구간(-1)만 뺀다
            for k in ("s3d", "xclip"):
                parts[k].append(z[k][keep])
            parts["pose"].append(z["pose"].reshape(len(z["pose"]), -1)[keep])
            n = int(keep.sum())
            parts["y"].append(label[keep]), parts["fold"].append(np.full(n, int(r["fold"])))
            parts["group"].append(np.array([f"{r['site']}_{r['date']}"] * n)), parts["behavior"].append(np.array([r["behavior"]] * n))
    return {k: np.concatenate(v) for k, v in parts.items()}


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    hit = y[order]
    return float((np.cumsum(hit) / np.arange(1, len(hit) + 1))[hit == 1].mean())


def recall_at_budgets(y: np.ndarray, s: np.ndarray) -> dict[str, dict]:
    neg, pos = np.sort(s[y == 0])[::-1], s[y == 1]
    hours = len(neg) * WINDOW_HOURS
    out = {}
    for b in BUDGETS_PER_HOUR:
        allowed = int(b * hours)
        thr = neg[allowed] + 1e-9 if allowed < len(neg) else -1.0  # 오경보 허용 수를 넘지 않는 가장 낮은 점수
        out[str(b)] = {"thr": float(thr), "recall": float((pos > thr).mean()), "false_windows": int((neg > thr).sum())}
    return out


def fit_gru(x: np.ndarray, y: np.ndarray, epochs: int = 30, hidden: int = 64):
    """포즈 시퀀스(창당 ABANDON_WIN_FRAMES x 17 x 3)를 시간 순서대로 GRU에 넣는 헤드. 표준화 + 클래스 균형, 시드 고정.
    ponytail: 에폭·은닉 크기는 TEMPORARY_DEFAULT(OOF 탐색 대상). 선형 기준선과 같은 fold에서 비교해 이득이 있을 때만 키운다."""
    seq = x.reshape(len(x), ABANDON_WIN_FRAMES, -1)
    mean, std = seq.mean((0, 1)), seq.std((0, 1)) + 1e-6
    xt, yt = torch.tensor((seq - mean) / std, dtype=torch.float32), torch.tensor(y)
    torch.manual_seed(0)
    gru, head = nn.GRU(seq.shape[2], hidden, batch_first=True), nn.Linear(hidden, 2)
    params = list(gru.parameters()) + list(head.parameters())
    counts = torch.bincount(yt, minlength=2).float()
    criterion, optimizer = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts)), torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 512):
            b = perm[i:i + 512]
            optimizer.zero_grad()
            criterion(head(gru(xt[b])[1][-1]), yt[b]).backward()
            optimizer.step()

    @torch.no_grad()
    def predict(values: np.ndarray) -> np.ndarray:
        v = torch.tensor((values.reshape(len(values), ABANDON_WIN_FRAMES, -1) - mean) / std, dtype=torch.float32)
        return head(gru(v)[1][-1]).softmax(1)[:, 1].numpy()

    return predict


def oof_scores(x: np.ndarray, y: np.ndarray, fold: np.ndarray, fit=_fit_linear) -> np.ndarray:
    s = np.zeros(len(y))
    for k in sorted(set(fold)):
        tr, te = fold != k, fold == k
        s[te] = fit(x[tr], y[tr])(x[te])
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", default="fall", choices=sorted(HEAD_BEHAVIOR))
    args = parser.parse_args()
    d = load(args.head)
    y, fold = d["y"].astype(np.int64), d["fold"]
    sets = {"pose(구조형)": d["pose"], "pose-GRU(시계열)": d["pose"], "s3d": d["s3d"], "xclip": d["xclip"], "s3d+pose(결합)": np.hstack([d["s3d"], d["pose"]]),
            "s3d+xclip+pose": np.hstack([d["s3d"], d["xclip"], d["pose"]])}
    info = {"head": args.head, "windows": int(len(y)), "positive": int(y.sum()), "by_behavior": dict(Counter(d["behavior"])), "folds": dict(sorted(Counter(fold.tolist()).items())),
            "window_hours_per_window": WINDOW_HOURS, "budgets_per_hour": list(BUDGETS_PER_HOUR)}
    result = {}
    for name, x in sets.items():
        s = oof_scores(x.astype(np.float32), y, fold, fit_gru if "GRU" in name else _fit_linear)
        r = {"ap": average_precision(y, s), "at_budget": recall_at_budgets(y, s)}
        clerk = (d["behavior"] == "반품") & np.char.endswith(d["group"], "10-16")  # 점원 장면이 몰린 그룹
        if clerk.any():
            r["clerk_group_false_rate_at_budget"] = {b: float((s[clerk] > v["thr"]).mean()) for b, v in r["at_budget"].items()}
        result[name] = r
        print(f"{name:16s} AP {r['ap']:.3f} | " + " ".join(f"{b}/h:재현율 {v['recall']:.2f}" for b, v in r["at_budget"].items()), flush=True)
    out = APP_ROOT / "output" / "window_head"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.head}_oof.json").write_text(json.dumps({"info": info, "result": result}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
