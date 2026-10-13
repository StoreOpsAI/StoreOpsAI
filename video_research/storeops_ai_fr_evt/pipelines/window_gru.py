"""연속 창 GRU(긴 맥락): 창마다 [A안 로짓 4 + C1 로짓 4]를 과거 K개 창(인과적, 미래 창 사용 안 함)으로 이어 읽어 4클래스를 판정한다.

- 입력은 이미 있는 OOF 점수(S3D를 다시 돌리지 않는다). 날짜 fold CV: 다른 fold의 OOF 점수로 학습, 보류 fold로 시험. 같은 영상의 앞쪽 창이 부족하면 첫 창을 반복해 채운다.
- 소형 GRU(은닉 HIDDEN, 에폭 EPOCHS, 시드 고정)는 TEMPORARY_DEFAULT. K=1은 한 창만 보는 기준선(선형 결합과 같은 정보).
- 평가는 eval_finetune과 같은 이벤트 단위(재현율, 시간당 오경보 구간). 점수 = 그 행동 로짓 - logsumexp(나머지 로짓).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.eval_finetune import CLASSES, evaluate
from pipelines.ensemble_eval import merged
from pipelines.fusion_eval import clips_for
from pipelines.oof_search import candidates_from_values

HIDDEN, EPOCHS, BATCH, LR = 32, 20, 512, 1e-2
KS = tuple(int(x) for x in __import__("os").environ.get("GRU_KS", "1,3,6").split(","))
RECALL_TARGETS = (0.90, 0.95)


def person_channels(d: dict[str, np.ndarray]) -> np.ndarray:
    """창마다 사람 수 변화 4채널(사람 검출·추적에서만 계산, 물건 탐지기 불필요): 평균 사람 수, 마지막 사람 수, 마지막-처음, 사라짐 여부(창 안 최대).
    '사람이 없어지면 유기'라는 조건을 학습이 쓸 수 있게 하는 채널. relwin의 n_persons(0번)·person_gone(11번) 채널을 쓴다."""
    out, cache = np.zeros((len(d["clip"]), 4), np.float32), {}
    for i, (clip, s) in enumerate(zip(d["clip"], d["start"])):
        if clip not in cache:
            z = np.load(Path(BEHAVIOR_FEATURE_ROOT) / "relwin" / f"{clip}.npz")
            cache[clip] = dict(zip(z["start"].tolist(), z["seq"]))
        w = cache[clip][int(s)]
        out[i] = [w[:, 0].mean(), w[-1, 0], w[-1, 0] - w[0, 0], w[:, 11].max()]
    return out


def sequences(d: dict[str, np.ndarray], k: int, person: bool = False) -> np.ndarray:
    """창마다 (K, 8 또는 12): 같은 영상의 이전 창들(오래된 것부터), 부족하면 첫 창 반복."""
    arms = [d["la"]] if os.environ.get("GRU_A_ONLY", "0") == "1" else [d["la"], d["lc"]]   # 서비스에는 A 모델만 있어 A 로짓만 쓰는 변형
    x = np.hstack(arms + ([person_channels(d)] if person else [])).astype(np.float32)
    out = np.zeros((len(x), k, x.shape[1]), np.float32)
    for clip in np.unique(d["clip"]):
        idx = np.flatnonzero(d["clip"] == clip)
        idx = idx[np.argsort(d["start"][idx])]
        for pos, i in enumerate(idx):
            take = idx[max(0, pos - k + 1):pos + 1]
            out[i] = x[np.concatenate([np.full(k - len(take), take[0]), take])]
    return out


class Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gru, self.out = nn.GRU(dim, HIDDEN, batch_first=True), nn.Linear(HIDDEN, 4)

    def forward(self, x):
        return self.out(self.gru(x)[1][-1])


def fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    mean, std = x_tr.reshape(-1, x_tr.shape[2]).mean(0), x_tr.reshape(-1, x_tr.shape[2]).std(0) + 1e-6
    xt, yt = torch.tensor((x_tr - mean) / std), torch.tensor(y_tr)
    torch.manual_seed(0)
    model, counts = Head(x_tr.shape[2]), np.bincount(y_tr, minlength=4).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (4 * np.maximum(counts, 1))))  # 클래스 균형
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
    with torch.no_grad():
        return model(torch.tensor((x_te - mean) / std)).numpy()


def score(logits: np.ndarray, c: int) -> np.ndarray:
    rest = np.delete(logits, c, axis=1)
    m = rest.max(1)
    return logits[:, c] - (m + np.log(np.exp(rest - m[:, None]).sum(1)))


def main() -> None:
    d = merged()
    folds = sorted(set(d["fold"].tolist()))
    print(f"창 {len(d['clip'])}개, fold {folds}")
    valid = d["label"] != -1
    y = np.where(valid, d["cls"], 0).astype(np.int64)
    person = os.environ.get("GRU_PERSON", "0") == "1"
    print("입력:", ("A 로짓" if os.environ.get("GRU_A_ONLY", "0") == "1" else "A+C1 로짓") + (" + 사람 수 변화 4채널" if person else ""))
    results = {k: np.zeros((len(y), 4), np.float32) for k in KS}
    for k in KS:
        x = sequences(d, k, person)
        for f in folds:
            tr, te = valid & (d["fold"] != f), d["fold"] == f
            results[k][te] = fit_predict(x[tr], y[tr], x[te])
    for c, name in CLASSES.items():
        print(f"\n[{name}]")
        for k in KS:
            z = score(results[k], c)
            clips = clips_for(d, z, c)
            cand = candidates_from_values(z[valid], 120)
            row = []
            for target in RECALL_TARGETS:
                ok = [(t, r) for t, r in ((t, evaluate(clips, t)) for t in cand) if r["recall"] >= target]
                if ok:
                    t, r = min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"]))
                    row.append(f"재현율>={target}: {r['recall']:.3f} 오경보 {r['fa_per_hour']:.2f}/시간")
                else:
                    row.append(f"재현율>={target}: 만족 없음")
            print(f"  K={k} ({k * 2}초 간격 {k}창) | " + " | ".join(row))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
