"""A+C1 스택 결합 내보내기: OOF 로짓(A·C1 각 4개)으로 클래스별 표준화 + 클래스 균형 로지스틱을 학습해 서비스용 JSON으로 저장한다.

- 임계값: 날짜 fold CV로 만든 스택 점수(OOF)에서 재현율 >= TARGET이면서 오경보 창 비율이 최소인 값(ensemble_eval과 같은 기준). 임의 값 없음.
- 서비스 계수: 모든 유효 창으로 다시 학습한 스택(표준화 평균·표준편차, 점수 z = (W1-W0)·x_std + (b1-b0) = 로지스틱의 로그 오즈).
- 한계: 스택은 fold 모델들의 OOF 로짓으로 학습했고, 서비스에는 전체 데이터로 학습한 최종 A·C1이 들어간다(점수 분포가 더 극단일 수 있음). 공식 validation 한 번 평가 때 확인한다.
출력: <서비스>/training/runs/ours_stack/stack.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pipelines.compare_backbones import _fit_linear
from pipelines.ensemble_eval import merged
from pipelines.eval_finetune import CLASSES, evaluate
from pipelines.fusion_eval import clips_for
from pipelines.oof_search import candidates_from_values
from config.config import SERVICE_STACK_DIR

TARGET = 0.90
SERVICE_KEY = {1: "쓰러짐", 2: "쓰레기 투기", 3: "절도"}
OUT = SERVICE_STACK_DIR / "stack.json"


def fit_params(x: np.ndarray, y: np.ndarray, epochs: int = 300) -> dict:
    """_fit_linear과 같은 학습(같은 시드·설정)에서 계수를 꺼낸다."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.tensor((x - mean) / std, dtype=torch.float32), torch.tensor(y)
    torch.manual_seed(42)
    model = nn.Linear(x.shape[1], 2)
    counts = torch.bincount(yt, minlength=2).float()
    crit = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        crit(model(xt), yt).backward()
        opt.step()
    w, b = model.weight.detach().numpy().astype(np.float64), model.bias.detach().numpy().astype(np.float64)
    return {"mean": mean.astype(np.float64).tolist(), "std": std.astype(np.float64).tolist(), "coef": (w[1] - w[0]).tolist(), "bias": float(b[1] - b[0])}


def stack_z(params: dict, x: np.ndarray) -> np.ndarray:
    return ((x - np.array(params["mean"])) / np.array(params["std"])) @ np.array(params["coef"]) + params["bias"]


def fused_z(params: dict, x: np.ndarray) -> np.ndarray:
    """서비스와 같은 행동별 점수: 로지스틱 스택(계수가 있을 때) 또는 동적 결합(확신 큰 쪽에 비중). x = [A 로짓 4, C1 로짓 4]."""
    if "coef" in params:
        return stack_z(params, x)
    c = params["index"]
    def lo(l):
        rest = np.delete(l, c, axis=1)
        m = rest.max(1)
        return l[:, c] - (m + np.log(np.exp(rest - m[:, None]).sum(1)))
    a, b = lo(x[:, :4]), lo(x[:, 4:])
    return (a * np.abs(a) + b * np.abs(b)) / (np.abs(a) + np.abs(b) + 1e-9)


def main() -> None:
    d = merged()
    x = np.hstack([d["la"], d["lc"]]).astype(np.float32)
    valid, folds = d["label"] != -1, sorted(set(d["fold"].tolist()))
    out: dict = {"input": "A 로짓 4 + C1 로짓 4 (label_map 순서 normal,fall,abandon,theft)", "target_recall": TARGET, "classes": {}}
    for c, name in CLASSES.items():
        y = np.where(d["cls"] == c, 1, 0).astype(np.int64)
        z = np.zeros(len(y))
        for k in folds:
            tr, te = valid & (d["fold"] != k), d["fold"] == k
            p = _fit_linear(x[tr], y[tr])(x[te])
            z[te] = np.log(p + 1e-6) - np.log(1 - p + 1e-6)
        clips, cand = clips_for(d, z, c), candidates_from_values(z[valid], 120)
        ok = [(t, r) for t, r in ((t, evaluate(clips, t)) for t in cand) if r["recall"] >= TARGET]
        t, r = min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"]))
        params = fit_params(x[valid], y[valid])
        # 계수 검증: 내보낸 계수의 점수가 _fit_linear 예측(확률->로그 오즈)과 같은지
        p_ref = _fit_linear(x[valid], y[valid])(x[valid][:2000])
        p_new = 1 / (1 + np.exp(-stack_z(params, x[valid][:2000].astype(np.float64))))     # 로그 오즈끼리 비교하면 극단에서 포화되므로 확률로 비교
        assert np.abs(p_new - p_ref).max() < 1e-3, f"계수 불일치 {np.abs(p_new - p_ref).max():.2e}"
        out["classes"][SERVICE_KEY[c]] = {"index": c, "threshold_logit": float(t), "oof_recall": r["recall"], "oof_fa_per_hour": r["fa_per_hour"], **params}
        print(f"{name}: thr(로그오즈) {t:.2f} 재현율 {r['recall']:.3f} 오경보 {r['fa_per_hour']:.2f}/시간", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
