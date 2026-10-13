"""점수 보정: OOF 로짓을 확률로 바꾸는 온도 스케일링(p = sigmoid(z/T), 모수 1개; Guo 외 ICML 2017)과 신뢰도 곡선 요약. Platt(a*z+b)는 점수가 거의 완전히 분리되면 발산해서(a~1e7) 쓰지 않는다.

로짓은 포화되지 않아서 기준 후보를 안정적으로 만들 수 있고, 확률은 서비스 표시용으로만 보정해서 쓴다(점수는 확률이 아니다).
보정은 dev의 OOF 점수·라벨로만 맞춘다(공식 validation 미사용). a, b는 selected_params와 함께 저장한다.
"""
from __future__ import annotations

import numpy as np


def fit_platt(z: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """로짓 z, 라벨 y(0/1)에서 p = sigmoid(a*z + b)의 (a, b). 뉴턴법. 클래스 불균형은 있는 그대로 둔다(실제 비율에서의 확률)."""
    a, b = 1.0, 0.0
    z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([((p - y) * z).sum(), (p - y).sum()])
        w = p * (1 - p) + 1e-9
        h = np.array([[(w * z * z).sum(), (w * z).sum()], [(w * z).sum(), w.sum()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """p = sigmoid(z / T)의 T > 0. 로그 우도를 log T에 대해 황금분할 탐색(모수 1개라 발산하지 않는다)."""
    z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)

    def nll(log_t: float) -> float:
        u = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, u) - y * u))

    lo, hi, phi = -5.0, 8.0, (np.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        lo, hi = (lo, b) if nll(a) < nll(b) else (a, hi)
    return float(np.exp((lo + hi) / 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """예측 확률 구간별 실제 양성 비율과 기대 보정 오차(ECE)."""
    edges, idx = np.linspace(0, 1, bins + 1), np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            rows.append({"bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"ece": float(ece), "bins": rows}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(-2, 2, 4000), rng.normal(2, 2, 1000)])
    y = np.concatenate([np.zeros(4000), np.ones(1000)])
    a, b = fit_platt(z, y)
    t = fit_temperature(z, y)
    assert 0.3 < t < 3.0, t  # 이미 보정된 합성 로짓이므로 T는 1 근처
    p = 1 / (1 + np.exp(-(a * z + b)))
    assert reliability(p, y)["ece"] < 0.03, reliability(p, y)["ece"]  # 보정 후 신뢰도 곡선이 대각선에 붙는다
    over = 1 / (1 + np.exp(-(3 * z)))  # 과신(포화) 확률
    assert reliability(over, y)["ece"] > reliability(p, y)["ece"]
    print("ok", round(a, 3), round(b, 3))
