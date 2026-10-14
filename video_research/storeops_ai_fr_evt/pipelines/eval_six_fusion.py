"""6클래스 A(화면 전체) + C1(사람 크롭) 동적 결합 OOF 평가: A 단독, C1 단독, A+C1을 같은 창에서 행동마다 비교한다.

- 결합 수식은 서비스와 같다: z = (a|a| + b|b|) / (|a| + |b|), a = A의 그 행동 로그 오즈, b = C1의 로그 오즈. 학습하는 파라미터가 없다.
- 평가는 eval_six_class.class_report를 그대로 쓴다: OOF 임계값(행동마다 목표 재현율을 만족하는 값 중 오경보 창이 가장 적은 값, 전도 0.95 나머지 0.90),
  중첩 임계값(안 본 fold에서 평가), 클립 부트스트랩 95% 구간, 매장이동 영상 오경보(MVP 기준). 채택 기준은 A 단독 대비 오경보 신뢰구간이 겹치지 않을 때만.
- 공식 validation은 읽지 않는다(개발용).
사용: python -m pipelines.eval_six_fusion   (EVAL_A_DIR, EVAL_C1_DIR, EVAL_FOLDS로 폴더·fold 변경 가능)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from pipelines.eval_finetune import log_odds
from pipelines.eval_six_class import NAMES, class_report, load

N_CLASS = 6


def align(d_a: dict[str, np.ndarray], d_c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """C1 로짓을 A의 창 순서로 맞춘다. 두 쪽의 (영상, 시작) 집합이 다르면 예외."""
    pos = {(c, int(s)): i for i, (c, s) in enumerate(zip(d_c["clip"], d_c["start"]))}
    keys = [(c, int(s)) for c, s in zip(d_a["clip"], d_a["start"])]
    missing = [k for k in keys if k not in pos]
    if missing or len(pos) != len(keys):
        raise ValueError(f"A와 C1의 창이 다르다: A {len(keys)}개, C1 {len(pos)}개, C1에 없는 A 창 {len(missing)}개")
    out = dict(d_a)
    out["logits_c1"] = d_c["logits"][[pos[k] for k in keys]]
    return out


def dynamic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * np.abs(a) + b * np.abs(b)) / (np.abs(a) + np.abs(b) + 1e-9)


def with_score(d: dict[str, np.ndarray], c: int, mode: str) -> dict[str, np.ndarray]:
    """eval_six_class가 log_odds(logits, c)로 점수를 읽으므로, 그 행동 열에 원하는 점수를 넣은 로짓을 만든다.
    나머지 5열이 0이면 log_odds = 열값 - log 5라서 열값에 log 5를 더해 점수가 그대로 나오게 한다."""
    a, b = log_odds(d["logits"], c), log_odds(d["logits_c1"], c)
    z = {"A": a, "C1": b, "A+C1": dynamic(a, b)}[mode]
    fake = np.zeros((len(z), N_CLASS), np.float64)
    fake[:, c] = z + np.log(N_CLASS - 1)
    return {**d, "logits": fake}


def main() -> None:
    a_dir, c_dir = os.getenv("EVAL_A_DIR", "finetune_s3d6"), os.getenv("EVAL_C1_DIR", "finetune_s3d_c16")
    folds = [x for x in os.getenv("EVAL_FOLDS", "").split(",") if x]
    d = align(load(a_dir, folds), load(c_dir, folds))
    rng = np.random.default_rng(0)
    print(f"A = {a_dir}, C1 = {c_dir}: fold {sorted(set(d['fold'].tolist()))}, 창 {len(d['clip'])}개, 영상 {len(set(d['clip']))}개, 애매 창 {int((d['label'] == -1).sum())}개 제외")
    print("행동마다 A 단독 / C1 단독 / A+C1 동적 결합 (대괄호 = 클립 부트스트랩 95% 구간, 채택 기준: A 단독 대비 오경보 구간이 겹치지 않을 때만)")
    for c in range(1, N_CLASS):
        print(f"\n[{NAMES[c]}]")
        for mode in ("A", "C1", "A+C1"):
            class_report(with_score(d, c, mode), c, mode, rng)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
