"""6클래스 결합 OOF의 이벤트 단위 요약표: 행동마다 Event Precision / Recall / F1 / 오경보(시간당)를 A, C1, A+C1로 낸다.

임계값은 eval_six_class와 같은 방식(목표 재현율 전도 0.95·나머지 0.90을 만족하는 후보 중 오경보 창이 가장 적은 값)으로 OOF 전체에서 정한다.
Event Precision = 울린 경보 구간 중 그 행동 사건 창과 겹친 구간의 비율. 다른 행동 사건 구간의 경보는 오경보(엄격).
사용: python -m pipelines.eval_six_summary   (EVAL_A_DIR, EVAL_C1_DIR)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from pipelines.eval_finetune import log_odds, per_clip, runs
from pipelines.eval_six_class import DEFAULT_TARGET, NAMES, TARGET, load, pick
from pipelines.eval_six_fusion import align, with_score
from pipelines.oof_search import candidates_from_values


def event_counts(clips: list[dict], thr: float) -> tuple[int, int, int, int]:
    """(사건 영상 수, 적중 수, 사건과 겹친 경보 구간 수, 오경보 구간 수)"""
    events = hits = true_runs = false_runs = 0
    for c in clips:
        r = runs(c["z"], thr)
        if (c["y"] == 1).any():
            events += 1
            hits += any((c["y"][a:b + 1] == 1).any() for a, b in r)
        true_runs += sum(1 for a, b in r if (c["y"][a:b + 1] == 1).any())
        false_runs += sum(1 for a, b in r if not (c["y"][a:b + 1] != 0).any())
    return events, hits, true_runs, false_runs


def row(d: dict[str, np.ndarray], c: int) -> str:
    valid = d["label"] != -1
    clips = per_clip(d, c)
    t, r = pick(clips, candidates_from_values(log_odds(d["logits"], c)[valid], 120), TARGET.get(c, DEFAULT_TARGET))
    if t is None:
        return "목표 재현율을 만족하는 임계값 없음"
    events, hits, tr, fr = event_counts(clips, t)
    prec = tr / max(tr + fr, 1)
    rec = hits / max(events, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f"P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}  오경보 {r['fa_per_hour']:.2f}/시간  (사건 {events}, 경보 구간 {tr + fr})"


def main() -> None:
    a_dir, c_dir = os.getenv("EVAL_A_DIR", "finetune_s3d6"), os.getenv("EVAL_C1_DIR", "finetune_s3d_c16")
    d = align(load(a_dir, []), load(c_dir, []))
    print(f"A = {a_dir}, C1 = {c_dir}: 창 {len(d['clip'])}개  (이벤트 단위, 엄격, 날짜 5-fold OOF)")
    f1s = {m: [] for m in ("A", "C1", "A+C1")}
    for c in range(1, 6):
        print(f"\n[{NAMES[c]}]")
        for m in f1s:
            line = row(with_score(d, c, m), c)
            print(f"  {m:5s} {line}")
            f1s[m].append(float(line.split("F1 ")[1].split()[0]) if "F1 " in line else float("nan"))
    print("\n행동 평균 Event F1: " + "  ".join(f"{m} {np.nanmean(v):.3f}" for m, v in f1s.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
