"""6클래스 train(재대입) 대 개발 OOF 차이: 같은 영상·같은 창에서 같은 지표·같은 임계값으로 비교한다(공식 validation은 쓰지 않는다).

- train: 최종 A·C1이 학습한 영상의 창을 다시 채점한 로짓(score_train_six.py 결과, output/train_eval/logits6_*.npz).
- OOF: 같은 창을 그 창의 fold를 학습하지 않은 모델이 채점한 로짓. 창 집합을 train과 똑같이 맞춘다.
- 임계값: OOF **전체 창**에서 eval_six_class와 같은 방식(목표 재현율 전도 0.95·나머지 0.90, 오경보 창 최소)으로 행동마다 한 번 정해 두 쪽에 동일 적용한다. train에서 다시 정하지 않는다.
- 지표: 이벤트 단위(엄격) Precision / Recall / F1 / 오경보(시간당). A, C1, A+C1(동적 결합).
사용: python -m pipelines.eval_six_train_gap   (EVAL_A_DIR, EVAL_C1_DIR, TRAIN_SUFFIX: 검수 정상 모델이면 _audit)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from pipelines.eval_finetune import evaluate, log_odds, per_clip
from pipelines.eval_six_class import DEFAULT_TARGET, NAMES, TARGET, load, pick
from pipelines.eval_six_fusion import align, with_score
from pipelines.eval_six_summary import event_counts
from pipelines.finetune_s3d import APP_ROOT
from pipelines.oof_search import candidates_from_values

TRAIN_DIR = APP_ROOT / "output" / "train_eval"


def train_table(suffix: str) -> dict[str, np.ndarray]:
    a, c = np.load(TRAIN_DIR / f"logits6_full{suffix}.npz"), np.load(TRAIN_DIR / f"logits6_c1{suffix}.npz")
    keys_c = {(str(k), int(s)): i for i, (k, s) in enumerate(zip(c["clip"], c["start"]))}
    keep = [i for i, (k, s) in enumerate(zip(a["clip"], a["start"])) if (str(k), int(s)) in keys_c]
    if len(keep) != len(a["clip"]) or len(keys_c) != len(keep):
        raise ValueError(f"A와 C1의 train 창이 다르다: A {len(a['clip'])}, C1 {len(keys_c)}, 공통 {len(keep)}")
    d = {k: a[k] for k in ("clip", "start", "label", "cls", "behavior", "fold")}
    d["logits"] = a["logits"]
    d["logits_c1"] = c["logits"][[keys_c[(str(k), int(s))] for k, s in zip(a["clip"], a["start"])]]
    return d


def subset_same(d: dict[str, np.ndarray], keys: set[tuple[str, int]]) -> dict[str, np.ndarray]:
    m = np.array([(str(k), int(s)) in keys for k, s in zip(d["clip"], d["start"])])
    return {k: v[m] for k, v in d.items()}


def metrics(d: dict[str, np.ndarray], c: int, thr: float) -> tuple[float, float, float, float, int]:
    clips = per_clip(d, c)
    events, hits, tr, fr = event_counts(clips, thr)
    r = evaluate(clips, thr)
    prec, rec = tr / max(tr + fr, 1), hits / max(events, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9), r["fa_per_hour"], events


def main() -> None:
    a_dir, c_dir = os.getenv("EVAL_A_DIR", "finetune_s3d6"), os.getenv("EVAL_C1_DIR", "finetune_s3d_c16")
    suffix = os.getenv("TRAIN_SUFFIX", "")
    oof = align(load(a_dir, []), load(c_dir, []))
    train = train_table(suffix)
    keys = {(str(k), int(s)) for k, s in zip(train["clip"], train["start"])}
    oof_same = subset_same(oof, keys)
    if len(oof_same["clip"]) != len(train["clip"]):
        raise ValueError(f"OOF에 없는 train 창이 있다: train {len(train['clip'])}, OOF 공통 {len(oof_same['clip'])}")
    print(f"A = {a_dir}, C1 = {c_dir}: train 창 {len(train['clip'])}개(영상 {len(set(train['clip'].tolist()))}개), 애매 창 제외. 임계값은 OOF 전체 창에서 정해 양쪽에 동일 적용")
    print("이벤트 단위(엄격) Precision / Recall / F1 / 오경보 시간당   train(재대입) | OOF(같은 창) | 차이(F1, 오경보)")
    gap = {m: [] for m in ("A", "C1", "A+C1")}
    for c in range(1, 6):
        print(f"\n[{NAMES[c]}]")
        for m in gap:
            full = with_score(oof, c, m)
            valid = full["label"] != -1
            t, _ = pick(per_clip(full, c), candidates_from_values(log_odds(full["logits"], c)[valid], 120), TARGET.get(c, DEFAULT_TARGET))
            if t is None:
                print(f"  {m:5s} 임계값 없음"); continue
            tr = metrics(with_score(train, c, m), c, t)
            oo = metrics(with_score(oof_same, c, m), c, t)
            gap[m].append((tr[2], oo[2]))
            print(f"  {m:5s} P {tr[0]:.3f} R {tr[1]:.3f} F1 {tr[2]:.3f} FA {tr[3]:5.2f} | P {oo[0]:.3f} R {oo[1]:.3f} F1 {oo[2]:.3f} FA {oo[3]:5.2f} | F1 {tr[2] - oo[2]:+.3f}, FA {oo[3] - tr[3]:+.2f}  (사건 {tr[4]}개)")
    print("\n행동 평균 Event F1: train(재대입) / OOF / 차이")
    for m, v in gap.items():
        a, b = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        print(f"  {m:5s} {a:.3f} / {b:.3f} / {a - b:+.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
