"""6클래스 A+C1 공식 validation 평가(1회): 설정 동결 후 최종 A·C1이 채점한 validation 창을 개발 OOF에서 정한 임계값 그대로 평가한다.

- 입력: score_official_val.py(S3D_CLASSES=6)가 만든 output/official_val_six/logits_full.npz, logits_c1.npz.
- 임계값: 개발 OOF 전체 창에서 eval_six_class와 같은 방식(목표 재현율 전도 0.95·나머지 0.90, 오경보 창 최소)으로 행동마다 한 번 정한다. validation에서 조정하지 않는다.
- 지표: 이벤트 단위(엄격) Precision / Recall / F1 / 오경보(시간당). A, C1, A+C1. 같은 임계값으로 train(재대입)·OOF(같은 창이 아닌 전체)·validation을 나란히 적는다.
- 파손·방화는 처음 보는 validation이지만 날짜 격리가 없다(방화 82개 중 67개 날짜가 train과 겹침). 전도·유기·절도는 4클래스 평가에서 이미 봤으므로 블라인드가 아니다.
- 결과가 이미 있으면 FORCE_REEVAL=1이 아니면 거부한다(AGENTS.md: test는 동결 후 한 번만). 사용: python -m pipelines.eval_official_val_six
"""
from __future__ import annotations

import os
import sys

import numpy as np

from pipelines.eval_finetune import log_odds, per_clip
from pipelines.eval_six_class import DEFAULT_TARGET, NAMES, TARGET, load, pick
from pipelines.eval_six_fusion import align, with_score
from pipelines.eval_six_train_gap import metrics, train_table
from pipelines.finetune_s3d import APP_ROOT
from pipelines.oof_search import candidates_from_values

VAL_DIR = APP_ROOT / "output" / "official_val_six"
REPORT = VAL_DIR / "report.log"
BLIND = {4: "처음 봄(날짜 격리 없음)", 5: "처음 봄(날짜 격리 없음)", 1: "비블라인드(4클래스에서 4회 봄)", 2: "비블라인드(4클래스에서 4회 봄)", 3: "비블라인드(4클래스에서 4회 봄)"}


def load_val() -> dict[str, np.ndarray]:
    a, c = np.load(VAL_DIR / "logits_full.npz"), np.load(VAL_DIR / "logits_c1.npz")
    d = {k: a[k] for k in ("clip", "start", "label", "cls", "behavior", "logits")}
    return align(d, {"clip": c["clip"], "start": c["start"], "logits": c["logits"]})


class Tee:
    """출력을 화면과 메모리에 같이 남겨, 평가가 끝까지 간 뒤에만 보고서 파일을 만든다(중간에 실패해도 '이미 평가함'으로 막히지 않게)."""
    def __init__(self) -> None:
        self.buf, self.out = [], sys.stdout

    def write(self, t: str) -> int:
        self.buf.append(t)
        return self.out.write(t)

    def flush(self) -> None:
        self.out.flush()


def main() -> None:
    if REPORT.is_file() and os.getenv("FORCE_REEVAL") != "1":
        sys.exit(f"이미 평가했습니다: {REPORT}. 공식 validation은 한 번만 평가합니다(다시 하려면 사유를 기록하고 FORCE_REEVAL=1).")
    tee = sys.stdout = Tee()
    a_dir, c_dir = os.getenv("EVAL_A_DIR", "finetune_s3d6"), os.getenv("EVAL_C1_DIR", "finetune_s3d_c16")
    oof, val, train = align(load(a_dir, []), load(c_dir, [])), load_val(), train_table("")
    print(f"A = {a_dir}, C1 = {c_dir}: validation 영상 {len(set(val['clip'].tolist()))}개, 창 {len(val['clip'])}개. 임계값은 개발 OOF 전체 창에서 정해 동결")
    print("이벤트 단위(엄격) F1 / 오경보(시간당)   train(재대입) | OOF(전체 창) | 공식 validation")
    mean = {m: ([], [], []) for m in ("A", "C1", "A+C1")}
    for c in range(1, 6):
        print(f"\n[{NAMES[c]}] {BLIND[c]}")
        for m in mean:
            full = with_score(oof, c, m)
            valid = full["label"] != -1
            t, _ = pick(per_clip(full, c), candidates_from_values(log_odds(full["logits"], c)[valid], 120), TARGET.get(c, DEFAULT_TARGET))
            if t is None:
                print(f"  {m:5s} 임계값 없음"); continue
            tr, oo, va = (metrics(with_score(x, c, m), c, t) for x in (train, oof, val))
            for lst, r in zip(mean[m], (tr, oo, va)):
                lst.append(r[2])
            print(f"  {m:5s} 임계 {t:5.2f} | train F1 {tr[2]:.3f} FA {tr[3]:5.2f} | OOF F1 {oo[2]:.3f} FA {oo[3]:5.2f} | val P {va[0]:.3f} R {va[1]:.3f} F1 {va[2]:.3f} FA {va[3]:5.2f} (사건 {va[4]}개)")
    print("\n행동 평균 Event F1: train / OOF / 공식 validation")
    for m, (a, b, v) in mean.items():
        print(f"  {m:5s} {np.mean(a):.3f} / {np.mean(b):.3f} / {np.mean(v):.3f}")
    REPORT.write_text("".join(tee.buf), encoding="utf-8")
    sys.stdout = tee.out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
