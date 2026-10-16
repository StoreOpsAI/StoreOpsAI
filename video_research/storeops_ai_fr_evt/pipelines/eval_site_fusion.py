"""장소 제외 평가의 결합판: A·C1·P와 그 동적 결합(C1+P, A+C1, A+C1+P)을 장소 × 행동별로 날짜 fold와 비교한다.

- eval_site_holdout.py와 같은 규칙이다: 임계값은 같은 조합의 날짜 fold OOF 전체에서 행동마다 한 번 정해 장소 제외 쪽에 그대로 쓴다.
- 세 스트림(A, C1, P)에 모두 있는 창만 쓴다(P는 관절 캐시가 있는 영상만). 날짜 fold 쪽과 장소 제외 쪽 모두 같은 창 집합으로 맞춘다.
- 공식 validation은 읽지 않는다(개발용).
사용: python -m pipelines.eval_site_fusion
환경변수: EVAL_{A,C1,P}_{DATE,SITE}_DIR 로 폴더 변경 (기본: finetune_s3d6_audit[_site], finetune_s3d_c16_audit[_site], pose_head6_audit[_site])
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

from pipelines.eval_finetune import log_odds, per_clip
from pipelines.eval_six_class import DEFAULT_TARGET, NAMES, TARGET, load, pick
from pipelines.eval_six_fusion import align, with_score
from pipelines.eval_site_holdout import cell, subset
from pipelines.global_split import SPLIT_PATH
from pipelines.oof_search import candidates_from_values

MODES = ("A", "C1", "P", "A+P", "C1+P", "A+C1", "A+C1+P")


def window_keys(d: dict[str, np.ndarray]) -> set[tuple[str, int]]:
    return {(c, int(s)) for c, s in zip(d["clip"], d["start"])}


def restrict(d: dict[str, np.ndarray], keys: set[tuple[str, int]]) -> dict[str, np.ndarray]:
    m = np.array([(c, int(s)) in keys for c, s in zip(d["clip"], d["start"])])
    return {k: v[m] for k, v in d.items()}


def load_aligned(kind: str) -> dict[str, np.ndarray]:
    """kind = DATE 또는 SITE. A·C1·P의 공통 창만 남겨 A의 순서로 맞춘다."""
    suffix = "_site" if kind == "SITE" else ""
    raw = {s: load(os.getenv(f"EVAL_{s}_{kind}_DIR", default + suffix), []) for s, default in
           (("A", "finetune_s3d6_audit"), ("C1", "finetune_s3d_c16_audit"), ("P", "pose_head6_audit"))}
    keys = set.intersection(*(window_keys(d) for d in raw.values()))
    print(f"{kind}: " + ", ".join(f"{s} {len(window_keys(d))}창" for s, d in raw.items()) + f" -> 공통 {len(keys)}창")
    r = {s: restrict(d, keys) for s, d in raw.items()}
    return align(r["A"], r["C1"], None, r["P"])


def main() -> None:
    d_date, d_site = load_aligned("DATE"), load_aligned("SITE")
    site_of = {r["clip"]: r["site"] for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))}
    sites = sorted({site_of[c] for c in d_site["clip"].tolist()})
    rng = np.random.default_rng(0)
    for c in range(1, 6):
        print(f"\n[{NAMES[c]}]  (대괄호 = 클립 부트스트랩 95% 구간, 임계값은 같은 조합의 날짜 fold OOF에서 정함)")
        for mode in MODES:
            dd, ds = with_score(d_date, c, mode), with_score(d_site, c, mode)
            valid = dd["label"] != -1
            thr, _ = pick(per_clip(dd, c), candidates_from_values(log_odds(dd["logits"], c)[valid], 120), TARGET.get(c, DEFAULT_TARGET))
            if thr is None:
                print(f"  {mode:7s} 목표 재현율을 만족하는 임계값이 없어 건너뜀")
                continue
            print(f"  {mode:7s} 임계값 {thr:.2f}")
            for s in sites:
                clips = {cl for cl in ds["clip"].tolist() if site_of[cl] == s}
                print(f"     {s:4s} 날짜 fold : {cell(subset(dd, clips), c, thr, rng)}")
                print(f"          장소 제외 : {cell(subset(ds, clips), c, thr, rng)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
