"""장소 제외 평가(SITE_FOLDS=1로 학습한 finetune_s3d6_audit_site) 대 날짜 fold(finetune_s3d6_audit)를 장소 × 행동별로 비교한다.

- 임계값은 날짜 fold OOF 전체에서 행동마다 한 번 정해 두 쪽에 똑같이 적용한다(장소 제외 결과로 임계값을 다시 정하지 않는다).
- 장소 제외 쪽은 그 장소를 통째로 빼고 학습한 모델이 본 결과, 날짜 fold 쪽은 같은 장소가 학습에 들어간 모델이 안 본 날짜를 본 결과다.
  두 값의 차이가 클수록 모델이 "그 장소의 배경"에 기대고 있다는 뜻이다(재현율 하락, 오경보 증가).
- 신뢰구간은 그 장소 영상을 다시 뽑는 부트스트랩이며 장소가 작으면 매우 넓다.
사용: python -m pipelines.eval_site_holdout   (EVAL_DATE_DIR, EVAL_SITE_DIR로 폴더 변경 가능)
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

from pipelines import eval_finetune as E
from pipelines.eval_finetune import evaluate, log_odds, per_clip
from pipelines.eval_six_class import DEFAULT_TARGET, NAMES, TARGET, load, pick
from pipelines.global_split import SPLIT_PATH
from pipelines.oof_search import candidates_from_values

BOOT = 200


def subset(d: dict[str, np.ndarray], clips: set[str]) -> dict[str, np.ndarray]:
    m = np.isin(d["clip"], list(clips))
    return {k: v[m] for k, v in d.items()}


def interval(clips: list[dict], thr: float, rng: np.random.Generator) -> tuple[tuple[float, float], tuple[float, float]]:
    rec, fa = [], []
    for _ in range(BOOT):
        r = evaluate([clips[i] for i in rng.integers(0, len(clips), len(clips))], thr)
        rec.append(r["recall"]); fa.append(r["fa_per_hour"])
    return tuple(np.percentile(rec, [2.5, 97.5])), tuple(np.percentile(fa, [2.5, 97.5]))


def cell(d: dict[str, np.ndarray], c: int, thr: float, rng: np.random.Generator) -> str:
    clips = per_clip(d, c)
    if not clips:
        return "자료 없음"
    r = evaluate(clips, thr)
    if r["events"] == 0:
        return f"사건 0개, 오경보 {r['fa_per_hour']:.2f}/시간"
    (rl, rh), (fl, fh) = interval(clips, thr, rng)
    return f"재현율 {r['recall']:.2f} [{rl:.2f}~{rh:.2f}] 오경보 {r['fa_per_hour']:.2f} [{fl:.2f}~{fh:.2f}] (사건 {r['events']})"


def main() -> None:
    date_dir = os.getenv("EVAL_DATE_DIR", "finetune_s3d6_audit")
    site_dir = os.getenv("EVAL_SITE_DIR", "finetune_s3d6_audit_site")
    d_date, d_site = load(date_dir, []), load(site_dir, [])
    site_of = {r["clip"]: r["site"] for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))}
    sites = sorted({site_of[c] for c in d_site["clip"].tolist()})
    missing = sorted(set(site_of[c] for c in d_site["clip"].tolist()) ^ set(site_of[c] for c in d_date["clip"].tolist()))
    print(f"날짜 fold {date_dir}: 창 {len(d_date['clip'])}개 / 장소 제외 {site_dir}: 창 {len(d_site['clip'])}개, 장소 {sites}" + (f", 한쪽에만 있는 장소 {missing}" if missing else ""))
    thr = {}
    for c in range(1, 6):
        valid = d_date["label"] != -1
        thr[c], _ = pick(per_clip(d_date, c), candidates_from_values(log_odds(d_date["logits"], c)[valid], 120), TARGET.get(c, DEFAULT_TARGET))
        print(f"  {NAMES[c]} 임계값(날짜 fold OOF, 양쪽에 동일 적용): {thr[c]}")
    rng = np.random.default_rng(0)
    for c in range(1, 6):
        if thr[c] is None:
            print(f"\n[{NAMES[c]}] 목표 재현율을 만족하는 임계값이 없어 건너뜀")
            continue
        print(f"\n[{NAMES[c]}] 장소별 (대괄호 = 클립 부트스트랩 95% 구간)")
        for s in sites:
            clips = {cl for cl in d_site["clip"].tolist() if site_of[cl] == s}
            print(f"  {s:4s} 날짜 fold : {cell(subset(d_date, clips), c, thr[c], rng)}")
            print(f"       장소 제외 : {cell(subset(d_site, clips), c, thr[c], rng)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
