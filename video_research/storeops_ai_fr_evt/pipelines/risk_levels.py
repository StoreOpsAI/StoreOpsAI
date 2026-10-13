"""위험도 단계용 동작점: 스택 OOF 점수에서 클래스별로 (a) 재현율>=0.95, (b) 재현율>=0.90, (c) 오경보 예산(0.42/시간) 이하에서 재현율 최대인 임계값을 뽑는다.
임의 값 없음: 예산 0.42/시간은 product-decisions의 '카메라당 하루 10건 이하'(24h 기준)에서 온 값. 서비스 단계 경계로 쓰려면 stack.json에 넣는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pipelines.compare_backbones import _fit_linear
from pipelines.ensemble_eval import merged
from pipelines.eval_finetune import CLASSES, evaluate
from pipelines.fusion_eval import clips_for
from pipelines.oof_search import candidates_from_values

BUDGET_PER_HOUR = 10 / 24
STACK = Path("C:/Users/user/Desktop/무인 매장 에이전트/storeops_ai/training/runs/ours_stack/stack.json")
KEY = {"전도": "쓰러짐", "유기": "쓰레기 투기", "절도": "절도"}


def main() -> None:
    d = merged()
    x = np.hstack([d["la"], d["lc"]]).astype(np.float32)
    valid, folds = d["label"] != -1, sorted(set(d["fold"].tolist()))
    spec = json.loads(STACK.read_text(encoding="utf-8"))
    for c, name in CLASSES.items():
        y = np.where(d["cls"] == c, 1, 0).astype(np.int64)
        z = np.zeros(len(y))
        for k in folds:
            tr, te = valid & (d["fold"] != k), d["fold"] == k
            p = _fit_linear(x[tr], y[tr])(x[te])
            z[te] = np.log(p + 1e-6) - np.log(1 - p + 1e-6)
        clips = clips_for(d, z, c)
        rows = [(t, evaluate(clips, t)) for t in candidates_from_values(z[valid], 160)]
        def best_fa(rec):
            ok = [(t, r) for t, r in rows if r["recall"] >= rec]
            return min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"])) if ok else None
        ok_b = [(t, r) for t, r in rows if r["fa_per_hour"] <= BUDGET_PER_HOUR]
        pts = {"재현율>=0.95": best_fa(0.95), "재현율>=0.90": best_fa(0.90), f"오경보<={BUDGET_PER_HOUR:.2f}/시간": max(ok_b, key=lambda tr: (tr[1]["recall"], -tr[0])) if ok_b else None}
        print(f"[{name}]")
        for k, v in pts.items():
            print(f"  {k:18s}", "만족 없음" if v is None else f"임계값(로그오즈) {v[0]:5.2f} | 재현율 {v[1]['recall']:.3f} | 오경보 {v[1]['fa_per_hour']:.2f}/시간")
        keys = list(pts)
        lv = {"주의": pts[keys[0]], "경고": None, "높음": pts[keys[2]]}
        spec["classes"][KEY[name]]["levels"] = {"주의": lv["주의"][0], "경고": spec["classes"][KEY[name]]["threshold_logit"], "높음": lv["높음"][0]}   # 경고 = 발급 임계값(재현율>=0.9)
    STACK.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print("단계 경계를 stack.json에 저장")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
