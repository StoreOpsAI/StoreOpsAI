"""동적 결합(확신 큰 쪽에 비중) 내보내기: 행동별 점수 z = (a|a| + b|b|) / (|a| + |b|), a = A 모델의 그 행동 로그 오즈, b = C1 모델의 로그 오즈.

학습하는 파라미터가 없다(수식). 임계값만 날짜 5-fold OOF에서 이벤트 재현율 >= TARGET이면서 오경보 창 비율이 최소인 값으로 정한다(export_stack과 같은 기준).
비교 근거: 개발 OOF에서 로지스틱 스택(입력 8개)·평균·투표·고정 가중·동적을 비교(임계값을 다른 fold에서 정해 안 본 fold에서 평가)했을 때 동적이 오경보 합계가 가장 낮았다(차이는 잡음 범위, 사용자 선택).
출력: <서비스>/training/runs/ours_stack/fusion.json (서비스 기본 설정이 이 파일을 읽는다). 기존 로지스틱 스택은 stack_logistic_old.json으로 보존한다.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from pipelines.ensemble_eval import merged
from pipelines.eval_finetune import CLASSES, evaluate, log_odds
from pipelines.fusion_eval import clips_for
from pipelines.oof_search import candidates_from_values

TARGET = 0.90
SMOOTH: dict = {}   # 이동 평균 끔(즉시 알림, 사용자 결정 2026-10-07). 켜려면 {1: 3, 2: 3}: 개발 OOF에서 전도 1.89→0.94, 유기 2.60→1.72 오경보/시간, 대신 지연 +4초
CLASS_TARGET = {1: 0.95}   # 전도만 0.95: 개발 fold별 재현율이 0.86~0.99로 흔들려 새 날짜에서 0.9 아래로 내려오는 것을 막는 여유(가장 낮은 fold 0.86 -> 0.90). 사용자 결정
SERVICE_KEY = {1: "쓰러짐", 2: "쓰레기 투기", 3: "절도"}
DIR = Path("C:/Users/user/Desktop/무인 매장 에이전트/storeops_ai/training/runs/ours_stack")


def causal_mean(d: dict, z: np.ndarray, m: int) -> np.ndarray:
    """영상마다 시간순으로 과거 m개 창(처음에는 있는 만큼)의 평균. 서비스의 상태 평균과 같다."""
    out = z.copy()
    for clip in np.unique(d["clip"]):
        idx = np.flatnonzero(d["clip"] == clip)
        idx = idx[np.argsort(d["start"][idx])]
        cs = np.cumsum(np.insert(z[idx], 0, 0.0))
        lo = np.maximum(0, np.arange(1, len(idx) + 1) - m)
        out[idx] = (cs[1:] - cs[lo]) / (np.arange(1, len(idx) + 1) - lo)
    return out


def dynamic_fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * np.abs(a) + b * np.abs(b)) / (np.abs(a) + np.abs(b) + 1e-9)


def main() -> None:
    d = merged()
    valid = d["label"] != -1
    out = {"method": "dynamic", "formula": "z = (a|a| + b|b|) / (|a| + |b|), a = A 로그 오즈, b = C1 로그 오즈", "target_recall": TARGET, "classes": {}}
    for c, name in CLASSES.items():
        m = SMOOTH.get(c, 1)
        z = causal_mean(d, dynamic_fuse(log_odds(d["la"], c), log_odds(d["lc"], c)), m) if m > 1 else dynamic_fuse(log_odds(d["la"], c), log_odds(d["lc"], c))
        clips, cand = clips_for(d, z, c), candidates_from_values(z[valid], 120)
        target = CLASS_TARGET.get(c, TARGET)
        ok = [(t, r) for t, r in ((t, evaluate(clips, t)) for t in cand) if r["recall"] >= target]
        t, r = min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"]))
        out["classes"][SERVICE_KEY[c]] = {"index": c, "method": "dynamic", "smooth": m, "target_recall": target, "threshold_logit": float(t), "oof_recall": r["recall"], "oof_fa_per_hour": r["fa_per_hour"]}
        print(f"{name}(평활 m={m}): thr(로그오즈) {t:.2f} 재현율 {r['recall']:.3f} 오경보 {r['fa_per_hour']:.2f}/시간", flush=True)
    if (DIR / "stack.json").is_file() and not (DIR / "stack_logistic_old.json").is_file():
        shutil.copy(DIR / "stack.json", DIR / "stack_logistic_old.json")
    (DIR / "fusion.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", DIR / "fusion.json")


def _self_check() -> None:
    assert abs(dynamic_fuse(np.array([4.0]), np.array([4.0]))[0] - 4.0) < 1e-6       # 같으면 그대로
    assert abs(dynamic_fuse(np.array([-3.0]), np.array([1.0]))[0] - (-2.0)) < 1e-6      # (-9+1)/4 = -2: 평균(-1)보다 확신 큰 쪽(-3)에 끌려간다
    assert abs(dynamic_fuse(np.array([0.0]), np.array([0.0]))[0]) < 1e-6               # 0이면 0
    print("ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _self_check() if len(sys.argv) > 1 and sys.argv[1] == "--self-check" else main()
