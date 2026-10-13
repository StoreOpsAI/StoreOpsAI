"""손 크롭(H)·관절 헤드(P) 스트림 효과 비교: A+C1 대 A+C1+H / A+C1+P 동적 결합, 날짜 5-fold OOF, 이벤트 단위(개발 OOF만, 공식 validation 미사용).

- 같은 창(A·C1·H 모두 OOF 점수가 있는 (영상, 시작 위치))에서만 비교한다. H는 반품 영상을 쓰지 않으므로(사용자 지시) 이 창 집합에는 반품이 없다.
  참고로 서비스 기준(반품 포함 전체 창)의 A+C1도 따로 낸다. 정상 구성이 달라서 두 숫자는 직접 비교하지 않는다.
- 결합: 동적 결합 z = sum(x|x|) / sum(|x|), x는 스트림별 그 행동 로그 오즈(파라미터 없음, export_fusion.dynamic_fuse의 3개 확장).
- 임계값: (1) 모든 fold OOF에서 재현율 >= 목표 + 오경보 창 최소(export_fusion과 같은 기준), (2) 중첩: 보류 fold를 뺀 나머지에서 정한 값으로 보류 fold를 채점해 합친다.
사용: python -m pipelines.eval_hand_fusion
"""
from __future__ import annotations

import sys

import numpy as np

from pipelines.ensemble_eval import load_source, merged
from pipelines.eval_finetune import CLASSES, evaluate, log_odds
from pipelines.fusion_eval import clips_for
from pipelines.oof_search import candidates_from_values
from pipelines.export_fusion import CLASS_TARGET, TARGET

PRINT = {"A+C1": ("a", "c"), "A+C1+H": ("a", "c", "h"), "A+C1+P": ("a", "c", "p"), "A+C1+H+P": ("a", "c", "h", "p"), "P": ("p",)}


def fuse(xs: list[np.ndarray]) -> np.ndarray:
    return sum(x * np.abs(x) for x in xs) / (sum(np.abs(x) for x in xs) + 1e-9)


def common_windows() -> dict[str, np.ndarray]:
    """A·C1·H가 모두 점수를 가진 fold의 공통 창. la/lc/lh = 4클래스 로짓."""
    d = merged()
    h, p = load_source("H"), load_source("P")
    rows = {k: [] for k in ("clip", "start", "label", "cls", "behavior", "fold", "la", "lc", "lh", "lp")}
    for k in sorted(set(d["fold"].tolist()) & set(h) & set(p)):
        key_h = {(c, int(s)): i for i, (c, s) in enumerate(zip(h[k]["clip"], h[k]["start"]))}
        key_p = {(c, int(s)): i for i, (c, s) in enumerate(zip(p[k]["clip"], p[k]["start"]))}
        for i in np.flatnonzero(d["fold"] == k):
            j, m = key_h.get((d["clip"][i], int(d["start"][i]))), key_p.get((d["clip"][i], int(d["start"][i])))
            if j is None or m is None:
                continue
            for key in ("clip", "start", "label", "cls", "behavior", "fold", "la", "lc"):
                rows[key].append(d[key][i])
            rows["lh"].append(h[k]["logits"][j]); rows["lp"].append(p[k]["logits"][m])
    return {k: np.array(v) for k, v in rows.items()}


def pick(clips: list[dict], cand: np.ndarray, target: float):
    ok = [(t, r) for t, r in ((t, evaluate(clips, t)) for t in cand) if r["recall"] >= target]
    return min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"])) if ok else (None, None)


def subset(d: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[m] for k, v in d.items()}


def nested(d: dict[str, np.ndarray], z: np.ndarray, c: int, target: float) -> tuple[float, float]:
    """보류 fold를 뺀 나머지에서 임계값을 정해 보류 fold를 채점하고 합친 (재현율, 오경보/시간)."""
    hits = events = false = hours = 0.0
    for k in sorted(set(d["fold"].tolist())):
        tr, te = d["fold"] != k, d["fold"] == k
        t, _ = pick(clips_for(subset(d, tr), z[tr], c), candidates_from_values(z[tr & (d["label"] != -1)], 120), target)
        if t is None:
            continue
        r = evaluate(clips_for(subset(d, te), z[te], c), t)
        hits += r["recall"] * r["events"]; events += r["events"]; false += r["false"]; hours += r["hours"]
    return hits / max(events, 1), false / max(hours, 1e-9)


def main() -> None:
    d = common_windows()
    valid = d["label"] != -1
    print(f"공통 창 {len(d['clip'])}개, 영상 {len(set(d['clip']))}개, fold {sorted(set(d['fold'].tolist()))} (반품 제외)")
    for c, name in CLASSES.items():
        if c == 0:
            continue
        target = CLASS_TARGET.get(c, TARGET)
        print(f"\n[{name}] 목표 재현율 {target}")
        src = {"a": log_odds(d["la"], c), "c": log_odds(d["lc"], c), "h": log_odds(d["lh"], c), "p": log_odds(d["lp"], c)}
        for label, keys in PRINT.items():
            z = fuse([src[k] for k in keys])
            t, r = pick(clips_for(d, z, c), candidates_from_values(z[valid], 120), target)
            n_rec, n_fa = nested(d, z, c, target)
            row = f"OOF 임계 {t:.2f}: 재현율 {r['recall']:.3f} 오경보 {r['fa_per_hour']:.2f}/시간" if t is not None else "OOF 목표 불가"
            print(f"  {label:7s} | {row} | 중첩: 재현율 {n_rec:.3f} 오경보 {n_fa:.2f}/시간", flush=True)
    ref = merged()
    print(f"\n참고(서비스 기준, 반품 포함 전체 창 {len(ref['clip'])}개) A+C1:")
    for c, name in CLASSES.items():
        if c == 0:
            continue
        z = fuse([log_odds(ref["la"], c), log_odds(ref["lc"], c)])
        t, r = pick(clips_for(ref, z, c), candidates_from_values(z[ref["label"] != -1], 120), CLASS_TARGET.get(c, TARGET))
        print(f"  {name}: 임계 {t:.2f} 재현율 {r['recall']:.3f} 오경보 {r['fa_per_hour']:.2f}/시간")


def _self_check() -> None:
    assert abs(fuse([np.array([4.0]), np.array([4.0]), np.array([4.0])])[0] - 4.0) < 1e-6
    assert abs(fuse([np.array([-3.0]), np.array([1.0])])[0] - (-2.0)) < 1e-6   # export_fusion.dynamic_fuse와 같다
    assert abs(fuse([np.array([0.0])])[0]) < 1e-6
    print("ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _self_check() if len(sys.argv) > 1 and sys.argv[1] == "--self-check" else main()
