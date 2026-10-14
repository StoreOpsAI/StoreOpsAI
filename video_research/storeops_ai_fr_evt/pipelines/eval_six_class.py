"""6클래스 S3D(정상·전도·유기·절도·파손·방화) 날짜 fold OOF 평가. 보류 fold의 모든 창을 쓰며 공식 validation은 읽지 않는다(개발용).

- 이벤트 단위: eval_finetune.evaluate(적중 = 그 행동 클립에서 경보 구간이 양성 창 포함, 오경보 = 양성이 아닌 구간의 경보).
  행동마다 목표 재현율(전도 0.95, 나머지 0.9)을 만족하면서 오경보 창이 가장 적은 임계값을 고른다(export_fusion과 같은 기준).
- 임계값 두 가지: (1) 모든 보류 fold를 모아 고른 값(OOF), (2) 보류 fold를 뺀 나머지 fold로 고른 값(중첩, 안 본 fold에서 평가).
- 신뢰구간: 클립을 다시 뽑는 부트스트랩(OOF 임계값 고정)으로 재현율·오경보/시간의 95% 구간.
- 창 단위: 6클래스 argmax의 혼동행렬과 클래스별 precision/recall/F1.
- MVP 기준(정상 = 매장이동 영상 전체, 걷기·서 있기): 같은 임계값에서 매장이동 영상의 경보 구간 수 / 시간도 함께 낸다. 앞의 이벤트 단위 오경보는 사건 영상의 사건 밖 구간까지 센 엄격한 기준이다.
- 기준선: 같은 창에서 기존 4클래스 A(finetune_s3d)의 전도·유기·절도 결과(정상 구성이 달라 반품은 제외한 같은 창만 비교).
- 폴더 지정: EVAL_DIR6(기본 finetune_s3d6). COMPARE_DIR6를 주면 그 폴더(예: finetune_s3d6_pre48_post48)를 같은 방식으로 이어서 낸다(정상 구성 비교).
사용: EVAL_FOLDS=0,1 python -m pipelines.eval_six_class
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

from pipelines import eval_finetune as E
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_finetune import STRIDE_SEC, evaluate, log_odds, per_clip, runs
from pipelines.oof_search import candidates_from_values

E.CLASSES.update({4: "파손", 5: "방화"})   # per_clip이 같은 사전을 읽는다
NAMES = {0: "정상", 1: "전도", 2: "유기", 3: "절도", 4: "파손", 5: "방화"}
TARGET = {1: 0.95}
DEFAULT_TARGET = 0.90
BOOT = 300
KEYS = ("clip", "start", "label", "cls", "behavior", "logits")


def load(folder: str, folds: list[str]) -> dict[str, np.ndarray]:
    parts = []
    for f in sorted(glob.glob(str(APP_ROOT / "output" / folder / "oof_fold*.npz"))):
        k = Path(f).stem.replace("oof_fold", "")
        if not folds or k in folds:
            z = np.load(f)
            parts.append({**{key: z[key] for key in KEYS}, "fold": np.full(len(z["clip"]), int(k))})
    return {k: np.concatenate([p[k] for p in parts]) for k in (*KEYS, "fold")}


def restrict(d: dict[str, np.ndarray], keep_keys: set[tuple]) -> dict[str, np.ndarray]:
    m = np.array([(c, int(s)) in keep_keys for c, s in zip(d["clip"], d["start"])])
    return {k: v[m] for k, v in d.items()}


def intended_normals() -> set[tuple[str, int]]:
    """검수에서 통과한 정상 창(서기·걷기·빈 배경). 검수 파일이 없으면 예외."""
    files = [f for f in sorted((APP_ROOT / "output").glob("normal_audit_claude_*.csv")) if f.stem != "normal_audit_claude"]
    if not files:
        raise FileNotFoundError("EVAL_NORMAL=audit인데 output/normal_audit_claude_<장소>.csv가 없다")
    return {(r["clip"], int(r["start"])) for f in files for r in csv.DictReader(open(f, encoding="utf-8")) if r["verdict"] == "pass"}


def limit_normals(d: dict[str, np.ndarray], keep: set[tuple[str, int]]) -> dict[str, np.ndarray]:
    """오경보를 세는 정상 창을 '의도한 정상'(검수 통과 창 + 매장이동)으로 제한한다. 그 밖의 사건 영상 정상 창은 애매(-1)로 빼서 양쪽 모델에 똑같이 적용한다. 양성과 다른 행동 사건 창은 그대로."""
    out = {k: v.copy() for k, v in d.items()}
    drop = np.array([lab == 0 and beh != "매장이동" and (c, int(s)) not in keep for lab, beh, c, s in zip(out["label"], out["behavior"], out["clip"], out["start"])])
    out["label"][drop] = -1
    return out


def pick(clips: list[dict], cand: list[float], target: float):
    ok = [(t, r) for t, r in ((t, evaluate(clips, t)) for t in cand) if r["recall"] >= target]
    return min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"])) if ok else (None, None)


def split_clips(d: dict[str, np.ndarray], c: int, mask: np.ndarray) -> list[dict]:
    return per_clip({k: v[mask] for k, v in d.items()}, c)


def nested(d: dict[str, np.ndarray], c: int, target: float) -> tuple[float, float, int]:
    """보류 fold를 뺀 나머지로 임계값을 정해 그 fold에서 평가한 뒤 합친 (재현율, 오경보/시간, 평가한 사건 수)."""
    hits = events = false = hours = 0.0
    for k in sorted(set(d["fold"].tolist())):
        others = d["fold"] != k
        valid = others & (d["label"] != -1)
        t, _ = pick(split_clips(d, c, others), candidates_from_values(log_odds(d["logits"], c)[valid], 100), target)
        if t is None:
            continue
        r = evaluate(split_clips(d, c, d["fold"] == k), t)
        hits += r["recall"] * r["events"]; events += r["events"]; false += r["false"]; hours += r["hours"]
    return hits / max(events, 1), false / max(hours, 1e-9), int(events)


def bootstrap(clips: list[dict], thr: float, rng: np.random.Generator) -> tuple[tuple[float, float], tuple[float, float]]:
    rec, fa = [], []
    for _ in range(BOOT):
        r = evaluate([clips[i] for i in rng.integers(0, len(clips), len(clips))], thr)
        rec.append(r["recall"]); fa.append(r["fa_per_hour"])
    return tuple(np.percentile(rec, [2.5, 97.5])), tuple(np.percentile(fa, [2.5, 97.5]))


def class_report(d: dict[str, np.ndarray], c: int, label: str, rng: np.random.Generator) -> None:
    target = TARGET.get(c, DEFAULT_TARGET)
    valid = d["label"] != -1
    clips = per_clip(d, c)
    t, r = pick(clips, candidates_from_values(log_odds(d["logits"], c)[valid], 120), target)
    if t is None:
        print(f"  {label:6s} | 목표 재현율 {target}를 만족하는 임계값 없음"); return
    (rl, rh), (fl, fh) = bootstrap(clips, t, rng)
    n_rec, n_fa, n_ev = nested(d, c, target)
    walk = set(d["clip"][d["behavior"] == "매장이동"].tolist())
    wc = [cl for cl in clips if cl["clip"] in walk]
    w_runs, w_hours = sum(len(runs(cl["z"], t)) for cl in wc), sum(len(cl["y"]) for cl in wc) * STRIDE_SEC / 3600
    print(f"  {label:6s} | OOF 임계 {t:5.2f}: 재현율 {r['recall']:.3f} [{rl:.2f}~{rh:.2f}] 오경보 {r['fa_per_hour']:.2f}/시간 [{fl:.2f}~{fh:.2f}] "
          f"| 중첩: 재현율 {n_rec:.3f} 오경보 {n_fa:.2f}/시간 | 사건 {r['events']}개, 음성 {r['hours']:.0f}시간 | MVP(매장이동) 오경보 {w_runs}건/{w_hours:.1f}시간")


def confusion(d: dict[str, np.ndarray]) -> None:
    valid = d["label"] != -1
    true, pred = d["cls"][valid], d["logits"][valid].argmax(1)
    n = len(NAMES)
    cm = np.zeros((n, n), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    print("\n창 단위 혼동행렬 (행=정답, 열=예측, 애매 창 제외, 양성·음성 모두 argmax)")
    print("        " + " ".join(f"{NAMES[j]:>6s}" for j in range(n)))
    for i in range(n):
        print(f"  {NAMES[i]:5s} " + " ".join(f"{cm[i, j]:6d}" for j in range(n)))
    print("클래스별 창 단위 precision / recall / F1")
    for i in range(1, n):
        tp = cm[i, i]; p = tp / max(cm[:, i].sum(), 1); rc = tp / max(cm[i].sum(), 1)
        print(f"  {NAMES[i]:5s} P {p:.3f}  R {rc:.3f}  F1 {2 * p * rc / max(p + rc, 1e-9):.3f}  (정답 창 {cm[i].sum()}개)")


def main() -> None:
    folds = [x for x in os.getenv("EVAL_FOLDS", "").split(",") if x]
    primary = os.getenv("EVAL_DIR6", "finetune_s3d6")
    d = load(primary, folds)
    keep = intended_normals() if os.getenv("EVAL_NORMAL") == "audit" else None
    if keep is not None:
        d = limit_normals(d, keep)
        print(f"EVAL_NORMAL=audit: 오경보를 세는 정상 = 검수 통과 창 {len(keep)}개 중 평가 fold에 있는 것 + 매장이동. 나머지 사건 영상 정상 창은 제외")
    rng = np.random.default_rng(0)
    print(f"[{primary}] 6클래스 A OOF: fold {sorted(set(d['fold'].tolist()))}, 창 {len(d['clip'])}개, 영상 {len(set(d['clip']))}개, 애매 창 {int((d['label'] == -1).sum())}개 제외")
    print("\n[6클래스 A] 이벤트 단위 (대괄호 = 클립 부트스트랩 95% 구간)")
    for c in range(1, 6):
        class_report(d, c, NAMES[c], rng)
    confusion(d)
    other = os.getenv("COMPARE_DIR6")
    if other:
        d2 = load(other, folds)
        if keep is not None:
            d2 = limit_normals(d2, keep)
        same = {(c, int(s)) for c, s in zip(d["clip"], d["start"])} & {(c, int(s)) for c, s in zip(d2["clip"], d2["start"])}
        a, b = restrict(d, same), restrict(d2, same)
        print(f"\n[정상 구성 비교] 같은 창 {len(a['clip'])}개에서 {primary}(위) 대 {other}(아래), 채택 기준: 오경보 신뢰구간이 겹치지 않을 때만")
        for c in range(1, 6):
            class_report(a, c, f"{NAMES[c]}(기준)", rng)
            class_report(b, c, f"{NAMES[c]}(새)", rng)
    old = load("finetune_s3d", folds)
    keys = {(c, int(s)) for c, s in zip(d["clip"], d["start"])}
    old = restrict(old, keys)
    d_same = restrict(d, {(c, int(s)) for c, s in zip(old["clip"], old["start"])})
    if len(old["clip"]):
        print(f"\n[기준선] 같은 창 {len(old['clip'])}개에서 비교: 4클래스 A(위)와 6클래스 A(아래)")
        for c in (1, 2, 3):
            class_report(old, c, f"{NAMES[c]}(4)", rng)
            class_report(d_same, c, f"{NAMES[c]}(6)", rng)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
