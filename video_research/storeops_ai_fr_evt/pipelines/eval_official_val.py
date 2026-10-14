"""공식 validation 최종 평가(1회): 최종 A + 최종 C1 로짓 -> 서비스와 같은 스택(stack.json) -> 이진 경보(재현율>=0.9 OOF 임계값).

지표: 이벤트 단위(클립 재현율, 시간당 오경보, 탐지 지연)와 창 단위(클래스별 precision/recall/F1, AUC, 혼동행렬). 임계값·계수는 stack.json 그대로이며 여기서 조정하지 않는다.
이미 report가 있으면 FORCE_REEVAL=1이 아니면 거부한다. 입력: score_official_val.py의 logits_full.npz, logits_c1.npz.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from pipelines.eval_finetune import CLASSES, evaluate
from pipelines.fusion_eval import clips_for
from pipelines.export_fusion import causal_mean
from pipelines.export_stack import fused_z
from config.config import SERVICE_STACK_DIR

OUT = Path(__file__).resolve().parents[1] / "output" / "official_val"
STACK = SERVICE_STACK_DIR / "fusion.json"
SERVICE_KEY = {1: "쓰러짐", 2: "쓰레기 투기", 3: "절도"}
WIN = 12


def load_aligned() -> dict[str, np.ndarray]:
    a, c = np.load(OUT / "logits_full.npz"), np.load(OUT / "logits_c1.npz")
    key_c = {(str(k), int(s)): i for i, (k, s) in enumerate(zip(c["clip"], c["start"]))}
    ia, ic = [], []
    for i, (k, s) in enumerate(zip(a["clip"], a["start"])):
        j = key_c.get((str(k), int(s)))
        if j is not None:
            ia.append(i), ic.append(j)
    d = {k: a[k][ia] for k in ("clip", "start", "label", "cls", "behavior")}
    d["la"], d["lc"] = a["logits"][ia], c["logits"][ic]
    return d


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """순위 기반 AUC(동점은 평균 순위)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delays(d: dict, z: np.ndarray, cls: int, thr: float) -> list[float]:
    out = []
    for clip in np.unique(d["clip"][d["cls"] == cls]):
        m = np.flatnonzero(d["clip"] == clip)
        m = m[np.argsort(d["start"][m])]
        pos = m[d["label"][m] == 1]
        hit = [i for i in m if z[i] >= thr and d["label"][i] != 0 and d["start"][i] + WIN >= d["start"][pos[0]]] if len(pos) else []
        if hit:
            out.append((d["start"][hit[0]] + WIN - d["start"][pos[0]]) / 3)
    return out


def main() -> None:
    report = OUT / "report.md"
    if report.is_file() and os.getenv("FORCE_REEVAL") != "1":
        sys.exit(f"이미 평가했습니다: {report}. 공식 validation은 설정 동결 후 한 번만 평가합니다.")
    spec = json.loads(STACK.read_text(encoding="utf-8"))["classes"]
    d = load_aligned()
    x = np.hstack([d["la"], d["lc"]]).astype(np.float64)
    z = {c: causal_mean(d, fused_z(spec[SERVICE_KEY[c]], x), int(spec[SERVICE_KEY[c]].get("smooth", 1))) for c in CLASSES}   # 서비스와 같은 점수(행동별 과거 m개 창 평균)
    thr = {c: spec[SERVICE_KEY[c]]["threshold_logit"] for c in CLASSES}
    lines = [f"# 공식 validation 최종 평가\n\n창 {len(d['clip'])}개, 클립 {len(np.unique(d['clip']))}개. 임계값(로그오즈): " + ", ".join(f"{CLASSES[c]} {thr[c]:.2f}" for c in CLASSES) + "\n"]
    summary = {}
    for mode, other in (("엄격(다른 행동 사건 구간의 경보도 오경보)", "0"), ("참고(다른 행동 사건 구간은 판정 제외)", "1")):
        os.environ["IGNORE_OTHER_EVENTS"] = other
        lines.append(f"\n## 이벤트 단위 - {mode}\n\n| 행동 | 클립 재현율 | 오경보/시간 | 음성 창 경보 비율/시간 | 지연 중앙값(초) | 지연 p90 |\n|---|---|---|---|---|---|")
        for c, name in CLASSES.items():
            r = evaluate(clips_for(d, z[c], c), thr[c])
            dl = delays(d, z[c], c, thr[c])
            lines.append(f"| {name} | {r['recall']:.3f} | {r['fa_per_hour']:.2f} | {r['fa_windows_per_hour']:.1f} | {np.median(dl) if dl else float('nan'):.1f} | {np.percentile(dl, 90) if dl else float('nan'):.1f} |")
            summary[(other, name)] = {k: float(v) for k, v in r.items() if isinstance(v, (int, float, np.floating))}
    valid = d["label"] != -1
    y = d["cls"][valid]
    zz = np.stack([np.full(valid.sum(), -np.inf)] + [z[c][valid] for c in CLASSES], axis=1)
    over = np.stack([np.zeros(valid.sum(), bool)] + [z[c][valid] >= thr[c] for c in CLASSES], axis=1)
    pred = np.where(over.any(1), np.argmax(np.where(over, zz, -np.inf), axis=1), 0)
    names = ["정상"] + list(CLASSES.values())
    cm = np.zeros((4, 4), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    lines.append("\n## 창 단위(임계값 적용, 애매 창 제외)\n\n| 클래스 | precision | recall | F1 | AUC(그 행동 vs 나머지) | 양성 창 |\n|---|---|---|---|---|---|")
    for k, name in enumerate(names):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k].sum() - cm[k, k]
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        a = auc(z[k][valid], y == k) if k > 0 and (y == k).any() else float("nan")
        lines.append(f"| {name} | {pr:.3f} | {rc:.3f} | {2 * pr * rc / max(pr + rc, 1e-9):.3f} | {a:.3f} | {int(cm[k].sum())} |")
    lines.append("\n혼동행렬(행=정답, 열=예측: " + ", ".join(names) + ")\n\n```\n" + "\n".join(" ".join(f"{v:7d}" for v in row) for row in cm) + "\n```")
    lines.append("\n## 비교용: OOF(개발) 값\n\n전도 재현율 0.923 / 오경보 1.42, 유기 0.905 / 4.46, 절도 0.928 / 8.10 (재현율, 시간당 오경보). train-validation 성능 차이는 위 표와 비교한다.")
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _self_check() -> None:
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(auc(s, np.array([False, False, True, True])) - 0.75) < 1e-9   # 고전 예제: AUC 0.75
    assert abs(auc(np.array([1.0, 1.0]), np.array([True, False])) - 0.5) < 1e-9   # 동점은 0.5
    print("ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _self_check() if len(sys.argv) > 1 and sys.argv[1] == "--self-check" else main()
