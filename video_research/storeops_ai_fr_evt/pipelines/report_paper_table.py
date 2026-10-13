"""공식 validation 결과를 논문 형식 표로 정리한다(이미 만든 로짓을 다시 집계할 뿐, 모델·임계값·스택은 바꾸지 않는다).

- 이벤트 단위(문서 02 필수 지표): Event Precision / Recall / F1, False Alarm / Hour. 경보 구간(연속 경보 창) 단위 TP/FP, 클립 단위 재현율.
- 창 단위(문서 02 보조 지표): 클래스별 Precision / Recall / F1, AUROC(그 행동 vs 나머지), 평균(macro)과 균형 정확도.
- 95% 신뢰구간: 영상(클립) 단위 부트스트랩(BOOT회 복원추출, 같은 영상의 창은 함께 뽑는다). AUROC는 BOOT_AUC회.
출력: output/official_val/paper_table.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pipelines.eval_finetune import CLASSES, STRIDE_SEC, runs
from pipelines.eval_official_val import OUT, SERVICE_KEY, STACK, auc, load_aligned
from pipelines.export_fusion import causal_mean
from pipelines.export_stack import fused_z

BOOT, BOOT_AUC, SEED = 1000, 200, 0


def clip_stats(d: dict, z: dict[int, np.ndarray], thr: dict[int, float]) -> list[dict]:
    """영상별 요약: 창 단위 혼동행렬(4x4), 클래스별 이벤트 TP/FP 구간 수, 클립 적중 여부, 음성 시간."""
    out = []
    for clip in np.unique(d["clip"]):
        m = np.flatnonzero(d["clip"] == clip)
        m = m[np.argsort(d["start"][m])]
        lab, cls = d["label"][m], d["cls"][m]
        valid = lab != -1
        zs = {c: z[c][m] for c in CLASSES}
        over = np.stack([np.zeros(len(m), bool)] + [zs[c] >= thr[c] for c in CLASSES], axis=1)
        zz = np.stack([np.full(len(m), -np.inf)] + [zs[c] for c in CLASSES], axis=1)
        pred = np.where(over.any(1), np.argmax(np.where(over, zz, -np.inf), axis=1), 0)
        cm = np.zeros((4, 4))
        np.add.at(cm, (cls[valid], pred[valid]), 1)
        ev = {}
        beh = d["behavior"][m][0]
        for c, name in CLASSES.items():
            y = np.where(lab == 1, 1 if beh == name else 0, lab)      # 엄격: 다른 행동 사건 구간의 경보도 오경보
            r = runs(zs[c], thr[c])
            tp = sum(1 for a, b in r if (y[a:b + 1] == 1).any())
            fp = sum(1 for a, b in r if not (y[a:b + 1] != 0).any())
            ev[c] = {"tp": tp, "fp": fp, "event": int((y == 1).any()), "hit": int(tp > 0), "hours": int((y == 0).sum()) * STRIDE_SEC / 3600}
        out.append({"cm": cm, "ev": ev, "idx": m})
    return out


def metrics(stats: list[dict], pick: np.ndarray) -> dict:
    cm = sum(stats[i]["cm"] for i in pick)
    res = {"cm": cm}
    f1s, recs = [], []
    for k in range(4):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k].sum() - cm[k, k]
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        res[f"w{k}"] = (p, r, f)
        f1s.append(f), recs.append(r)
    res["macro_f1"], res["bal_acc"], res["acc"] = float(np.mean(f1s)), float(np.mean(recs)), float(np.trace(cm) / cm.sum())
    for c in CLASSES:
        tp = sum(stats[i]["ev"][c]["tp"] for i in pick)
        fp = sum(stats[i]["ev"][c]["fp"] for i in pick)
        events = sum(stats[i]["ev"][c]["event"] for i in pick)
        hits = sum(stats[i]["ev"][c]["hit"] for i in pick)
        hours = sum(stats[i]["ev"][c]["hours"] for i in pick)
        p, r = tp / max(tp + fp, 1), hits / max(events, 1)
        res[f"e{c}"] = (p, r, 2 * p * r / max(p + r, 1e-9), fp / max(hours, 1e-9))
    return res


def ci(vals: list[float]) -> str:
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def main() -> None:
    d = load_aligned()
    spec = json.loads(STACK.read_text(encoding="utf-8"))["classes"]
    x = np.hstack([d["la"], d["lc"]]).astype(np.float64)
    z = {c: causal_mean(d, fused_z(spec[SERVICE_KEY[c]], x), int(spec[SERVICE_KEY[c]].get("smooth", 1))) for c in CLASSES}
    thr = {c: spec[SERVICE_KEY[c]]["threshold_logit"] for c in CLASSES}
    stats = clip_stats(d, z, thr)
    point = metrics(stats, np.arange(len(stats)))
    rng = np.random.default_rng(SEED)
    boots = [metrics(stats, rng.integers(0, len(stats), len(stats))) for _ in range(BOOT)]
    valid, y = d["label"] != -1, d["cls"]
    aucs, auc_ci = {}, {}
    clip_idx = {c: np.flatnonzero(d["clip"] == c) for c in np.unique(d["clip"])}
    names = list(clip_idx)
    for c in CLASSES:
        aucs[c] = auc(z[c][valid], (y[valid] == c))
        vals = []
        for _ in range(BOOT_AUC):
            pick = np.concatenate([clip_idx[names[i]] for i in rng.integers(0, len(names), len(names))])
            v = pick[d["label"][pick] != -1]
            if (y[v] == c).any() and (y[v] != c).any():
                vals.append(auc(z[c][v], y[v] == c))
        auc_ci[c] = ci(vals)
    n_clip = {b: len(np.unique(d["clip"][d["behavior"] == b])) for b in ("전도", "유기", "절도", "매장이동")}
    L = ["# 공식 validation 결과 (논문 형식)\n",
         f"평가 대상: 공식 validation {len(stats)}개 영상(전도 {n_clip['전도']}, 유기 {n_clip['유기']}, 절도 {n_clip['절도']}, 매장이동(정상) {n_clip['매장이동']}), 창 {int((d['label'] != -1).sum())}개(애매 창 제외). "
         "모델·스택·임계값은 개발용 날짜 5-fold OOF에서 정해 동결했고 공식 validation은 이 평가에서 1회만 사용했다. 95% 신뢰구간은 영상 단위 부트스트랩(1000회).\n",
         "## Table 1. 이벤트 단위\n", "| 행동 | Event Precision | Event Recall | Event F1 | False Alarm / Hour |", "|---|---|---|---|---|"]
    for c, name in CLASSES.items():
        p, r, f, fa = point[f"e{c}"]
        L.append(f"| {name} | {p:.3f} {ci([b[f'e{c}'][0] for b in boots])} | {r:.3f} {ci([b[f'e{c}'][1] for b in boots])} | {f:.3f} {ci([b[f'e{c}'][2] for b in boots])} | {fa:.2f} {ci([b[f'e{c}'][3] for b in boots])} |")
    L += ["\nEvent Precision = 사건과 겹친 경보 구간 / 전체 경보 구간, Event Recall = 경보가 사건을 한 번이라도 잡은 영상 비율, 경보 구간 = 연속 경보 창. 정상이 매장이동 영상뿐이라 False Alarm / Hour는 쉬운 조건의 값이다.\n",
          "## Table 2. 창 단위 (4초 창, 임계값 적용)\n", "| 클래스 | Precision | Recall | F1 | AUROC |", "|---|---|---|---|---|"]
    for k, name in enumerate(["정상"] + list(CLASSES.values())):
        p, r, f = point[f"w{k}"]
        a = f"{aucs[k]:.3f} {auc_ci[k]}" if k else "-"
        L.append(f"| {name} | {p:.3f} {ci([b[f'w{k}'][0] for b in boots])} | {r:.3f} {ci([b[f'w{k}'][1] for b in boots])} | {f:.3f} {ci([b[f'w{k}'][2] for b in boots])} | {a} |")
    L += [f"| **평균(macro)** | - | **{point['bal_acc']:.3f}** {ci([b['bal_acc'] for b in boots])} | **{point['macro_f1']:.3f}** {ci([b['macro_f1'] for b in boots])} | **{np.mean(list(aucs.values())):.3f}** |",
          f"\n정확도 {point['acc']:.3f}(정상 비중 90%라 참고용), 평균 Recall = 균형 정확도.\n",
          "## Table 3. 혼동행렬 (행=정답, 열=예측)\n", "| 정답＼예측 | 정상 | 전도 | 유기 | 절도 |", "|---|---|---|---|---|"]
    for k, name in enumerate(["정상", "전도", "유기", "절도"]):
        L.append(f"| {name} | " + " | ".join(f"{int(v)}" for v in point["cm"][k]) + " |")
    (OUT / "paper_table.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
