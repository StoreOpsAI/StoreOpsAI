"""train / 개발 검증(OOF) / 공식 validation 성능을 같은 지표·같은 임계값으로 비교해 train-validation 차이를 기록한다(AGENTS.md).

- train: 최종 A·C1이 학습한 영상 전체를 채점한 값(재대입 성능; 낙관적이다). output/train_eval/logits_{full,c1}.npz(score_train_final.py).
- OOF: 날짜 5-fold에서 그 fold를 학습하지 않은 모델의 점수 + 날짜 fold CV 스택(개발용 held-out). 임계값 선정에 쓴 데이터다.
- val: 공식 validation(1회 평가, 튜닝 미사용).
임계값·스택 계수는 모두 stack.json(동결)이며 여기서 바꾸지 않는다. 이벤트 단위(엄격)와 창 단위 평균 F1을 같은 코드(report_paper_table)로 계산한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pipelines.compare_backbones import _fit_linear
from pipelines.ensemble_eval import merged
from pipelines.eval_finetune import CLASSES, runs
from pipelines.eval_official_val import OUT as VAL_OUT, SERVICE_KEY, STACK, load_aligned
from pipelines.export_fusion import causal_mean
from pipelines.export_stack import fused_z
from pipelines.report_paper_table import clip_stats, metrics

TRAIN_DIR = VAL_OUT.parent / "train_eval"


def train_data() -> dict[str, np.ndarray]:
    a, c = np.load(TRAIN_DIR / "logits_full.npz"), np.load(TRAIN_DIR / "logits_c1.npz")
    key_c = {(str(k), int(s)): i for i, (k, s) in enumerate(zip(c["clip"], c["start"]))}
    ia, ic = zip(*[(i, key_c[(str(k), int(s))]) for i, (k, s) in enumerate(zip(a["clip"], a["start"])) if (str(k), int(s)) in key_c])
    d = {k: a[k][list(ia)] for k in ("clip", "start", "label", "cls", "behavior")}
    d["la"], d["lc"] = a["logits"][list(ia)], c["logits"][list(ic)]
    return d


def oof_z(d: dict, spec: dict) -> dict[int, np.ndarray]:
    x = np.hstack([d["la"], d["lc"]]).astype(np.float32)
    valid, folds = d["label"] != -1, sorted(set(d["fold"].tolist()))
    z = {}
    for c in CLASSES:
        y = np.where(d["cls"] == c, 1, 0).astype(np.int64)
        s = np.zeros(len(y))
        for k in folds:
            tr, te = valid & (d["fold"] != k), d["fold"] == k
            p = _fit_linear(x[tr], y[tr])(x[te])
            s[te] = np.log(p + 1e-6) - np.log(1 - p + 1e-6)
        z[c] = s
    return z


def summarize(d: dict, z: dict[int, np.ndarray], thr: dict[int, float]) -> dict:
    stats = clip_stats(d, z, thr)
    return metrics(stats, np.arange(len(stats)))


def normal_view(d: dict, z: dict[int, np.ndarray], thr: dict[int, float]) -> dict[int, tuple]:
    """정상 = 걷기·서 있기(매장이동 영상 전체): 오경보는 이 영상들의 경보 구간만, 재현율은 사건 영상 전체. 반환 {행동: (recall, 오경보 구간 수, 오경보/시간, precision, F1)}."""
    normal = np.unique(d["clip"][d["behavior"] == "매장이동"])
    hours = sum(int((d["clip"] == c).sum()) for c in normal) * 2 / 3600
    out = {}
    for c in CLASSES:
        fa = sum(len(runs(z[c][m][np.argsort(d["start"][m])], thr[c])) for m in (np.flatnonzero(d["clip"] == k) for k in normal))
        tp = hit = ev = 0
        for k in np.unique(d["clip"][d["cls"] == c]):
            m = np.flatnonzero(d["clip"] == k)
            m = m[np.argsort(d["start"][m])]
            t = sum(1 for a, b in runs(z[c][m], thr[c]) if (d["label"][m][a:b + 1] == 1).any())
            ev, tp, hit = ev + 1, tp + t, hit + (t > 0)
        rec, prec = hit / max(ev, 1), tp / max(tp + fa, 1)
        out[c] = (rec, fa, fa / max(hours, 1e-9), prec, 2 * prec * rec / max(prec + rec, 1e-9))
    return out


def main() -> None:
    spec = json.loads(STACK.read_text(encoding="utf-8"))["classes"]
    thr = {c: spec[SERVICE_KEY[c]]["threshold_logit"] for c in CLASSES}
    deployed = lambda d: {c: causal_mean(d, fused_z(spec[SERVICE_KEY[c]], np.hstack([d["la"], d["lc"]]).astype(np.float64)), int(spec[SERVICE_KEY[c]].get("smooth", 1))) for c in CLASSES}
    dt, dv, do = train_data(), load_aligned(), merged()
    res = {"train(재대입)": summarize(dt, deployed(dt), thr), "OOF(개발 held-out)": summarize(do, deployed(do), thr), "공식 validation": summarize(dv, deployed(dv), thr)}
    L = ["# train - validation 차이 (같은 지표·같은 임계값)\n",
         "train = 최종 A·C1이 학습한 영상(재대입, 낙관적) / OOF = 날짜 5-fold held-out(개발, 임계값 선정에 사용) / 공식 validation = 1회 평가. 임계값(로그오즈): " + ", ".join(f"{CLASSES[c]} {thr[c]:.2f}" for c in CLASSES) + "\n",
         "## 이벤트 단위(엄격)  Precision / Recall / F1 / 오경보(시간당)\n", "| 행동 | " + " | ".join(res) + " | train-val F1 차이 | OOF-val F1 차이 |", "|---|" + "---|" * (len(res) + 2)]
    for c, name in CLASSES.items():
        cells = []
        for s in res.values():
            p, r, f, fa = s[f"e{c}"]
            cells.append(f"{p:.3f} / {r:.3f} / {f:.3f} / {fa:.2f}")
        ft, fo, fv = (res[k][f"e{c}"][2] for k in res)
        L.append(f"| {name} | " + " | ".join(cells) + f" | {ft - fv:+.3f} | {fo - fv:+.3f} |")
    mean = {k: np.mean([s[f"e{c}"][2] for c in CLASSES]) for k, s in res.items()}
    L.append("| **평균 Event F1** | " + " | ".join(f"{v:.3f}" for v in mean.values()) + f" | {list(mean.values())[0] - list(mean.values())[2]:+.3f} | {list(mean.values())[1] - list(mean.values())[2]:+.3f} |")
    L += ["\n## 창 단위  Precision / Recall / F1 (클래스별), 평균 F1, 균형 정확도\n", "| 구분 | 정상 | 전도 | 유기 | 절도 | 평균 F1 | 균형 정확도 |", "|---|---|---|---|---|---|---|"]
    for k, s in res.items():
        L.append(f"| {k} | " + " | ".join("{:.3f} / {:.3f} / {:.3f}".format(*s[f"w{j}"]) for j in range(4)) + f" | {s['macro_f1']:.3f} | {s['bal_acc']:.3f} |")
    zs = {"train(재대입)": (dt, deployed(dt)), "OOF(개발 held-out)": (do, deployed(do)), "공식 validation": (dv, deployed(dv))}
    nv = {k: normal_view(d, z, thr) for k, (d, z) in zs.items()}
    L += ["\n## 정상 = 걷기·서 있기(매장이동 영상 전체)로 오경보를 센 기준: Recall / 오경보 수(시간당) / Precision / F1\n", "| 행동 | " + " | ".join(zs) + " | train-val F1 차이 | OOF-val F1 차이 |", "|---|" + "---|" * (len(zs) + 2)]
    for c, name in CLASSES.items():
        cells = [f"{v[c][0]:.3f} / {v[c][1]} ({v[c][2]:.2f}) / {v[c][3]:.3f} / {v[c][4]:.3f}" for v in nv.values()]
        ft, fo, fv = (v[c][4] for v in nv.values())
        L.append(f"| {name} | " + " | ".join(cells) + f" | {ft - fv:+.3f} | {fo - fv:+.3f} |")
    m3 = [np.mean([v[c][4] for c in CLASSES]) for v in nv.values()]
    L.append("| **평균 F1** | " + " | ".join(f"{x:.3f}" for x in m3) + f" | {m3[0] - m3[2]:+.3f} | {m3[1] - m3[2]:+.3f} |")
    L.append("\n오경보는 데이터마다 정상 영상 구성이 달라 단순 비교가 어렵다(train·OOF는 반품 등 어려운 정상 포함, 공식 validation 정상은 매장이동뿐).")
    (VAL_OUT / "train_val_gap.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
