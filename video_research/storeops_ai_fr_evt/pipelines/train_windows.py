"""캐시한 시간 창 특징(포즈·S3D·X-CLIP)으로 행동 헤드(MLP)를 학습·평가한다. 비교 조건:
포즈 / S3D / X-CLIP / 셋을 이어 붙인 것(concat) / 소스별 점수 평균(후반 결합).
분할 두 가지: 세션 fold(저장된 fold, 5-fold)와 매장 단위(사이트 하나씩 빼기). 같은 영상의 창은 항상 같은 fold(AGENTS.md 규칙 1·2·6).
라벨 -1(전환 구간)은 학습·평가에서 뺀다. 공식 validation은 쓰지 않는다. 결과: output/windows_<행동>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pipelines.relation_sequence import window_sequences
from config.config import (BEHAVIOR_FEATURE_ROOT, WIN_MLP_DROPOUT, WIN_MLP_EPOCHS, WIN_MLP_HIDDEN, WIN_MLP_LR,
                           WIN_MLP_SEEDS, WIN_MLP_WEIGHT_DECAY)

SOURCES = ("pose", "s3d", "xclip")  # "+entry" 조건은 같은 추적의 첫 창 특징(<소스>_e)을 덧붙인다


def average_precision_score(y, score):
    """AP = 양성이 나올 때마다의 정밀도 평균(점수 내림차순)."""
    order = np.argsort(-score, kind="stable")
    hit = y[order] == 1
    return float((np.cumsum(hit)[hit] / (np.flatnonzero(hit) + 1)).mean())


def roc_auc_score(y, score):
    """AUC = 양성이 음성보다 점수가 높을 확률(동점은 반반), 순위 합으로 계산."""
    ranks = np.empty(len(score))
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    start = 0
    for end in range(1, len(score) + 1):
        if end == len(score) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def precision_recall_fscore_support(y, pred, average="binary", zero_division=0):
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), None


def _metrics_self_check():
    y, s = np.array([1, 0, 1, 0, 0]), np.array([0.9, 0.8, 0.7, 0.3, 0.1])
    assert abs(average_precision_score(y, s) - (1 + 2 / 3) / 2) < 1e-9
    assert abs(roc_auc_score(y, s) - 5 / 6) < 1e-9
    assert roc_auc_score(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5


def load_windows(folder: Path, rel_dir: Path | None = None, cache_dir: Path | None = None):
    """label이 0/1인 창만. 반환: {소스: (N, D)}, label, fold, site, clip.
    rel_dir이 있으면 같은 영상의 사람-물건 연결 특징(rel)을 (tid, start)로 맞춰 붙인다(없는 영상·창은 뺀다)."""
    parts = {s: [] for s in [*SOURCES, *(["rel"] if rel_dir else []), *(["relseq"] if cache_dir else [])]}
    entry = {s: [] for s in SOURCES}
    meta = {"label": [], "fold": [], "site": [], "clip": []}
    for f in sorted(folder.glob("*.npz")):
        if ".tmp" in f.name:
            continue
        d = np.load(f)
        if int(d["n"]) == 0:
            continue
        keep = d["label"] != -1
        if rel_dir:
            if not (rel_dir / f.name).is_file():
                continue
            r = np.load(rel_dir / f.name)
            index = {(int(t), int(s)): i for i, (t, s) in enumerate(zip(r["tid"], r["start"]))}
            rows = np.array([index.get((int(t), int(s)), -1) for t, s in zip(d["tid"], d["start"])])
            keep &= rows >= 0
            parts["rel"].append(r["rel"][rows[keep]].astype(np.float32))
        if cache_dir:
            if not (cache_dir / f.name).is_file():
                continue
            cache = np.load(cache_dir / f.name)
            win_seq, entry_seq = window_sequences(cache)
            index = {(int(t), int(st)): i for i, (t, st) in enumerate(zip(cache["win_tid"], cache["win_start"]))}
            rows = np.array([index.get((int(t), int(st)), -1) for t, st in zip(d["tid"], d["start"])])
            keep &= rows >= 0
            seq = np.concatenate([win_seq.reshape(len(win_seq), -1), entry_seq.reshape(len(entry_seq), -1)], axis=1)
            parts["relseq"].append(seq[rows[keep]])
        first = {int(t): int(np.flatnonzero((d["tid"] == t) & (d["start"] == d["start"][d["tid"] == t].min()))[0]) for t in set(d["tid"])}
        entry_idx = np.array([first[int(t)] for t in d["tid"]])[keep]  # 같은 추적의 첫 창(자기 자신이 첫 창이면 자기 자신)
        for s in SOURCES:
            flat = d[s].reshape(len(d["label"]), -1).astype(np.float32)
            parts[s].append(flat[keep])
            entry[s].append(flat[entry_idx])
        n = int(keep.sum())
        meta["label"].append(d["label"][keep])
        meta["fold"] += [int(d["fold"])] * n
        meta["site"] += [str(d["clip"]).split("_BU_")[1].split("_")[0]] * n
        meta["clip"] += [str(d["clip"])] * n
    x = {s: np.concatenate(v) for s, v in parts.items()}
    x.update({f"{s}_e": np.concatenate(v) for s, v in entry.items()})
    return x, np.concatenate(meta["label"]).astype(np.float32), np.array(meta["fold"]), np.array(meta["site"]), np.array(meta["clip"])


def fit_predict(x_tr, y_tr, x_te_list, seed):
    """표준화 + MLP(은닉 1층). x_te_list의 각 배열에 대한 점수를 돌려준다. GPU가 있으면 GPU로 학습한다."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = x_tr.mean(0), x_tr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(x_tr.shape[1], WIN_MLP_HIDDEN), nn.ReLU(), nn.Dropout(WIN_MLP_DROPOUT), nn.Linear(WIN_MLP_HIDDEN, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=WIN_MLP_LR, weight_decay=WIN_MLP_WEIGHT_DECAY)
    xt, yt = torch.from_numpy((x_tr - mean) / std).to(dev), torch.from_numpy(y_tr)[:, None].to(dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(WIN_MLP_EPOCHS):
        net.train()
        for idx in torch.randperm(len(xt), generator=g).to(dev).split(256):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return [torch.sigmoid(net(torch.from_numpy((x - mean) / std).to(dev)))[:, 0].cpu().numpy() for x in x_te_list]


def build_conds(has_rel: bool, with_entry: bool, has_seq: bool = False) -> dict[str, list[str]]:
    conds = {**{s: [s] for s in SOURCES}, "concat": list(SOURCES)}
    if with_entry:
        conds.update({f"{s}+entry": [s, f"{s}_e"] for s in SOURCES})
        conds["concat+entry"] = [*SOURCES, *[f"{s}_e" for s in SOURCES]]
    if has_rel:
        conds.update({"rel": ["rel"], "pose+rel": ["pose", "rel"], "concat+rel": [*SOURCES, "rel"]})
    if has_seq:
        conds.update({"relseq": ["relseq"], "pose+relseq": ["pose", "relseq"], "concat+relseq": [*SOURCES, "relseq"]})
    return conds


def learn_threshold(y, score):
    """학습으로 정하는 판정 기준: 학습에 안 쓴(out-of-fold) 점수에서 F1이 최대인 점수. 기준 후보는 점수 자체(균등 분위수)."""
    cand = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    f1 = [precision_recall_fscore_support(y, score >= t, average="binary", zero_division=0)[2] for t in cand]
    return float(cand[int(np.argmax(f1))])


def oof_scores(feats, y, groups, seed):
    """groups 단위 4-fold로 학습에 안 쓴 영상의 점수를 만든다(기준 학습용). 같은 그룹은 한 fold에 둔다."""
    ids = sorted(set(groups))
    out = np.zeros(len(y))
    for k in range(4):
        held = np.isin(groups, ids[k::4])
        out[held] = fit_predict(feats[~held], y[~held], [feats[held]], seed)[0]
    return out


def run_split(x, y, groups, seeds, conds):
    """groups(fold 또는 사이트)로 하나씩 빼며 조건별 점수와, 학습한 기준으로 낸 판정을 모은다."""
    scores = {c: [np.zeros(len(y)) for _ in seeds] for c in [*conds, "late_avg"]}
    decisions = {c: [np.zeros(len(y), bool) for _ in seeds] for c in conds}
    thresholds = {c: [[] for _ in seeds] for c in conds}
    train_ap = {c: [[] for _ in seeds] for c in conds}
    for g in sorted(set(groups)):
        te, tr = groups == g, groups != g
        for si, seed in enumerate(seeds):
            per_source = {}
            for c, srcs in conds.items():
                feats = np.concatenate([x[s] for s in srcs], axis=1)
                p_te, p_tr = fit_predict(feats[tr], y[tr], [feats[te], feats[tr]], seed)
                thr = learn_threshold(y[tr], oof_scores(feats[tr], y[tr], groups[tr], seed))
                scores[c][si][te], decisions[c][si][te] = p_te, p_te >= thr
                thresholds[c][si].append(thr)
                train_ap[c][si].append(average_precision_score(y[tr], p_tr))
                if c in SOURCES:
                    per_source[c] = p_te
            scores["late_avg"][si][te] = np.mean([per_source[s] for s in SOURCES], axis=0)
    return scores, train_ap, decisions, thresholds


def summarize(y, scores, train_ap, subsets=None, decisions=None, thresholds=None):
    """subsets = {이름: 불리언 마스크}: 그 부분 창만으로 낸 AP도 같이(예: 탐지기가 학습에 안 쓴 fold).
    decisions/thresholds가 있으면 학습한 기준으로 낸 P/R/F1(_learned)과 기준 평균(thr_mean)도 같이."""
    out = {}
    for c, runs in scores.items():
        aps = [average_precision_score(y, p) for p in runs]
        f1 = [precision_recall_fscore_support(y, p >= 0.5, average="binary", zero_division=0)[:3] for p in runs]
        out[c] = {"AP_mean": float(np.mean(aps)), "AP_std": float(np.std(aps)),
                  "AUC": float(np.mean([roc_auc_score(y, p) for p in runs])),
                  "P@0.5": float(np.mean([a[0] for a in f1])), "R@0.5": float(np.mean([a[1] for a in f1])),
                  "F1@0.5": float(np.mean([a[2] for a in f1]))}
        if decisions and c in decisions:
            learned = [precision_recall_fscore_support(y, d, average="binary", zero_division=0)[:3] for d in decisions[c]]
            out[c].update({"P_learned": float(np.mean([a[0] for a in learned])), "R_learned": float(np.mean([a[1] for a in learned])),
                           "F1_learned": float(np.mean([a[2] for a in learned])),
                           "thr_mean": float(np.mean([np.mean(v) for v in thresholds[c]]))})
        for name, mask in (subsets or {}).items():
            out[c][name] = float(np.mean([average_precision_score(y[mask], p[mask]) for p in runs]))
        if c in train_ap:
            out[c]["train_AP"] = float(np.mean([np.mean(v) for v in train_ap[c]]))
    return out


def main():
    _metrics_self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(Path(BEHAVIOR_FEATURE_ROOT) / "abandon_v1"))
    parser.add_argument("--rel-dir", default="", help="사람-물건 연결 특징 폴더(abandon_relations 결과). 주면 rel 조건을 추가")
    parser.add_argument("--cache-dir", default="", help="abandon_cache 결과 폴더. 주면 임계값 없는 관계 시계열(relseq) 조건을 추가")
    parser.add_argument("--entry", action="store_true", help="등장 창 특징 조건도 포함")
    parser.add_argument("--name", default="abandon")
    args = parser.parse_args()
    x, y, fold, site, _ = load_windows(Path(args.features), Path(args.rel_dir) if args.rel_dir else None,
                                         Path(args.cache_dir) if args.cache_dir else None)
    conds = build_conds(bool(args.rel_dir), args.entry, bool(args.cache_dir))
    print(f"창 {len(y)}개 (양성 {int(y.sum())}), 사이트 {sorted(set(site))}, 조건 {list(conds)}", flush=True)
    result = {"windows": int(len(y)), "positives": int(y.sum()), "conditions": list(conds)}
    for split, groups, subsets in (("session_fold", fold, {"AP_fold0": fold == 0}), ("leave_one_site_out", site, None)):
        scores, train_ap, decisions, thresholds = run_split(x, y, groups, WIN_MLP_SEEDS, conds)
        result[split] = summarize(y, scores, train_ap, subsets, decisions, thresholds)
        print(split, json.dumps({c: {k: round(v, 3) for k, v in m.items()} for c, m in result[split].items()}, ensure_ascii=False), flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output", f"windows_{args.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
