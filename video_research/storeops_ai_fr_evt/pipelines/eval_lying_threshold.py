"""전도 dev 5-fold에서 점수 경보 기준(0.5 고정)을 학습으로 정한 것과 비교한다. 공식 validation·quarantine은 쓰지 않는다.

각 바깥 fold k: 학습 fold들(≠k)을 안쪽에서 한 fold씩 빼며(inner leave-one-fold-out) 학습에 안 쓴 사람 점수(out-of-fold)를 모으고,
그 점수에서 F1이 최대인 기준을 정한다(train_windows.learn_threshold). 이 기준으로 바깥 dev fold 경보를 판정한다(dev는 기준 선택에 안 쓰임).
출력: 사람 단위 P/R/F1(기준 0.5 vs 학습한 기준), 클립 적중·분당 오경보, 기준을 바꿔 가며 본 운영 곡선(바깥 dev 합산).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from config.config import LYING_TRANSITION_FRAMES
from models.fall_detector import lying_alarms
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_lying import FPS, fit_method, judge_alarms, person_scores
from pipelines.lying_manifest import FOLDS
from pipelines.lying_pose_dataset import load_dataset
from pipelines.train_windows import learn_threshold

SWEEP = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)


def prf(y, pred):
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3)}


def alarms_at(clip, scores, thr):
    by_frame = defaultdict(list)
    for f, box, s in zip(clip["frame"], clip["box"], scores):
        if s >= thr:
            by_frame[int(f)].append(tuple(map(float, box)))
    return lying_alarms([(f, by_frame.get(f, [])) for f in range(clip["frames"])], FPS)


def alarm_totals(clips, score, thr):
    """(적중 클립, 전도 클립, 영역별 오경보, 영역별 정상 분)."""
    hits = fall_clips = 0
    false_by, minutes_by = {"전도 영상 정상": 0, "매장이동 정상": 0}, {"전도 영상 정상": 0.0, "매장이동 정상": 0.0}
    for c in clips:
        hit, false, _ = judge_alarms(c, alarms_at(c, score(c), thr))
        domain = "전도 영상 정상" if c["falls"] else "매장이동 정상"
        hits, fall_clips = hits + hit, fall_clips + bool(c["falls"])
        false_by[domain] += false
        first = max(min(f["start"] for f in c["falls"]) - LYING_TRANSITION_FRAMES, 0) if c["falls"] else c["frames"]
        minutes_by[domain] += first / FPS / 60
    return hits, fall_clips, false_by, minutes_by


def inner_oof(train, folds_train, method):
    """학습 fold를 하나씩 빼며 학습에 안 쓴 사람 점수·라벨을 모은다."""
    scores, labels = [], []
    for j in sorted(set(folds_train)):
        inner_train = [c for c, f in zip(train, folds_train) if f != j]
        held = [c for c, f in zip(train, folds_train) if f == j and not c.get("aug")]
        s, y = person_scores(held, fit_method(inner_train, method))
        scores.append(s), labels.append(y)
    return np.concatenate(scores), np.concatenate(labels)


def evaluate(ds, method):
    folds = np.array([c["fold"] for c in ds])
    out = {"folds": [], "sweep": {str(t): {"hits": 0, "fall_clips": 0, "false": 0, "minutes": 0.0} for t in SWEEP}}
    pooled = {"learned": [], "fixed": []}
    totals = {"learned": [0, 0, 0, 0.0], "fixed": [0, 0, 0, 0.0]}
    for k in range(FOLDS):
        train = [c for c, f in zip(ds, folds) if f != k]
        dev = [c for c, f in zip(ds, folds) if f == k and not c.get("aug")]
        score = fit_method(train, method)
        s_oof, y_oof = inner_oof(train, [f for f in folds if f != k], method)
        thr = learn_threshold(y_oof.astype(np.float32), s_oof)
        sd, yd = person_scores(dev, score)
        out["folds"].append({"fold": k, "learned_threshold": round(thr, 3), "dev_person_learned": prf(yd, sd >= thr), "dev_person_fixed05": prf(yd, sd >= 0.5)})
        for name, t in (("learned", thr), ("fixed", 0.5)):
            h, fc, fb, mb = alarm_totals(dev, score, t)
            totals[name][0] += h
            totals[name][1] += fc
            totals[name][2] += sum(fb.values())
            totals[name][3] += sum(mb.values())
            pooled[name].append((yd, sd >= t))
        for t in SWEEP:
            h, fc, fb, mb = alarm_totals(dev, score, t)
            e = out["sweep"][str(t)]
            e["hits"] += h
            e["fall_clips"] += fc
            e["false"] += sum(fb.values())
            e["minutes"] += sum(mb.values())
    for name in ("learned", "fixed"):
        y, p = (np.concatenate([a for a, _ in pooled[name]]), np.concatenate([b for _, b in pooled[name]]))
        h, fc, fb, mb = totals[name]
        out[name] = {"person": prf(y, p), "clip_hits": f"{h}/{fc}", "false_alarms": fb, "minutes": round(mb, 1), "fp_per_min": round(fb / mb, 3)}
    for e in out["sweep"].values():
        e["recall"], e["fp_per_min"], e["minutes"] = round(e["hits"] / e["fall_clips"], 3), round(e["false"] / e["minutes"], 3), round(e["minutes"], 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="mlp", choices=["mlp", "mlp_aug"])
    args = parser.parse_args()
    ds = load_dataset(include_normal=True, include_aug=args.method == "mlp_aug")
    result = evaluate(ds, args.method)
    path = APP_ROOT / "output" / "lying_eval" / f"threshold_{args.method}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("learned", "fixed")}, ensure_ascii=False))
    for t, e in result["sweep"].items():
        print(f"기준 {t}: 재현율 {e['recall']} 오경보 {e['false']}건 / {e['minutes']}분 = {e['fp_per_min']}/분")


if __name__ == "__main__":
    main()
