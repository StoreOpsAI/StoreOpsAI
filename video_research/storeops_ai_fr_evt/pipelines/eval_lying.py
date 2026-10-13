"""전도 방식 A(CLIP 사진) / B(관절 특징, 선형 pose·MLP mlp) / C(A·B 둘 다 누움일 때만, and·and_mlp) 5-fold 평가. 폴드 = 촬영 날짜 그룹(lying_manifest).

공식 validation·quarantine 미사용. 분류기는 모두 표준화 + 클래스 균형 로지스틱 회귀(설정 고정, dev로 고르지 않음), 임계값 0.5.
- 사람 단위: 누움/정상 precision·recall·F1·혼동행렬, train-dev F1 차이
- 클립 단위: 점수 >= 0.5 박스를 models/fall_detector(같은 자리 LYING_HOLD_SEC 이상 누움)에 넣어 경보를 낸다
  · 적중: 라벨 전도 박스와 겹치는 경보(전환 구간 시작 이후). 오경보: 전환 구간 앞의 같은 장면 정상, 또는 전도 이후 다른 사람의 경보
  · 전환 구간 안의 다른 사람 경보는 판정하지 않고 따로 센다(라벨 시작보다 눕는 순간이 이른 클립이 있음)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

from config.config import LYING_POSITIVE_IOU, LYING_TRANSITION_FRAMES
from models.fall_detector import lying_alarms
from models.new_object_detector import _iou
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.lying_clip_extract import OUTPUT_DIR as CLIP_DIR
from pipelines.lying_augment import augmented_features
from pipelines.lying_manifest import FOLDS
from pipelines.lying_models import fit_mlp
from pipelines.lying_pose_dataset import load_dataset
from pipelines.train_abandonment import _metrics

FPS = 3.0


def attach_clip_features(ds):
    for c in ds:
        c["clip_feat"] = np.load(CLIP_DIR / f"{c['clip']}.npz")["feat"].astype(np.float32)
        assert len(c["clip_feat"]) == len(c["feat"]), f"사람 수 불일치 {c['clip']}"


def labeled(ds, key):
    x = np.concatenate([c[key][c["label"] >= 0] for c in ds])
    y = np.concatenate([c["label"][c["label"] >= 0] for c in ds]).astype(np.int64)
    return x, y


def fit_method(train, method):
    """clip(c) → 사람마다 누움 점수. pose/clip/and는 선형, mlp는 관절 특징 MLP, and_mlp는 MLP·CLIP선형 둘 다 누움일 때."""
    pose = {"pose": _fit_linear, "and": _fit_linear, "mlp": fit_mlp, "mlp_aug": fit_mlp, "and_mlp": fit_mlp}.get(method)
    if method == "mlp_aug":  # 증강은 학습 fold에만: 원본 + 증강 복사본으로 학습(dev는 원본 그대로 평가)
        x, y = labeled(train, "feat")
        xa, ya = augmented_features(train)
        pose = pose(np.concatenate([x, xa]), np.concatenate([y, ya]))
    elif pose:
        pose = pose(*labeled(train, "feat"))
    clip = _fit_linear(*labeled(train, "clip_feat")) if method in ("clip", "and", "and_mlp") else None
    if method in ("pose", "mlp", "mlp_aug"):
        return lambda c: pose(c["feat"])
    if method == "clip":
        return lambda c: clip(c["clip_feat"])
    return lambda c: np.minimum(pose(c["feat"]), clip(c["clip_feat"]))  # 둘 다 누움일 때만 높은 점수


def person_scores(ds, score):
    s = np.concatenate([score(c)[c["label"] >= 0] for c in ds])
    return s, np.concatenate([c["label"][c["label"] >= 0] for c in ds])


def clip_alarms(clip, scores):
    by_frame = defaultdict(list)
    for f, box, s in zip(clip["frame"], clip["box"], scores):
        if s >= 0.5:
            by_frame[int(f)].append(tuple(map(float, box)))
    return lying_alarms([(f, by_frame.get(f, [])) for f in range(clip["frames"])], FPS)


def judge_alarms(clip, alarms):
    """(적중 여부, 오경보 수, 전환 구간 경보 수)."""
    if not clip["falls"]:  # 정상 클립: 경보는 모두 오경보
        return False, len(alarms), 0
    first = min(f["start"] for f in clip["falls"])
    hit, false, transition = False, 0, 0
    for frame, box in alarms:
        on_fallen = any(frame >= f["start"] - LYING_TRANSITION_FRAMES and _iou(box, f["box_end"]) >= LYING_POSITIVE_IOU for f in clip["falls"])
        if on_fallen:
            hit = True
        elif frame < first - LYING_TRANSITION_FRAMES or frame >= first:
            false += 1
        else:
            transition += 1
    return hit, false, transition


def evaluate(ds, method):
    folds = np.array([c["fold"] for c in ds])
    per_fold, dev_true, dev_score = [], [], []
    hits = transition_total = 0
    false_by = {"전도 영상 정상": 0, "매장이동 정상": 0}
    minutes_by = {"전도 영상 정상": 0.0, "매장이동 정상": 0.0}
    for k in range(FOLDS):
        train = [c for c, f in zip(ds, folds) if f != k]  # 영상 증강 클립은 학습 fold에만 들어간다
        dev = [c for c, f in zip(ds, folds) if f == k and not c.get("aug")]  # dev는 원본만
        score = fit_method(train, method)
        (st, yt), (sd, yd) = person_scores(train, score), person_scores(dev, score)
        per_fold.append({"fold": k, "clips": len(dev), "train": _metrics(yt.tolist(), st.tolist()), "dev": _metrics(yd.tolist(), sd.tolist())})
        dev_true.append(yd), dev_score.append(sd)
        for c in dev:
            hit, false, transition = judge_alarms(c, clip_alarms(c, score(c)))
            domain = "전도 영상 정상" if c["falls"] else "매장이동 정상"
            hits, transition_total, false_by[domain] = hits + hit, transition_total + transition, false_by[domain] + false
            normal_frames = max(min(f["start"] for f in c["falls"]) - LYING_TRANSITION_FRAMES, 0) if c["falls"] else c["frames"]
            minutes_by[domain] += normal_frames / FPS / 60
    yt, ss = np.concatenate(dev_true), np.concatenate(dev_score)
    n_orig = sum(not c.get("aug") for c in ds)
    fall_clips = sum(bool(c["falls"]) for c in ds if not c.get("aug"))
    fp = {d: {"false_alarms": false_by[d], "minutes": round(minutes_by[d], 1),
              "fp_per_min": round(false_by[d] / minutes_by[d], 3) if minutes_by[d] else None} for d in false_by}
    return {"method": method, "person_dev": _metrics(yt.tolist(), ss.tolist()),
            "train_dev_f1_gap": round(float(np.mean([f["train"]["f1"] - f["dev"]["f1"] for f in per_fold])), 3),
            "fall_clips": fall_clips, "normal_clips": n_orig - fall_clips, "aug_clips_in_train": len(ds) - n_orig, "clip_recall": round(hits / fall_clips, 3), "clips_hit": int(hits),
            "fp": fp, "transition_alarms_not_judged": transition_total, "folds": per_fold}


def main():
    parser = argparse.ArgumentParser(description="전도 방식 A/B/C 5-fold 평가")
    parser.add_argument("--methods", nargs="+", default=["pose", "clip", "and"], choices=["pose", "clip", "and", "mlp", "mlp_aug", "and_mlp"])
    parser.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 꼬리표(다른 설정 실험이 기존 결과를 덮어쓰지 않게)")
    parser.add_argument("--with-aug", action="store_true", help="영상 증강 클립(output/lying_pose_aug)을 학습 fold에만 추가")
    parser.add_argument("--with-normal", action="store_true", help="매장이동 정상 클립을 학습·오경보 측정에 포함")
    args = parser.parse_args()
    ds = load_dataset(include_normal=args.with_normal, include_aug=args.with_aug)
    if set(args.methods) & {"clip", "and", "and_mlp"}:
        attach_clip_features(ds)
    out_dir = APP_ROOT / "output" / "lying_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        summary = evaluate(ds, method)
        (out_dir / f"{method}{'_normal' if args.with_normal else ''}{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        p = summary["person_dev"]
        print(f"[{method}] 사람 누움 P/R/F1 {p['precision']:.3f}/{p['recall']:.3f}/{p['f1']:.3f} 혼동 {p['confusion_matrix']} gap {summary['train_dev_f1_gap']} | "
              f"클립 적중 {summary['clips_hit']}/{summary['fall_clips']} 오경보 {summary['fp']} 전환구간 {summary['transition_alarms_not_judged']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
