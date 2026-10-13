"""파손 결합 평가: 파손 행동 판정기(X-CLIP + PCA 32 + 선형) AND 새 물건 여러 개.

- 파손 train 영상(양성)과 구매행동 반품 영상(정상 음성)만 쓴다. 공식 validation은 쓰지 않는다.
- 판정기는 cv_behaviors와 같은 세션 fold·증강(학습 쪽만)·PCA로 학습하고, 각 영상에는 그 영상 세션을 뺀 fold의 판정기만 쓴다.
- 결합 규칙: 파손 확률 >= 0.5인 사람 창의 시작 ~ 끝 + FUSION_AFTER_SEC 사이에 새 물건이 min_objects개 이상 생김.
  던진 물건은 사람에게서 멀리 떨어질 수 있으므로 위치는 보지 않는다.
- 같은 사람의 겹치는 양성 창은 한 경보 후보로 묶는다. 파손 사건 [start, end]과 겹치면 적중,
  사건 전후 여유(eval_broken_evidence의 MARGIN) 밖이면 오경보, 그 사이 전환 구간은 세지 않는다.
- 새 물건 목록은 eval_broken_evidence의 프레임 캐시를 쓴다(먼저 그 스크립트를 돌린다).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from config.config import FUSION_AFTER_SEC
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.compare_backbones import _fit_linear, load_xclip, xclip_embed
from pipelines.cv_abandonment import assign_folds
from pipelines.cv_behaviors import AUGMENTATIONS, fit_pca
from pipelines.eval_abandon_fusion import build_windows, load_tracks, window_clips
from pipelines.eval_broken_evidence import MARGIN_AFTER_SEC, MARGIN_BEFORE_SEC
from pipelines.prepare_behaviors import list_videos
from pipelines.train_abandonment import session_key

BEHAVIOR = "파손"
PCA_DIMS = 32
MIN_OBJECTS = (0, 1, 2, 3)  # 0 = 판정기 단독


def fold_heads(manifest, feature_cache, folds):
    """(fold별 파손 판정기, {세션: fold}). 증강본은 학습 쪽 샘플 것만 쓴다."""
    with manifest.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] != "test"]
    data = np.load(feature_cache)
    if data["ids"].tolist() != [row["sample_id"] for row in rows]:
        raise RuntimeError("특징 캐시와 manifest가 다릅니다. cv_behaviors를 다시 실행하세요.")
    fold_of = assign_folds(rows, folds)
    fold_ids = np.array([fold_of[row["group"]] for row in rows])
    labels = np.array([row["label"] == BEHAVIOR for row in rows], dtype=np.int64)
    heads = []
    for fold in range(folds):
        train = np.where(fold_ids != fold)[0]
        x_train = np.concatenate([data["features"][train], data["augmented"][train].reshape(-1, data["features"].shape[1])])
        y_train = np.concatenate([labels[train], np.repeat(labels[train], AUGMENTATIONS)])
        project = fit_pca(x_train, PCA_DIMS)
        predict = _fit_linear(project(x_train), y_train)
        heads.append(lambda values, predict=predict, project=project: predict(project(values)))
    return heads, fold_of


def alarm_runs(windows, probabilities):
    """같은 사람의 겹치는 양성 창을 묶은 [{"start", "end", "windows"}]."""
    runs = []
    for window, probability in sorted(zip(windows, probabilities), key=lambda item: (item[0]["track"], item[0]["start"])):
        if probability < 0.5:
            continue
        last = runs[-1] if runs else None
        if last is None or last["track"] != window["track"] or window["start"] > last["end"]:
            runs.append({"track": window["track"], "start": window["start"], "end": window["end"], "windows": []})
        runs[-1]["end"] = max(runs[-1]["end"], window["end"])
        runs[-1]["windows"].append(window)
    return runs


def run_objects(run, added, fps):
    """경보 후보의 창들 중 가장 많은 새 물건 수 (창 시작 ~ 끝 + FUSION_AFTER_SEC)."""
    return max(sum(w["start"] <= f <= w["end"] + FUSION_AFTER_SEC * fps for f in added) for w in run["windows"])


def score_video(runs, added, events, fps, frames):
    """min_objects별 {matched, false_alarms}와 음성 구간 초."""
    spans = [(e["start_frame"] - MARGIN_BEFORE_SEC * fps, e["end_frame"] + MARGIN_AFTER_SEC * fps) for e in events]
    counts = [run_objects(run, added, fps) for run in runs]
    result = {}
    for min_objects in MIN_OBJECTS:
        kept = [run for run, count in zip(runs, counts) if count >= min_objects]
        matched = {e["event_index"] for e in events for r in kept if r["start"] <= e["end_frame"] and r["end"] >= e["start_frame"]}
        false_alarms = [r for r in kept if not any(r["start"] <= b and r["end"] >= a for a, b in spans)]
        result[min_objects] = {"matched": len(matched),
                               "false_alarms": [(r["start"], r["end"]) for r in false_alarms]}
    negative_frames = sum(1 for f in range(frames) if not any(a <= f <= b for a, b in spans))
    return result, negative_frames / fps


def list_sources():
    broken = [(mp4, parse_annotation(xml, "broken"))
              for xml, mp4, split in list_videos(WORKSPACE_ROOT / BEHAVIOR, BEHAVIOR)[0] if split == "train"]
    returns = [(mp4, []) for mp4 in sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))]
    return {"파손": broken, "반품": returns}


def main():
    parser = argparse.ArgumentParser(description="파손 결합 평가 (판정기 AND 새 물건 여러 개, train·반품만)")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "behavior_training" / "samples_purchase.csv")
    parser.add_argument("--feature-cache", type=Path,
                        default=APP_ROOT / "output" / "behavior_cv_purchase" / "xclip_features.npz")
    parser.add_argument("--evidence-dir", type=Path, default=APP_ROOT / "output" / "broken_evidence" / "frames")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "broken_fusion")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    heads, fold_of = fold_heads(args.manifest, args.feature_cache, args.folds)
    processor, model = load_xclip(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary = {}
    for source, videos in list_sources().items():
        totals = {m: {"matched": 0, "false_alarms": []} for m in MIN_OBJECTS}
        events_total, negative_seconds = 0, 0.0
        for video, events in videos:
            head = heads[fold_of[session_key(video)]]  # 없으면 KeyError: fold 밖 영상은 평가하지 않는다
            motion, added, _, fps = json.loads((args.evidence_dir / f"{Path(video).stem}.json").read_text(encoding="utf-8"))
            windows = build_windows(load_tracks(video, APP_ROOT / "output" / "fusion_eval" / "tracks"), fps)
            clips = window_clips(video, windows)
            features = [xclip_embed(processor, model, clips[i:i + 8]) for i in range(0, len(clips), 8)]
            probabilities = head(np.concatenate(features)).tolist() if features else []
            per_min, seconds = score_video(alarm_runs(windows, probabilities), added, events, fps, len(motion))
            events_total += len(events)
            negative_seconds += seconds
            for min_objects, item in per_min.items():
                totals[min_objects]["matched"] += item["matched"]
                totals[min_objects]["false_alarms"] += [{"video": Path(video).name, "frames": span}
                                                        for span in item["false_alarms"]]
        minutes = negative_seconds / 60
        summary[source] = {"videos": len(videos), "events": events_total, "negative_minutes": round(minutes, 1),
                           **{f"min_objects_{m}": {
                               "recall": round(t["matched"] / events_total, 3) if events_total else None,
                               "false_alarms": len(t["false_alarms"]),
                               "fp_per_minute": round(len(t["false_alarms"]) / minutes, 3) if minutes else 0.0,
                               "alarms": t["false_alarms"]} for m, t in totals.items()}}
        print(source, json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "alarms"} if isinstance(v, dict) else v)
                                  for k, v in summary[source].items()}, ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
