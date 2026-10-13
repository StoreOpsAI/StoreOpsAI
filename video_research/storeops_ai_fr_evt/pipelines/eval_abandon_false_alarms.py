"""유기가 없는 다른 행동 영상(절도·파손·방화 train, 구매행동 반품)에서 유기 경보가 몇 번 울리는지 방식별로 센다.

방식: object(새 물건 + 물건 확인), fused(행동 AND 새 물건, eval_abandon_fusion과 같은 규칙), fused_verified(둘 다).
- 공식 validation 영상은 쓰지 않는다(list_videos가 test로 표시한 것 제외).
- 행동 끝단은 유기 train 전체로 학습한다. 이 영상들은 유기 학습에 쓰지 않았으므로 누수가 없다.
- 이 영상들에는 유기 사건이 없으므로 경보는 모두 오경보로 센다.
  (방화처럼 은박 쟁반을 실제로 두고 가는 장면은 사람이 보고 판단하도록 경보 장면을 이미지로 남긴다)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import OBJECT_VERIFY_WEIGHTS
from models.object_verifier import ObjectVerifier
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT
from pipelines.compare_backbones import load_xclip, xclip_embed
from pipelines.eval_abandon_fusion import build_windows, full_head, fuse, load_tracks, window_clips
from pipelines.eval_new_object import detect_new_objects, load_person_boxes
from pipelines.prepare_behaviors import list_videos


def list_sources():
    """{출처: [mp4, ...]}. 행동 영상은 train split만, 반품은 구매행동/videos 전체."""
    sources = {name: [mp4 for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]
               for name in ("절도", "파손", "방화")}
    sources["반품"] = sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))
    return sources


def _tile(video, alarm, size=200):
    """경보 박스 주변을 잘라 박스를 그린 타일."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, alarm["confirmed_frame"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((size, size, 3), np.uint8)
    x1, y1, x2, y2 = alarm["box"]
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = max(0, x1 - w), max(0, y1 - h)
    crop = frame[cy1:y2 + h, cx1:x2 + w].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
    return cv2.resize(crop, (size, size))


METHODS = ("object", "fused", "fused_verified")


def video_alarms(video, verifier, head, processor, model, whole_runs=False, keep=None):
    """(방식별 경보 목록, 초). fused 경보는 사건 단위로, 기본은 사건의 첫 물건, whole_runs면 사건의 물건 목록 전체다.

    keep(물건 → bool)을 주면 확인된 물건 중 keep을 통과한 것만으로 object_keep·fused_keep도 만든다(예: 손 확인).
    keep이 {이름: 함수}면 이름마다 object_<이름>·fused_<이름>을 만든다(행동 판정기를 한 번만 돌리고 여러 규칙 비교).
    """
    persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons")
    detections, fps = detect_new_objects(video, persons, verifier=verifier)
    verified = [item for item in detections if item["verified"]]
    windows = build_windows(load_tracks(video, APP_ROOT / "output" / "fusion_eval" / "tracks"), fps)
    clips = window_clips(video, windows)
    features = [xclip_embed(processor, model, clips[i:i + 8]) for i in range(0, len(clips), 8)]
    probabilities = head(np.concatenate(features)).tolist() if features else []
    pick = (lambda run: run) if whole_runs else (lambda run: run[0])
    alarms = {"object": verified, "fused": [pick(run) for run in fuse(windows, probabilities, detections, fps)],
              "fused_verified": [pick(run) for run in fuse(windows, probabilities, verified, fps)]}
    for name, rule in ({"keep": keep} if callable(keep) else keep or {}).items():
        kept = [item for item in verified if rule(item)]
        alarms[f"object_{name}"] = kept
        alarms[f"fused_{name}"] = [pick(run) for run in fuse(windows, probabilities, kept, fps)]
    return alarms, len(persons) / fps


def main():
    parser = argparse.ArgumentParser(description="다른 행동 영상에서 유기 오경보 측정 (방식별)")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "abandonment_training" / "samples.csv")
    parser.add_argument("--feature-cache", type=Path, default=APP_ROOT / "output" / "backbone_compare" / "xclip.npz")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "abandon_false_alarms")
    args = parser.parse_args()

    verifier = ObjectVerifier(APP_ROOT / OBJECT_VERIFY_WEIGHTS)
    head = full_head(args.manifest, args.feature_cache)
    processor, model = load_xclip(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary, tiles = {}, defaultdict(list)
    for source, videos in list_sources().items():
        alarms, seconds = defaultdict(list), 0.0
        for video in videos:
            per_method, length = video_alarms(video, verifier, head, processor, model)
            seconds += length
            for method in METHODS:
                alarms[method] += [{"video": Path(video).name, **item} for item in per_method[method]]
            tiles[source] += [_tile(video, item) for item in per_method["fused"]]
        minutes = seconds / 60
        summary[source] = {"videos": len(videos), "minutes": round(minutes, 1)}
        for method in METHODS:
            summary[source][method] = {"false_alarms": len(alarms[method]),
                                       "fp_per_minute": round(len(alarms[method]) / minutes, 3) if minutes else 0.0,
                                       "alarms": alarms[method]}
        print(source, json.dumps({m: {k: v for k, v in summary[source][m].items() if k != "alarms"} for m in METHODS},
                                 ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    for source, items in tiles.items():
        if not items:
            continue
        items += [np.zeros_like(items[0])] * (-len(items) % 6)
        sheet = np.vstack([np.hstack(items[i:i + 6]) for i in range(0, len(items), 6)])
        cv2.imencode(".jpg", sheet)[1].tofile(str(args.output_dir / f"fused_alarms_{source}.jpg"))


if __name__ == "__main__":
    main()
