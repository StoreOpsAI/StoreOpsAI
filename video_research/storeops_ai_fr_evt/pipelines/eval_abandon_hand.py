"""유기 손 확인(models/placement_check) 효과: 새 물건이 생긴 자리에 직전 손이 있었던 것만 유기로 볼 때.

방식: object_verified(새 물건 + 물건 확인), fused_verified(+ 행동 판정기), fused_hand(+ 손 확인 + 주인이 화면에서 사라짐, v4),
fused_pets{0,5,10}(+ 손으로 소유자 연결 + PETS 2006 방치 판정, 유예 초).
- 유기 train 30개: 재현율·오경보. 행동 판정기는 세션 5-fold로, 영상마다 그 영상을 학습하지 않은 fold 끝단을 쓴다(누수 없음)
- 합성 53개(make_abandon_composites, 내려놓은 사람 없음): 붙인 물건에 울리면 오경보
- 다른 행동·반품 114개: 분당 오경보 (행동 끝단은 유기 train 전체)
임계값: BEFORE_SEC는 사전 값, MARGIN 1.0은 유기 train 진단으로 고름(낙관적). 합성은 장면별(person/no_person)로도 센다. 공식 validation 미사용.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from config.config import OBJECT_VERIFY_WEIGHTS
from models.object_verifier import ObjectVerifier
from models.abandon_state import abandon_alarm
from models.placement_check import owner_left, placer
from pipelines.aihub_abandonment import APP_ROOT, parse_annotation, parse_object_boxes
from pipelines.compare_backbones import load_xclip
from pipelines.eval_abandon_composites import is_hit
from pipelines.eval_abandon_false_alarms import list_sources, video_alarms
from pipelines.eval_abandon_fusion import fold_heads, full_head, match_alarms
from pipelines.eval_broken_pose import load_pose_tracks
from pipelines.eval_new_object import load_person_boxes, match_detections
from pipelines.train_abandonment import session_key

PETS_GRACES = (0, 5, 10)  # PETS 2006은 30초지만 AI Hub 영상이 1분이라 짧은 값만 평가할 수 있다
METHODS = {"object_verified": "object", "fused_verified": "fused_verified",
           "fused_hand": "fused_hand",  # v4: 손 확인 + 주인이 화면에서 사라짐
           **{f"fused_pets{g}": f"fused_pets{g}" for g in PETS_GRACES}}  # PETS 방치 판정(models/abandon_state)
POSE_CACHE = APP_ROOT / "output" / "broken_pose"


def alarms_with_hand(video, verifier, head, processor, model):
    import cv2

    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 3.0
    poses = load_pose_tracks(video, POSE_CACHE)
    persons = load_person_boxes(video, APP_ROOT / "output" / "new_object_eval" / "persons")

    last = len(persons) - 1
    owners = {}

    def owner(item):
        key = (tuple(item["box"]), item["first_frame"])
        if key not in owners:
            owners[key] = placer(item, poses, fps, persons)
        return owners[key]

    rules = {  # 행동 판정기는 한 번만 돌리고 규칙만 바꿔 비교
        "hand": lambda item: owner(item) is not None and owner_left(owner(item), poses, fps, last) is not None,
        **{f"pets{g}": (lambda item, g=g: owner(item) is not None
                        and abandon_alarm(item, owner(item), poses, fps, last, grace_sec=g) is not None)
           for g in PETS_GRACES}}
    alarms, seconds = video_alarms(video, verifier, head, processor, model, whole_runs=True, keep=rules)
    return {name: alarms[key] for name, key in METHODS.items()}, seconds, fps


def main():
    parser = argparse.ArgumentParser(description="유기 손 확인 평가")
    parser.add_argument("--source-manifest", type=Path,
                        default=APP_ROOT / "output" / "abandonment_training" / "source_manifest.csv")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "abandonment_training" / "samples.csv")
    parser.add_argument("--feature-cache", type=Path, default=APP_ROOT / "output" / "backbone_compare" / "xclip.npz")
    parser.add_argument("--composite-dir", type=Path, default=APP_ROOT / "output" / "abandon_composites")
    parser.add_argument("--output", type=Path, default=APP_ROOT / "output" / "abandon_hand.json")
    args = parser.parse_args()

    verifier = ObjectVerifier(APP_ROOT / OBJECT_VERIFY_WEIGHTS)
    heads, fold_of = fold_heads(args.manifest, args.feature_cache, 5)
    full = full_head(args.manifest, args.feature_cache)
    processor, model = load_xclip(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary = {name: {"events": 0, "matched": 0, "train_false": 0, "train_minutes": 0.0, "composite_hits": 0,
                      "composites": 0, "other_false": 0, "other_minutes": 0.0} for name in METHODS}

    with args.source_manifest.open(encoding="utf-8-sig") as file:
        abandon = sorted({(r["video"], r["annotation"]) for r in csv.DictReader(file) if r["split"] == "train"})
    for video, annotation in abandon:
        events, boxes = parse_annotation(Path(annotation)), parse_object_boxes(Path(annotation))
        alarms, seconds, fps = alarms_with_hand(video, verifier, heads[fold_of[session_key(video)]], processor, model)
        for name, items in alarms.items():
            if name.startswith("fused"):
                matched, false = match_alarms(items, events, boxes, fps)
            else:
                matched, false = match_detections(items, events, boxes, fps)
            s = summary[name]
            s["events"] += len(events)
            s["matched"] += len(matched)
            s["train_false"] += len(false)
            s["train_minutes"] += seconds / 60

    for label in json.loads((args.composite_dir / "labels.json").read_text(encoding="utf-8")):
        alarms, _, _ = alarms_with_hand(args.composite_dir / label["video"], verifier, full, processor, model)
        for name, items in alarms.items():
            hit = any(is_hit(item, label) for item in items)
            summary[name]["composites"] += 1
            summary[name]["composite_hits"] += hit
            scenario = summary[name].setdefault(f"composite_{label['scenario']}", [0, 0])  # [울림, 전체]
            scenario[0] += hit
            scenario[1] += 1

    for videos in list_sources().values():
        for video in videos:
            alarms, seconds, _ = alarms_with_hand(video, verifier, full, processor, model)
            for name, items in alarms.items():
                summary[name]["other_false"] += len(items)
                summary[name]["other_minutes"] += seconds / 60

    for s in summary.values():
        s["recall"] = round(s["matched"] / max(1, s["events"]), 2)
        s["train_fp_per_min"] = round(s["train_false"] / max(1e-9, s["train_minutes"]), 3)
        s["composite_false_rate"] = round(s["composite_hits"] / max(1, s["composites"]), 2)
        s["other_fp_per_min"] = round(s["other_false"] / max(1e-9, s["other_minutes"]), 3)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, s in summary.items():
        print(name, {k: s[k] for k in ("recall", "train_fp_per_min", "composite_false_rate", "other_fp_per_min",
                                       "composite_person", "composite_no_person")})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
