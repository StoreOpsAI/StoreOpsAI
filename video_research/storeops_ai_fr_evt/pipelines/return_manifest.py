"""정상 풀 확장: 구매행동 반품 train 영상(output/lying_pose_return)의 목록과 fold를 만든다(data/return_manifest.json).

- 라벨 관절·구간이 없는 클립이고 이 클립의 사람은 정상(0)으로 쓴다. 단 사람이 검수한 것이 아니라 data/hard_normal_review.json에서
  "uncertain"으로 판정된 클립은 라벨 오염 위험으로 quarantine한다.
- 그룹 = 촬영 날짜(전도·매장이동과 같은 규칙). 전도(학습 가능 전체) 또는 매장이동 usable과 날짜가 같으면 그 fold를 그대로 따른다(같은 날짜가 두 fold로 갈라지지 않게).
  그 외는 정상 클립 수가 가장 적은 fold에 날짜째 넣는다.
- 공식 validation(전도 val, 매장이동 val)과 날짜·장면이 겹치면 quarantine.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config.config import AIHUB_ORGANIZED_ROOT
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.hard_normal_candidates import POSE_DIR
from pipelines.lying_manifest import FOLDS, MANIFEST_PATH as LYING_PATH
from pipelines.normal_manifest import NORMAL_MANIFEST_PATH

RETURN_MANIFEST_PATH = APP_ROOT / "data" / "return_manifest.json"
REVIEW_PATH = APP_ROOT / "data" / "hard_normal_review.json"
NAME = re.compile(r"C_2_5_(\d+)_BU_([A-Z]+)_(\d\d-\d\d)_(\d\d-\d\d-\d\d)_(C[A-F])_")


def build() -> list[dict]:
    lying = json.loads(LYING_PATH.read_text(encoding="utf-8"))
    normal = json.loads(NORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    val_dates = {c["date"] for c in lying["validation"] + normal["validation"]}
    val_scenes = {c["scene"] for c in lying["validation"] + normal["validation"]}
    date_fold = {c["date"]: c["fold"] for c in lying["train"] if not c["quarantine"]}
    date_fold |= {c["date"]: c["fold"] for c in normal["train"] if not c["quarantine"]}
    uncertain = {r["clip"] for r in json.loads(REVIEW_PATH.read_text(encoding="utf-8"))["candidates"].values() if r["label"] == "uncertain"}
    clips = []
    for npz in sorted(POSE_DIR.glob("*.npz")):
        scene, site, date, _, camera = NAME.match(npz.name).groups()
        clips.append({"clip": npz.stem, "video": str(next(Path(AIHUB_ORGANIZED_ROOT).glob(f"반품/*/videos/{npz.stem}.mp4"), Path(AIHUB_ORGANIZED_ROOT) / "반품" / "train" / "videos" / f"{npz.stem}.mp4")),
                      "scene": scene, "site": site, "date": date, "camera": camera, "official_split": "train", "frames": int(np.load(npz)["n_frames"]),
                      "falls": [], "quarantine": date in val_dates or scene in val_scenes or npz.stem in uncertain})
    load = Counter(date_fold.values())
    groups = defaultdict(list)
    for c in (c for c in clips if not c["quarantine"]):
        groups[c["date"]].append(c)
    for date in sorted(groups, key=lambda d: (-len(groups[d]), d)):
        fold = date_fold.get(date, min(range(FOLDS), key=lambda k: (load[k], k)))
        load[fold] += len(groups[date])
        for c in groups[date]:
            c["fold"] = fold
    return clips


def main() -> None:
    clips = build()
    RETURN_MANIFEST_PATH.write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    usable = [c for c in clips if not c["quarantine"]]
    print(json.dumps({"clips": len(clips), "quarantine": len(clips) - len(usable), "usable_by_fold": dict(sorted(Counter(c["fold"] for c in usable).items())),
                      "usable_dates": dict(Counter(c["date"] for c in usable))}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
