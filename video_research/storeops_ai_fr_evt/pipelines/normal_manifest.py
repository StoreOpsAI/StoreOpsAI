"""정상 데이터(AI Hub 71549 매장이동, 3fps 약 900프레임=5분 클립)의 목록과 분할을 만든다(data/normal_manifest.json).

- 라벨(xml)은 관절 이동 정보(`moving`)뿐이라 이 클립의 사람은 모두 정상(0)으로 쓴다
- 그룹 = 촬영 날짜. 공식 validation과 날짜 또는 장면 번호가 겹치는 train 클립은 quarantine(학습·dev 미사용). validation은 손대지 않는다
- 남은 날짜 그룹을 전도 first_pass의 fold에 이어 붙인다: 정상 사람 수(클립 수)가 가장 적은 fold에 날짜 그룹째 넣는다
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from config.config import AIHUB_ORGANIZED_ROOT
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_manifest import FOLDS, MANIFEST_PATH as LYING_MANIFEST_PATH

NORMAL_MANIFEST_PATH = APP_ROOT / "data" / "normal_manifest.json"
NAME = re.compile(r"C_1_1_(\d+)_BU_([A-Z]+)_(\d\d-\d\d)_(\d\d-\d\d-\d\d)_(C[A-F]|[a-f])_")


def list_clips(split: str):
    base = Path(AIHUB_ORGANIZED_ROOT) / "매장이동" / split
    clips = []
    for video in sorted((base / "videos").glob("*.mp4")):
        scene, site, date, _, camera = NAME.match(video.name).groups()
        label = base / "labels" / f"{video.stem}.xml"
        assert label.is_file(), f"영상에 대응하는 라벨 없음: {label}"
        frames = int(ET.parse(label).getroot().find("meta/task/size").text)
        clips.append({"clip": video.stem, "video": str(video), "scene": scene, "site": site, "date": date,
                      "camera": camera, "official_split": split, "frames": frames, "falls": []})
    return clips


def assign_folds(clips):
    lying = json.loads(LYING_MANIFEST_PATH.read_text(encoding="utf-8"))
    load = Counter(c["fold"] for c in lying["train"] if c.get("first_pass"))  # 전도 쪽 부하에서 시작해 균형을 맞춘다
    groups = defaultdict(list)
    for clip in clips:
        groups[clip["date"]].append(clip)
    for date in sorted(groups, key=lambda d: (-len(groups[d]), d)):
        fold = min(range(FOLDS), key=lambda k: (load[k], k))
        load[fold] += len(groups[date])
        for clip in groups[date]:
            clip["fold"] = fold


def build():
    train, val = list_clips("train"), list_clips("validation")
    val_dates, val_scenes = {c["date"] for c in val}, {c["scene"] for c in val}
    for clip in train:
        clip["quarantine"] = clip["date"] in val_dates or clip["scene"] in val_scenes
    usable = [c for c in train if not c["quarantine"]]
    assign_folds(usable)
    lying = json.loads(LYING_MANIFEST_PATH.read_text(encoding="utf-8"))
    lying_dates = {c["date"] for c in lying["train"] + lying["validation"]}
    assert not ({c["date"] for c in usable} & lying_dates), "전도와 매장이동이 촬영 날짜를 공유함"
    for key in ("date", "scene", "clip"):
        assert not ({c[key] for c in usable} & {c[key] for c in val}), f"validation과 {key} 겹침"
    return train, val


def main():
    train, val = build()
    NORMAL_MANIFEST_PATH.write_text(json.dumps({"train": train, "validation": val}, ensure_ascii=False, indent=1), encoding="utf-8")
    usable = [c for c in train if not c["quarantine"]]
    print(json.dumps({"train": len(train), "quarantine": len(train) - len(usable), "usable": len(usable), "validation": len(val),
                      "usable_dates": dict(Counter(c["date"] for c in usable)), "usable_by_fold": dict(sorted(Counter(c["fold"] for c in usable).items())),
                      "usable_minutes": round(sum(c["frames"] for c in usable) / 3 / 60)}, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
