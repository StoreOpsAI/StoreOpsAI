"""전도 새 데이터(AI Hub 71550, 3fps 180프레임 클립)의 라벨을 읽어 클립 목록과 분할을 만든다(data/lying_manifest.json).

- 라벨(CVAT xml): fall_start~fall_end 구간과 그때 사람 박스. 누운 구간은 fall_start부터 클립 끝까지 이어진다
- 그룹 = 촬영 날짜(AGENTS.md: 같은 날 = 같은 세트·조명·배우·카메라 배치). 다중 카메라 클립은 같은 날짜 그룹에 든다
- 공식 validation(81)은 그대로 validation. train 중 validation과 날짜 또는 장면 번호가 겹치는 클립은 quarantine(학습·dev 미사용)
- 나머지 train을 날짜 그룹 단위 FOLDS-fold로 나누고, 첫 학습 클립(LYING_FIRST_PASS_CLIPS 이상)은 fold를 돌아가며 날짜 그룹째 채운다
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from config.config import AIHUB_ORGANIZED_ROOT, LYING_FIRST_PASS_CLIPS
from pipelines.aihub_abandonment import APP_ROOT

FOLDS = 5
MANIFEST_PATH = APP_ROOT / "data" / "lying_manifest.json"
NAME = re.compile(r"C_3_7_(\d+)_BU_([A-Z]+)_(\d\d-\d\d)_(\d\d-\d\d-\d\d)_(C[A-F])_")


def read_label(xml_path: Path):
    """{frames, falls:[{start, end, box_start, box_end}]}. box = (x1, y1, x2, y2)."""
    root = ET.parse(xml_path).getroot()
    tracks = {"fall_start": [], "fall_end": []}
    for track in root.iter("track"):
        if track.get("label") in tracks:
            box = next(b for b in track.iter("box") if b.get("outside") == "0")
            coords = tuple(round(float(box.get(k)), 1) for k in ("xtl", "ytl", "xbr", "ybr"))
            tracks[track.get("label")].append((int(box.get("frame")), coords))
    starts, ends = sorted(tracks["fall_start"]), sorted(tracks["fall_end"])
    assert len(starts) == len(ends) >= 1, f"fall_start/fall_end 개수 불일치: {xml_path.name}"
    falls = [{"start": s[0], "end": e[0], "box_start": s[1], "box_end": e[1]} for s, e in zip(starts, ends)]
    return {"frames": int(root.find("meta/task/size").text), "falls": falls}


def list_clips(split: str):
    base = Path(AIHUB_ORGANIZED_ROOT) / "전도" / split
    clips = []
    for xml_path in sorted((base / "labels").glob("*.xml")):
        scene, site, date, time, camera = NAME.match(xml_path.name).groups()
        video = base / "videos" / f"{xml_path.stem}.mp4"
        assert video.is_file(), f"라벨에 대응하는 영상 없음: {video}"
        clips.append({"clip": xml_path.stem, "video": str(video), "scene": scene, "site": site, "date": date,
                      "camera": camera, "official_split": split, **read_label(xml_path)})
    return clips


def assign_folds(clips):
    """날짜 그룹을 큰 것부터 클립이 가장 적은 fold에 넣는다(같은 크기면 fold 번호 순, 그룹 순서는 시드 고정)."""
    groups = defaultdict(list)
    for clip in clips:
        groups[clip["date"]].append(clip)
    load = [0] * FOLDS
    order = sorted(groups, key=lambda d: (-len(groups[d]), d))
    for date in order:
        fold = load.index(min(load))
        load[fold] += len(groups[date])
        for clip in groups[date]:
            clip["fold"] = fold


def pick_first_pass(clips, target, seed=0):
    """fold를 돌아가며 날짜 그룹째 넣어 target개 이상이 될 때까지 채운다."""
    rng = random.Random(seed)
    by_fold = {k: defaultdict(list) for k in range(FOLDS)}
    for clip in clips:
        by_fold[clip["fold"]][clip["date"]].append(clip)
    queues = {k: rng.sample(sorted(g), len(g)) for k, g in by_fold.items()}
    chosen, k = [], 0
    while len(chosen) < target and any(queues.values()):
        if queues[k % FOLDS]:
            chosen += [c["clip"] for c in by_fold[k % FOLDS][queues[k % FOLDS].pop()]]
        k += 1
    return set(chosen)


def build():
    train, val = list_clips("train"), list_clips("validation")
    val_dates, val_scenes = {c["date"] for c in val}, {c["scene"] for c in val}
    for clip in train:
        clip["quarantine"] = clip["date"] in val_dates or clip["scene"] in val_scenes
    usable = [c for c in train if not c["quarantine"]]
    assign_folds(usable)
    first = pick_first_pass(usable, LYING_FIRST_PASS_CLIPS)
    for clip in usable:
        clip["first_pass"] = clip["clip"] in first
    return train, val


def check(train, val):
    """AGENTS.md 자동 검사: validation과 학습 후보(quarantine 제외)가 날짜·장면·클립을 공유하지 않는다."""
    usable = [c for c in train if not c["quarantine"]]
    for key in ("date", "scene", "clip"):
        shared = {c[key] for c in usable} & {c[key] for c in val}
        assert not shared, f"validation과 {key} 겹침: {sorted(shared)}"
    folds_of_date = defaultdict(set)
    for c in usable:
        folds_of_date[c["date"]].add(c["fold"])
    assert all(len(f) == 1 for f in folds_of_date.values()), "한 날짜가 두 fold에 걸침"


def main():
    train, val = build()
    check(train, val)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({"train": train, "validation": val}, ensure_ascii=False, indent=1), encoding="utf-8")
    usable = [c for c in train if not c["quarantine"]]
    first = [c for c in usable if c["first_pass"]]
    print(json.dumps({"train_clips": len(train), "quarantine": len(train) - len(usable), "usable": len(usable),
                      "validation": len(val), "first_pass": len(first),
                      "first_pass_folds": dict(sorted(Counter(c["fold"] for c in first).items())),
                      "first_pass_dates": len({c["date"] for c in first}),
                      "usable_by_fold": dict(sorted(Counter(c["fold"] for c in usable).items()))}, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
