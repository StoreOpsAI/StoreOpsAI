"""유기 데이터(AI Hub 71550 유기, 3fps 180프레임 클립)의 라벨을 읽어 클립 목록과 분할을 만든다(data/abandon_manifest.json).

- 라벨(CVAT xml): abandon_start~abandon_end 구간(클립당 여러 개 가능)과 놓인 물건 박스(object_abandon 트랙: 나타난 프레임·박스)
- 그룹 = (장면 번호, 사이트, 날짜) 촬영 세션. 같은 촬영을 찍은 카메라들은 같은 그룹이다(AGENTS.md 규칙 2).
  날짜만 같은 다른 장면은 갈라도 된다(사용자 결정 2026-09-29: 행동 학습이 목표라 날짜 겹침은 통제하지 않는다)
- 공식 validation(80)은 그대로 validation. train 중 validation과 같은 세션(장면·사이트·날짜)인 클립은 quarantine
- 나머지 train을 세션 그룹 단위 FOLDS-fold로 나누고, 첫 학습 클립(전도와 같은 수, ABANDON_FIRST_PASS_CLIPS)을 fold를 돌아가며 세션째 채운다
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from config.config import ABANDON_FIRST_PASS_CLIPS, AIHUB_ORGANIZED_ROOT
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_manifest import FOLDS

ABANDON_MANIFEST_PATH = APP_ROOT / "data" / "abandon_manifest.json"
NAME = re.compile(r"C_3_11_(\d+)_BU_([A-Z]+)_(\d\d-\d\d)_(\d\d-\d\d-\d\d)_(C[A-F]|[a-f])_")


def read_label(xml_path: Path):
    """{frames, events:[{start, end}], objects:[{frame, box}]}. box = (x1, y1, x2, y2), frame = 물건이 처음 보이는 프레임."""
    root = ET.parse(xml_path).getroot()
    starts, ends, objects = [], [], []
    for track in root.iter("track"):
        label = track.get("label")
        boxes = [b for b in track.iter("box") if b.get("outside") == "0"]
        if not boxes:
            continue
        if label == "abandon_start":
            starts.append(int(boxes[0].get("frame")))
        elif label == "abandon_end":
            ends.append(int(boxes[0].get("frame")))
        elif label == "object_abandon":
            box = boxes[0]
            objects.append({"frame": int(box.get("frame")),
                            "box": [round(float(box.get(k)), 1) for k in ("xtl", "ytl", "xbr", "ybr")]})
    assert len(starts) == len(ends) >= 1, f"abandon_start/abandon_end 개수 불일치: {xml_path.name}"
    events = [{"start": s, "end": e} for s, e in zip(sorted(starts), sorted(ends))]
    return {"frames": int(root.find("meta/task/size").text), "events": events, "objects": sorted(objects, key=lambda o: o["frame"])}


def list_clips(split: str):
    base = Path(AIHUB_ORGANIZED_ROOT) / "유기" / split
    clips = []
    for xml_path in sorted((base / "labels").glob("*.xml")):
        scene, site, date, _, camera = NAME.match(xml_path.name).groups()
        video = base / "videos" / f"{xml_path.stem}.mp4"
        assert video.is_file(), f"라벨에 대응하는 영상 없음: {video}"
        clips.append({"clip": xml_path.stem, "video": str(video), "scene": scene, "site": site, "date": date, "camera": camera,
                      "session": f"{scene}_{site}_{date}", "official_split": split, **read_label(xml_path)})
    return clips


def assign_folds(clips):
    groups = defaultdict(list)
    for clip in clips:
        groups[clip["session"]].append(clip)
    load = [0] * FOLDS
    for session in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        fold = load.index(min(load))
        load[fold] += len(groups[session])
        for clip in groups[session]:
            clip["fold"] = fold


def pick_first_pass(clips, target, seed=0):
    rng = random.Random(seed)
    by_fold = {k: defaultdict(list) for k in range(FOLDS)}
    for clip in clips:
        by_fold[clip["fold"]][clip["session"]].append(clip)
    queues = {k: rng.sample(sorted(g), len(g)) for k, g in by_fold.items()}
    chosen, k = [], 0
    while len(chosen) < target and any(queues.values()):
        if queues[k % FOLDS]:
            chosen += [c["clip"] for c in by_fold[k % FOLDS][queues[k % FOLDS].pop()]]
        k += 1
    return set(chosen)


def build():
    train, val = list_clips("train"), list_clips("validation")
    val_sessions = {c["session"] for c in val}
    for clip in train:
        clip["quarantine"] = clip["session"] in val_sessions
    usable = [c for c in train if not c["quarantine"]]
    assign_folds(usable)
    first = pick_first_pass(usable, ABANDON_FIRST_PASS_CLIPS)
    for clip in usable:
        clip["first_pass"] = clip["clip"] in first
    return train, val


def check(train, val):
    """AGENTS.md 자동 검사: validation과 학습 후보(quarantine 제외)가 세션·클립을 공유하지 않고, 한 세션이 두 fold에 걸치지 않는다."""
    usable = [c for c in train if not c["quarantine"]]
    for key in ("session", "clip"):
        assert not ({c[key] for c in usable} & {c[key] for c in val}), f"validation과 {key} 겹침"
    folds_of = defaultdict(set)
    for c in usable:
        folds_of[c["session"]].add(c["fold"])
    assert all(len(f) == 1 for f in folds_of.values()), "한 세션이 두 fold에 걸침"


def main():
    train, val = build()
    check(train, val)
    ABANDON_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABANDON_MANIFEST_PATH.write_text(json.dumps({"train": train, "validation": val}, ensure_ascii=False, indent=1), encoding="utf-8")
    usable = [c for c in train if not c["quarantine"]]
    first = [c for c in usable if c["first_pass"]]
    print(json.dumps({"train_clips": len(train), "quarantine": len(train) - len(usable), "usable": len(usable), "validation": len(val),
                      "first_pass": len(first), "first_pass_folds": dict(sorted(Counter(c["fold"] for c in first).items())),
                      "first_pass_sessions": len({c["session"] for c in first}),
                      "events_first_pass": sum(len(c["events"]) for c in first), "usable_by_fold": dict(sorted(Counter(c["fold"] for c in usable).items()))},
                     ensure_ascii=False))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
