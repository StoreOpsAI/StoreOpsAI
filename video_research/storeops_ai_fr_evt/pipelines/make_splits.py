"""모든 행동 모듈이 같이 쓰는 fold 분할(data/splits.json)을 만든다. AGENTS.md 분할 규칙.

- 대상: 유기·절도·파손·방화 train 영상, 구매행동 반품 영상, 전도/train 클립. 공식 validation은 넣지 않는다(최종 시험용 보존)
- 그룹 = 촬영 날짜(파일명의 MM-DD). 같은 날은 같은 세트·조명·배우·카메라 배치이므로 다중 카메라·다른 행동 영상까지 한 그룹
- 그룹을 FOLDS개 fold에 나눈다. 큰 그룹부터, 행동별 비율이 가장 덜 찬 fold에 넣어 행동마다 고르게 퍼지게 한다
어떤 모듈이든 fold k를 평가할 때 fold k 영상(다른 행동의 음성 포함)은 학습·어려운 음성 수집에 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT
from pipelines.prepare_behaviors import list_videos

FOLDS = 5
SPLITS_PATH = APP_ROOT / "data" / "splits.json"


def shooting_date(name):
    return re.match(r"C_\d+_\d+_\d+_BU_[A-Z]+_(\d\d-\d\d)_", name).group(1)


def collect():
    """{파일(또는 클립 폴더) 이름: 행동}."""
    items = {}
    for behavior in ("유기", "절도", "파손", "방화"):
        for _, mp4, split in list_videos(WORKSPACE_ROOT / behavior, behavior)[0]:
            if split == "train":
                items[Path(mp4).name] = behavior
    items.update({p.name: "반품" for p in (WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4")})

    # 전도는 <workspace>/전도/train 클립만. 공식 validation(전도/val, handoff val 목록)과 겹치면 멈춘다
    from csv import DictReader
    with (WORKSPACE_ROOT / "handoff_0921" / "manifests" / "val_manifest.csv").open(encoding="utf-8-sig") as file:
        val_clips = {row["clip_folder"] for row in DictReader(file)}
    falls = {p.name for p in (WORKSPACE_ROOT / "전도" / "train").glob("*__fall")}
    assert not falls & val_clips, f"전도 train에 공식 validation 클립: {sorted(falls & val_clips)[:3]}"
    session = lambda name: re.sub(r"_C[A-F]_RGB", "_RGB", name)
    assert not {session(name) for name in falls} & {session(name) for name in val_clips}, \
        "전도 train과 공식 validation에 같은 촬영 세션이 있음"
    items.update({name: "전도" for name in falls})
    return items


def assign(items, fixed=None):
    """fixed({그룹: fold})에 있는 그룹은 그 fold를 유지하고(이미 학습·평가에 쓴 분할이 흔들리지 않게), 새 그룹만 배정한다."""
    fixed = fixed or {}
    groups = defaultdict(Counter)
    for name, behavior in items.items():
        groups[shooting_date(name)][behavior] += 1
    totals = Counter(items.values())
    filled = [Counter() for _ in range(FOLDS)]
    fold_of = {}
    for group in [g for g in groups if g in fixed]:
        fold_of[group] = fixed[group]
        filled[fold_of[group]] += groups[group]
    new = [g for g in groups if g not in fixed]
    for group in sorted(new, key=lambda g: (-sum(groups[g].values()), hashlib.sha1(g.encode()).hexdigest())):
        load = [max((filled[k][b] + n) / totals[b] for b, n in groups[group].items()) for k in range(FOLDS)]
        fold_of[group] = load.index(min(load))
        filled[fold_of[group]] += groups[group]
    return {name: {"behavior": b, "group": shooting_date(name), "fold": fold_of[shooting_date(name)]}
            for name, b in sorted(items.items())}, filled


def load_splits():
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


def main():
    fixed = {item["group"]: item["fold"] for item in load_splits().values()} if SPLITS_PATH.is_file() else {}
    splits, filled = assign(collect(), fixed)
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
    groups_per_fold = defaultdict(set)
    for item in splits.values():
        groups_per_fold[item["fold"]].add(item["group"])
    for k in range(FOLDS):
        print(k, dict(filled[k]), sorted(groups_per_fold[k]))
    # 누수 검사: 한 그룹이 두 fold에 있으면 안 된다
    folds_of_group = defaultdict(set)
    for item in splits.values():
        folds_of_group[item["group"]].add(item["fold"])
    assert all(len(f) == 1 for f in folds_of_group.values()), "그룹이 여러 fold에 걸침"
    print(f"{len(splits)}개 → {SPLITS_PATH}")


if __name__ == "__main__":
    main()
