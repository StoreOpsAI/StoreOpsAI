"""공식 방화·전도 평가 전에 원본·촬영 그룹 누수를 검사한다."""

import re

from pipelines.aihub_abandonment import WORKSPACE_ROOT
from pipelines.make_splits import shooting_date
from pipelines.prepare_behaviors import list_videos


def check():
    scene_day = lambda n: re.search(r"BU_[A-Z]+_\d\d-\d\d", n).group()
    fire, missing = list_videos(WORKSPACE_ROOT / "방화", "방화")
    # AI Hub의 train 라벨 6개는 대응 영상이 없다. 공식 validation 라벨 누락은 허용하지 않는다.
    val_xml = list((WORKSPACE_ROOT / "방화").rglob("*validation/*.xml"))
    assert len(val_xml) == sum(split == "test" for _, _, split in fire), "방화 validation 영상 누락"
    train = {p.name for _, p, split in fire if split == "train"}
    val = {p.name for _, p, split in fire if split == "test"}
    assert train and val and not train & val, "방화 원본 영상 중복 또는 split 비어 있음"
    train_dates = {shooting_date(n) for n in train}
    fire_val_dates = {shooting_date(n) for n in val}
    fire_overlap = train_dates & fire_val_dates
    fire_scene_overlap = {scene_day(n) for n in train} & {scene_day(n) for n in val}
    all_val_dates = {shooting_date(p.name) for behavior in ("방화", "절도", "파손", "유기")
                     for _, p, split in list_videos(WORKSPACE_ROOT / behavior, behavior)[0] if split == "test"}
    all_val_scenes = {scene_day(p.name) for behavior in ("방화", "절도", "파손", "유기")
                      for _, p, split in list_videos(WORKSPACE_ROOT / behavior, behavior)[0] if split == "test"}
    all_val_dates |= {shooting_date(n) for n in (p.name for p in (WORKSPACE_ROOT / "전도" / "val").glob("*__fall"))}
    all_val_scenes |= {scene_day(p.name) for p in (WORKSPACE_ROOT / "전도" / "val").glob("*__fall")}

    fall_train = {p.name for p in (WORKSPACE_ROOT / "전도" / "train").glob("*__fall")}
    fall_val = {p.name for p in (WORKSPACE_ROOT / "전도" / "val").glob("*__fall")}
    assert fall_train and fall_val and not fall_train & fall_val, "전도 원본 클립 중복 또는 split 비어 있음"
    session = lambda n: re.sub(r"_C[A-F]_RGB", "_RGB", n)
    fall_session_overlap = {session(n) for n in fall_train} & {session(n) for n in fall_val}
    train_dates = {shooting_date(n) for n in fall_train}
    fall_val_dates = {shooting_date(n) for n in fall_val}
    fall_overlap = train_dates & fall_val_dates
    fall_scene_overlap = {scene_day(n) for n in fall_train} & {scene_day(n) for n in fall_val}
    print(f"방화 train {len(train)} / validation {len(val)} (짝 없는 train 라벨 {missing}), 촬영 날짜 중복 {sorted(fire_overlap)}, 제외 후 train {sum(shooting_date(n) not in fire_val_dates for n in train)}")
    print(f"전도 train {len(fall_train)} / validation {len(fall_val)}, 다중 카메라 세션 중복 {len(fall_session_overlap)}, 촬영 날짜 중복 {sorted(fall_overlap)}, 제외 후 train {sum(shooting_date(n) not in fall_val_dates for n in fall_train)}")
    print(f"모든 행동 validation 촬영 날짜 {sorted(all_val_dates)}, 전부 제외 시 방화 train {sum(shooting_date(n) not in all_val_dates for n in train)}, 전도 train {sum(shooting_date(n) not in all_val_dates for n in fall_train)}")
    print(f"매장·날짜 기준 중복: 방화 {sorted(fire_scene_overlap)}, 전도 {sorted(fall_scene_overlap)}")
    print(f"매장·날짜 중복 제외 후 train: 방화 {sum(scene_day(n) not in {scene_day(v) for v in val} for n in train)}, 전도 {sum(scene_day(n) not in {scene_day(v) for v in fall_val} for n in fall_train)}")
    print(f"모든 행동 validation 매장·날짜 중복 제외 후 train: 방화 {sum(scene_day(n) not in all_val_scenes for n in train)}, 전도 {sum(scene_day(n) not in all_val_scenes for n in fall_train)}")
    assert not fire_overlap and not fall_session_overlap and not fall_overlap, "공식 평가 전 촬영 그룹 누수"


if __name__ == "__main__":
    check()
