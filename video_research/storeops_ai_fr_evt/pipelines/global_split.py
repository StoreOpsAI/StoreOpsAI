"""전 행동 공통 분할표(data/global_split.csv): 촬영 날짜 그룹 fold + 헤드별 격리 + 자동 겹침 검사 (AGENTS.md 1~7, 스펙 02 §2).

- 그룹 = 촬영 날짜. 한 날짜는 어떤 행동의 영상이든 fold 하나에만 속한다(행동끼리 정상·부정 예시를 섞어 쓰는 헤드에서 누수가 없도록).
- fold 우선순위: 전도 manifest > 매장이동 usable > 반품 usable > 나머지(클립 수가 가장 적은 fold에 날짜째). 기존 전도·정상 fold는 바꾸지 않는다.
- 헤드 H의 봉인 검증 = H의 공식 validation + 매장이동 validation. 반품 101클립은 공식 봉인이 아닌 개발용 풀이라 전부 학습·OOF에 쓴다(날짜 fold). 학습 가능한 클립 중 봉인 검증과 날짜 또는 장면이 겹치면 H에서 격리(q_<H>=True).
- 영상은 지워도 된다: 영상 목록은 organized 영상 폴더와 창 특징 폴더(C:/data/features)의 파일 이름에서 모은다.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config.config import AIHUB_ORGANIZED_ROOT, BEHAVIOR_FEATURE_ROOT, BEHAVIOR_FOLDS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_manifest import MANIFEST_PATH as LYING_PATH
from pipelines.normal_manifest import NORMAL_MANIFEST_PATH
from pipelines.return_manifest import RETURN_MANIFEST_PATH

SPLIT_PATH = APP_ROOT / "data" / "global_split.csv"
NAME = re.compile(r"C_\d_\d+_(\d+)_BU_([A-Z]+)_(\d\d-\d\d)_\d\d-\d\d-\d\d_(C[A-F]|[a-f])_")
BEHAVIORS = ("전도", "유기", "절도", "방화", "매장이동", "반품")
FEATURE_DIRS = {"fall": "전도", "abandon": "유기", "theft": "절도", "fire": "방화", "normal": None}  # normal은 이름 앞부분으로 매장이동/반품 구분
HEADS = {"fall": ("전도", "매장이동"), "abandon": ("유기", "매장이동"), "theft": ("절도", "매장이동"), "fire": ("방화", "매장이동")}  # 헤드별 봉인 검증(정상 봉인은 매장이동 validation)


def _behavior_of(stem: str, default: str | None) -> str:
    return default or ("반품" if stem.startswith("C_2_5_") else "매장이동")


def collect_clips() -> dict[str, dict]:
    """{영상 이름: {behavior, official_split, scene, site, date, camera}}. 영상 폴더와 창 특징 폴더의 이름을 합친다."""
    clips: dict[str, dict] = {}

    def add(stem: str, behavior: str, split: str) -> None:
        split = "train" if behavior == "반품" else split  # 반품 101개는 공식 봉인 세트가 아니라 우리가 나눈 개발용 풀: 전체를 날짜 fold로 쓴다
        m = NAME.match(stem)
        if m and stem not in clips:
            clips[stem] = {"clip": stem, "behavior": behavior, "official_split": split, "scene": m.group(1), "site": m.group(2), "date": m.group(3), "camera": m.group(4)}

    for b in BEHAVIORS:
        for split in ("train", "validation"):
            for v in (Path(AIHUB_ORGANIZED_ROOT) / b / split / "videos").glob("*.mp4"):
                add(v.stem, b, split)
    for prefix, behavior in FEATURE_DIRS.items():
        for split in ("train", "validation"):
            for f in (Path(BEHAVIOR_FEATURE_ROOT) / f"{prefix}_{split}").glob("*.npz"):
                if not f.name.endswith(".tmp.npz"):
                    add(f.stem, _behavior_of(f.stem, behavior), split)
    return clips


def assign_folds(clips: dict[str, dict]) -> dict[str, int]:
    lying = json.loads(LYING_PATH.read_text(encoding="utf-8"))
    normal = json.loads(NORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    ret = json.loads(RETURN_MANIFEST_PATH.read_text(encoding="utf-8"))
    fold: dict[str, int] = {}
    for rows in ([c for c in lying["train"] if not c["quarantine"]], [c for c in normal["train"] if not c["quarantine"]], [c for c in ret if not c["quarantine"]]):
        for c in rows:
            if fold.setdefault(c["date"], c["fold"]) != c["fold"]:
                raise ValueError(f"기존 manifest끼리 날짜 {c['date']}의 fold가 다름")
    load = Counter()
    for c in clips.values():
        if c["date"] in fold:
            load[fold[c["date"]]] += 1
    by_date = Counter(c["date"] for c in clips.values() if c["date"] not in fold)
    for date, n in sorted(by_date.items(), key=lambda kv: (-kv[1], kv[0])):
        k = min(range(BEHAVIOR_FOLDS), key=lambda i: (load[i], i))
        fold[date], load[k] = k, load[k] + n
    return fold


def build() -> list[dict]:
    clips = collect_clips()
    fold = assign_folds(clips)
    sealed = {h: [c for c in clips.values() if c["official_split"] == "validation" and c["behavior"] in bs] for h, bs in HEADS.items()}
    seal_dates = {h: {c["date"] for c in rows} for h, rows in sealed.items()}
    seal_scenes = {h: {c["scene"] for c in rows} for h, rows in sealed.items()}
    rows = []
    for c in sorted(clips.values(), key=lambda c: (c["behavior"], c["clip"])):
        row = {**c, "group_id": c["date"], "fold": fold[c["date"]]}
        for h in HEADS:
            is_val = c["official_split"] == "validation" and c["behavior"] in HEADS[h]
            row[f"q_{h}"] = (not is_val) and (c["date"] in seal_dates[h] or c["scene"] in seal_scenes[h])
        rows.append(row)
    return rows


def check_overlap(rows: list[dict]) -> dict[str, dict]:
    """학습 가능(격리 아님) 클립과 봉인 검증이 날짜·장면·영상 이름을 공유하지 않는지, 날짜마다 fold가 하나인지 검사한다."""
    report = {}
    date_folds = defaultdict(set)
    for r in rows:
        date_folds[r["group_id"]].add(r["fold"])
    assert all(len(v) == 1 for v in date_folds.values()), "날짜 하나가 여러 fold에 있음"
    for h, bs in HEADS.items():
        val = [r for r in rows if r["official_split"] == "validation" and r["behavior"] in bs]
        train = [r for r in rows if r["official_split"] == "train" and not r[f"q_{h}"]]
        for key in ("clip", "date", "scene"):
            shared = {r[key] for r in val} & {r[key] for r in train}
            assert not shared, f"{h}: 봉인 검증과 학습이 {key}를 공유: {sorted(shared)[:3]}"
        report[h] = {"sealed_val": len(val), "train_usable": len(train), "train_quarantined": sum(r[f"q_{h}"] for r in rows),
                     "usable_by_behavior": dict(Counter(r["behavior"] for r in train)), "usable_by_fold": dict(sorted(Counter(r["fold"] for r in train).items()))}
    return report


def main() -> None:
    rows = build()
    report = check_overlap(rows)
    with open(SPLIT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"클립 {len(rows)}개 -> {SPLIT_PATH}")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
