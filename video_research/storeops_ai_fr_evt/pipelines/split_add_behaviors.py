"""분할표(data/global_split.csv)에 파손·방화 영상을 **덧붙인다**(기존 행은 한 줄도 바꾸지 않는다. global_split.py를 다시 돌리지 않는 이유: 영상 목록을 폴더에서 다시 모아 기존 fold가 바뀔 수 있다).

- train: output/diverse_<행동>_train.csv에서 selected=True인 영상(select_diverse가 고른 것), validation: AI Hub val 영상 전부. 공식 split은 그대로 둔다.
- 날짜는 격리에 쓰지 않는다(사용자 결정 2026-10-08): q_* 열은 모두 False. 그래서 파손·방화 평가는 공식 validation과 학습의 날짜가 겹칠 수 있고 점수가 낙관적일 수 있다(결과와 함께 밝힌다).
- fold: 분할표에 이미 있는 날짜는 그 fold를 따르고, 새 날짜는 영상 수가 가장 적은 fold에 날짜째 넣는다(한 날짜는 fold 하나, global_split.assign_folds와 같은 규칙).
- 이어붙이기 전 분할표를 data/global_split.before_<행동>.csv로 복사하고, 이미 있는 영상 이름은 건너뛴다(여러 번 실행해도 같다).
사용: python -m pipelines.split_add_behaviors
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

from config.config import AIHUB_ORGANIZED_ROOT, BEHAVIOR_FOLDS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.global_split import NAME, SPLIT_PATH

BEHAVIORS = ("파손", "방화")
Q_COLUMNS = ("q_fall", "q_abandon", "q_theft", "q_fire")


def new_rows(behavior: str, existing: set[str]) -> list[dict]:
    selected = {r["clip"] for r in csv.DictReader(open(APP_ROOT / "output" / f"diverse_{behavior}_train.csv", encoding="utf-8")) if r["selected"] == "True"}
    val = {p.stem for p in (Path(AIHUB_ORGANIZED_ROOT) / behavior / "validation" / "labels").glob("*.xml")}
    rows = []
    for split, clips in (("train", selected), ("validation", val)):
        for stem in sorted(clips - existing):
            m = NAME.match(stem)
            if m:
                rows.append({"clip": stem, "behavior": behavior, "official_split": split, "scene": m.group(1), "site": m.group(2), "date": m.group(3), "camera": m.group(4)})
    return rows


def assign(rows: list[dict], fold_of: dict[str, int], load: Counter) -> None:
    by_date = Counter(r["date"] for r in rows if r["date"] not in fold_of)
    for date, n in sorted(by_date.items(), key=lambda kv: (-kv[1], kv[0])):
        k = min(range(BEHAVIOR_FOLDS), key=lambda i: (load[i], i))
        fold_of[date], load[k] = k, load[k] + n
    for r in rows:
        r.update(group_id=r["date"], fold=fold_of[r["date"]], **{q: False for q in Q_COLUMNS})


def main() -> None:
    old = list(csv.DictReader(open(SPLIT_PATH, encoding="utf-8")))
    fields = list(old[0])
    fold_of = {r["date"]: int(r["fold"]) for r in old}
    load = Counter(int(r["fold"]) for r in old)
    existing = {r["clip"] for r in old}
    for b in BEHAVIORS:
        rows = new_rows(b, existing)
        if not rows:
            print(f"{b}: 추가할 영상 없음"); continue
        shutil.copy(SPLIT_PATH, SPLIT_PATH.with_name(f"global_split.before_{b}.csv"))
        assign(rows, fold_of, load)
        with open(SPLIT_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)
        existing |= {r["clip"] for r in rows}
        print(f"{b}: train {sum(r['official_split'] == 'train' for r in rows)}개, validation {sum(r['official_split'] == 'validation' for r in rows)}개 추가, 날짜 {len({r['date'] for r in rows})}개")
    check()


def check() -> None:
    rows = list(csv.DictReader(open(SPLIT_PATH, encoding="utf-8")))
    folds: dict[str, set[str]] = {}
    for r in rows:
        folds.setdefault(r["group_id"], set()).add(r["fold"])
    assert all(len(v) == 1 for v in folds.values()), "날짜 하나가 여러 fold에 있음"
    assert len({r["clip"] for r in rows}) == len(rows), "영상 이름 중복"
    print("점검 통과: 날짜당 fold 하나, 영상 이름 중복 없음. fold별 영상 수", dict(sorted(Counter(r["fold"] for r in rows if r["official_split"] == "train").items())))


def _self_check() -> None:
    rows = [{"date": "01-01"}, {"date": "01-01"}, {"date": "02-02"}, {"date": "03-03"}]
    fold_of, load = {"01-01": 2}, Counter({0: 5, 1: 1, 2: 9, 3: 5, 4: 5})
    assign(rows, fold_of, load)
    assert rows[0]["fold"] == rows[1]["fold"] == 2                      # 이미 있는 날짜는 그 fold
    assert rows[2]["fold"] == 1 and rows[3]["fold"] == 1                # 새 날짜는 영상 수가 가장 적은 fold(1: 1개 -> 2개, 여전히 최소)
    assert all(r["q_fall"] is False and r["group_id"] == r["date"] for r in rows)
    print("ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _self_check() if "--self-check" in sys.argv else main()
