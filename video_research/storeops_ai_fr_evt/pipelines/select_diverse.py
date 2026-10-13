"""파손·방화 학습 영상을 개수에 맞춰 고르되, 장소(사이트·장면)·카메라 각도·출연자(파일명 끝 F2/M3 등) 종류가 최대한 남게 뽑는다. 날짜는 보지 않는다(사용자 결정 2026-10-08).

- 입력: AI Hub 71550 라벨 zip의 xml 파일 이름(영상 이름과 같음). 영상이 풀려 있지 않아도 실행할 수 있다.
- 방법: (사이트, 카메라, 출연자) 묶음마다 장면 순서로 번갈아 한 개씩 뽑는 돌려뽑기. 한 묶음에서는 아직 안 뽑은 장면을 먼저 뽑고, 장면은 이름순(고정)으로 정한다. 같은 장면의 다중 카메라 영상은 카메라가 달라 서로 다른 각도로 센다.
- 출력: output/diverse_<행동>_<split>.csv (clip, site, scene, camera, actor, date, selected)와 선택 전후 종류 수. 공식 split(train/validation)은 그대로 두고 train만 고른다.
사용: python -m pipelines.select_diverse --behavior 파손 --target 600
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
import zipfile
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from config.config import AIHUB_ORGANIZED_ROOT
from pipelines.aihub_abandonment import APP_ROOT

NAME = re.compile(r"C_\d_\d+_(\d+)_BU_([A-Z]+)_(\d\d-\d\d)_\d\d-\d\d-\d\d_(C[A-F]|[a-f])_.*_([FM]\d)$")
CODES = {"파손": "08", "방화": "09"}
ROOT = Path(AIHUB_ORGANIZED_ROOT).parent / "71550"


def clips_from_names(names: list[str]) -> list[dict]:
    out = []
    for n in names:
        stem = Path(n).stem
        m = NAME.match(stem)
        if m:
            out.append({"clip": stem, "scene": m.group(1), "site": m.group(2), "date": m.group(3), "camera": m.group(4), "actor": m.group(5)})
    return out


def pick_diverse(clips: list[dict], target: int) -> list[dict]:
    """(사이트, 카메라, 출연자) 묶음을 돌아가며 한 개씩. 묶음 안에서는 새 장면 우선."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in sorted(clips, key=lambda c: (c["scene"], c["clip"])):
        groups[(c["site"], c["camera"], c["actor"])].append(c)
    queues = []
    for key in sorted(groups):
        seen, first, rest = set(), [], []
        for c in groups[key]:
            (rest if c["scene"] in seen else first).append(c)
            seen.add(c["scene"])
        queues.append(first + rest)
    chosen = []
    for row in zip_longest(*queues):
        chosen += [c for c in row if c]
        if len(chosen) >= target:
            break
    return chosen[:target]


def kinds(clips: list[dict]) -> str:
    return " / ".join(f"{k} {len({c[k] for c in clips})}" for k in ("site", "scene", "camera", "actor", "date"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True, choices=list(CODES))
    ap.add_argument("--target", type=int, required=True)
    args = ap.parse_args()
    zf = glob.glob(str(ROOT / f"*/*/Training/02.라벨링데이터/TL_03.이상행동_{CODES[args.behavior]}*.zip"))
    names = zipfile.ZipFile(zf[0]).namelist() if zf else [p.name for p in (Path(AIHUB_ORGANIZED_ROOT) / args.behavior / "train" / "labels").glob("*.xml")]
    clips = clips_from_names(names)
    chosen = pick_diverse(clips, args.target)
    print(f"{args.behavior} train {len(clips)}개 -> {len(chosen)}개 선택")
    print(f"  전체: {kinds(clips)}\n  선택: {kinds(chosen)}")
    pick = {c["clip"] for c in chosen}
    out = APP_ROOT / "output" / f"diverse_{args.behavior}_train.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(clips[0]) + ["selected"])
        w.writeheader()
        w.writerows({**c, "selected": c["clip"] in pick} for c in clips)
    print("저장:", out)


def _self_check() -> None:
    names = [f"C_3_8_{s}_BU_{site}_08-10_14-41-19_{cam}_RGB_DF2_{a}.xml" for s, site, cam, a in
             [(1, "DYA", "CA", "F2"), (1, "DYA", "CB", "F2"), (2, "DYA", "CA", "F2"), (3, "DYB", "CC", "M3"), (4, "DYB", "CC", "M3")]]
    cl = clips_from_names(names)
    assert len(cl) == 5 and cl[3]["actor"] == "M3"
    pick = pick_diverse(cl, 3)
    assert len(pick) == 3 and {c["site"] for c in pick} == {"DYA", "DYB"} and len({(c["site"], c["camera"], c["actor"]) for c in pick}) == 3   # 서로 다른 묶음에서 한 개씩
    print("ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _self_check() if "--self-check" in sys.argv else main()
