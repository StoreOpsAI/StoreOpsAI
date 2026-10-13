"""행동 zip(AI Hub 71550 TS/TL/VS/VL)에서 학습할 영상만 organized/<행동>/<split>/{videos,labels}로 푼다 (디스크가 모자라 전부 풀지 않는다).

train: 세션(장면, 사이트, 날짜) 단위로 무작위(시드 고정) 순서로 BEHAVIOR_TRAIN_CLIPS개가 될 때까지 세션째 채운다(같은 장면의 다중 카메라 포함).
validation: 전부. 풀 때 CRC 검사(끝까지 읽음)와 크기 대조를 하고, 모두 일치할 때만 영상 zip을 지운다. 라벨 zip은 전부 풀고 지운다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path

from config.config import AIHUB_ORGANIZED_ROOT, BEHAVIOR_TRAIN_CLIPS
from pipelines.behavior_features import NAME
from pipelines.behavior_labels import PREFIX

CODES = {"전도": "07", "파손": "08", "방화": "09", "유기": "11", "절도": "12", "폭행": "13"}
SEED = "behavior_select_v1"


def find_zip(kind: str, split_dir: str, folder: str, code: str) -> Path:
    root = Path(AIHUB_ORGANIZED_ROOT).parent / "71550"
    found = list(root.glob(f"*/*/{split_dir}/{folder}/{kind}_03.이상행동_{code}*.zip"))
    if len(found) != 1:
        raise FileNotFoundError(f"{kind} {code} zip {len(found)}개: {split_dir}")
    return found[0]


def write_member(z: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> bool:
    with z.open(info) as src, open(target, "wb") as dst:  # 끝까지 읽으면 zipfile이 CRC를 검사한다
        while chunk := src.read(1 << 24):
            dst.write(chunk)
    return target.stat().st_size == info.file_size


def select_sessions(videos: list[zipfile.ZipInfo], labeled: set[str], limit: int) -> set[str]:
    groups = defaultdict(list)
    for info in videos:
        stem = Path(info.filename).stem
        if stem in labeled:
            m = NAME.match(Path(info.filename).name)
            groups["_".join(m.group(1, 2, 3))].append(stem)
    order = sorted(groups, key=lambda s: hashlib.md5((SEED + s).encode()).hexdigest())
    chosen, n = set(), 0
    for session in order:
        if n >= limit:
            break
        chosen.update(groups[session])
        n += len(groups[session])
    return chosen


def diverse_selection(behavior: str) -> set[str] | None:
    """select_diverse가 고른 train 영상(output/diverse_<행동>_train.csv). 파일이 없으면 None(세션 단위 선택을 쓴다). 고르지 않은 영상은 풀지 않아 디스크를 아낀다."""
    f = Path(__file__).resolve().parents[1] / "output" / f"diverse_{behavior}_train.csv"
    return {r["clip"] for r in csv.DictReader(open(f, encoding="utf-8")) if r["selected"] == "True"} if f.is_file() else None


def extract(behavior: str, split: str) -> None:
    code, (v_split, l_split) = CODES[behavior], (("Training",) * 2 if split == "train" else ("Validation",) * 2)
    prefix_v, prefix_l = ("TS", "TL") if split == "train" else ("VS", "VL")
    zv = find_zip(prefix_v, v_split, "01.원천데이터", code)
    zl = find_zip(prefix_l, l_split, "02.라벨링데이터", code)
    dest = Path(AIHUB_ORGANIZED_ROOT) / behavior / split
    (dest / "videos").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    ok = True
    with zipfile.ZipFile(zl) as lz:
        labels = [i for i in lz.infolist() if not i.is_dir()]
        for info in labels:
            ok &= write_member(lz, info, dest / "labels" / Path(info.filename).name)
    labeled = {Path(i.filename).stem for i in labels}
    with zipfile.ZipFile(zv) as vz:
        videos = [i for i in vz.infolist() if not i.is_dir()]
        picked = diverse_selection(behavior) if split == "train" else None
        chosen = set(labeled & {Path(i.filename).stem for i in videos}) if split == "validation" else \
            (picked & labeled if picked is not None else select_sessions(videos, labeled, BEHAVIOR_TRAIN_CLIPS))
        for info in videos:
            if Path(info.filename).stem in chosen:
                ok &= write_member(vz, info, dest / "videos" / Path(info.filename).name)
    # 고르지 않은 라벨은 남겨 둔다(정상 구간 통계용). 영상 없는 라벨은 추출기가 건너뛴다
    print(behavior, split, "영상", len(chosen), "개 / 라벨", len(labels), "개", "크기 일치" if ok else "**크기 불일치**", flush=True)
    if ok:
        zv.unlink()
        zl.unlink()
        print("zip 삭제", zv.name, zl.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, choices=sorted(PREFIX))
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    extract(args.behavior, args.split)


if __name__ == "__main__":
    main()
