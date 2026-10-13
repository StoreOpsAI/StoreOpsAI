"""삭제 전 점검: 분할 목록(global_split.csv)의 영상마다 캐시(창 목록·화면 전체·사람 크롭·손 크롭·관절)가 있고 서로 맞는지 확인한다. 영상은 읽지 않는다.

- 대상: 격리(q_fall)되지 않은 영상. 학습(train)과 공식 validation을 나눠 센다. 반품은 손 크롭·관절을 만들지 않았으므로(사용자 지시) 그 둘은 검사에서 뺀다.
- 일치 검사: 창 시작 목록 = 화면 전체/사람 크롭/손 크롭 캐시의 창 키, 각 창 JPEG 12개가 열리는지(영상마다 첫·마지막 창만), 관절 캐시 프레임 범위가 창 범위를 덮는지.
사용: python -m pipelines.audit_features
"""
from __future__ import annotations

import collections
import csv
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.fullscene_windows import WIN
from pipelines.global_split import SPLIT_PATH

ROOT = Path(BEHAVIOR_FEATURE_ROOT)
POSE = Path(__file__).resolve().parents[1] / "output" / "pose_cache"


def jpeg_ok(items: list[bytes]) -> bool:
    return len(items) == WIN and all(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) is not None for b in items)


def check(r: dict) -> list[str]:
    clip, bad = r["clip"], []
    ff = ROOT / "fullscene" / f"{clip}.npz"
    if not ff.is_file():
        return ["fullscene 없음"]
    starts = np.load(ff)["start"].tolist()
    if not starts:
        return ["창 0개"]
    for name in ("frames224", "crops_c1", "crops_h"):
        if name == "crops_h" and (r["behavior"] == "반품" or r["official_split"] == "validation"):   # 반품·공식 validation은 손 크롭을 만들지 않았다
            continue
        f = ROOT / name / f"{clip}.pkl"
        if not f.is_file():
            bad.append(f"{name} 없음"); continue
        data = pickle.loads(f.read_bytes())
        if name == "frames224":
            wins = [data["jpeg"][s:s + WIN] for s in (starts[0], starts[-1])]
            if len(data["jpeg"]) < starts[-1] + WIN:
                bad.append("frames224 길이 부족")
        else:
            if sorted(k for k in data if k != "region") != sorted(starts):
                bad.append(f"{name} 창 키 불일치")
                continue
            wins = [data[starts[0]], data[starts[-1]]]
        if not all(jpeg_ok(w) for w in wins):
            bad.append(f"{name} JPEG 손상")
    if r["behavior"] != "반품":
        p = POSE / r["behavior"] / ("validation" if r["official_split"] == "validation" else "train") / f"{clip}.npz"
        if not p.is_file():
            bad.append("관절 없음")
        elif int(np.load(p)["n_frames"]) < starts[-1] + WIN:
            bad.append("관절 프레임 부족")
    return bad


def main() -> None:
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")) if r["q_fall"] != "True"]
    miss: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for r in rows:
        key = (r["behavior"], r["official_split"])
        total[key] += 1
        for b in check(r):
            miss[key][b] += 1
    for key in sorted(total):
        print(f"{key[0]:5s} {key[1]:10s} 영상 {total[key]:4d} | " + (", ".join(f"{k} {v}" for k, v in miss[key].items()) or "모두 정상"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
