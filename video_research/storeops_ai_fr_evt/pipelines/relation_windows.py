"""사람 박스·물건 박스만으로 만든 창 단위 사람-물건 관계 시계열(B안 후기 결합용). 관절은 쓰지 않는다.

입력: persons/<영상>.npz(사람 검출·추적), objdet/<영상>.npz(교차 적합 물건 탐지, conf 하한만 낮게), fullscene/<영상>.npz(창 시작 위치).
창 = 12표본(4초, 3fps), 간격 6표본(2초). 프레임마다 장면 전체의 관계 채널 CHANNELS를 만들고 창 안 시계열 (12, C)와 요약(평균·최대·최소·끝-처음)을 저장한다.
임계값(물건 점수, 거리)은 정하지 않고 연속값으로 둔다: 헤드가 시계열에서 쥠·놓음·정지·멀어짐·사라짐을 배운다.
근거: Chao 외 WACV 2018(HICO-DET: 사람·물건 박스 공간 관계를 학습망 입력으로), Shan 외 CVPR 2020(손-물건 접촉 상태를 학습으로 추정). 우리는 손 대신 사람 박스를 쓴다(근거 약함).
출력: C:/data/features/relwin/<영상>.npz = seq(창, 12, C), summary(창, 4C), start
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT

WIN = 12
CHANNELS = ("n_persons", "n_objects", "max_obj_conf", "near_dist", "near_inside", "n_orphans", "orphan_conf", "person_speed",
            "near_obj_speed", "co_move", "obj_size_ratio", "person_gone", "person_new", "valid")
FAR = 1.0       # 사람 키의 몇 배 넘게 떨어진 물건을 '남겨진 물건' 후보로 보는 연속 채널의 기준 길이(채널 정의용, 판정 임계값 아님)
CLIP = 5.0
ROOT = Path(BEHAVIOR_FEATURE_ROOT)
SUFFIX = os.getenv("OBJDET_SUFFIX", "")  # objdet{SUFFIX} -> relwin{SUFFIX}


def group_by_frame(frame: np.ndarray, *arrays: np.ndarray) -> dict[int, tuple[np.ndarray, ...]]:
    out: dict[int, list[np.ndarray]] = {}
    for f in np.unique(frame):
        m = frame == f
        out[int(f)] = tuple(a[m] for a in arrays)
    return {f: v for f, v in out.items()}


def centers(box: np.ndarray) -> np.ndarray:
    return np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], axis=1)


def point_box_distance(pt: np.ndarray, box: np.ndarray) -> float:
    dx = max(box[0] - pt[0], 0.0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0.0, pt[1] - box[3])
    return float(np.hypot(dx, dy))


def frame_channels(pb: np.ndarray, ob: np.ndarray, oc: np.ndarray, prev: dict | None) -> tuple[np.ndarray, dict]:
    """한 프레임의 관계 채널 벡터와 다음 프레임 계산용 상태. pb: 사람 박스(n,4), ob/oc: 물건 박스(m,4)·점수(m)."""
    v = np.zeros(len(CHANNELS), np.float32)
    v[0], v[1], v[13] = len(pb), len(ob), 1.0
    v[2] = float(oc.max()) if len(oc) else 0.0
    height = float(np.median(pb[:, 3] - pb[:, 1])) if len(pb) else 0.0
    pc = centers(pb) if len(pb) else np.zeros((0, 2))
    state = {"pc": pc, "oc": centers(ob) if len(ob) else np.zeros((0, 2)), "h": height, "n": len(pb)}
    if prev is not None:
        v[11], v[12] = float(len(pb) < prev["n"]), float(len(pb) > prev["n"])
        if len(pc) and len(prev["pc"]) and height > 0:
            v[7] = float(np.median([np.min(np.hypot(*(prev["pc"] - c).T)) for c in pc]) / height)
    near_obj = None
    if len(pb) and len(ob) and height > 0:
        dist = np.array([[point_box_distance(c, b) for b in pb] for c in state["oc"]]) / height     # (물건, 사람)
        nearest = dist.min(1)
        k = int(nearest.argmin())
        v[3], v[4] = nearest[k], float(nearest[k] == 0.0)
        far = nearest > FAR
        v[5], v[6] = float(far.sum()), float(oc[far].max()) if far.any() else 0.0
        v[10] = float((ob[k, 3] - ob[k, 1]) / height)
        near_obj = k
    elif len(ob):
        v[5], v[6], v[3] = float(len(ob)), float(oc.max()), CLIP  # 사람이 없으면 모든 물건이 남겨진 후보
    if near_obj is not None and prev is not None and len(prev["oc"]) and height > 0:
        d = np.hypot(*(prev["oc"] - state["oc"][near_obj]).T)
        j = int(d.argmin())
        if d[j] <= height:                                    # 이전 프레임의 가장 가까운 물건을 같은 물건으로 본다
            v[8] = d[j] / height
            pj = int(np.argmin(np.hypot(*(pc - state["oc"][near_obj]).T)))
            pprev = np.hypot(*(prev["pc"] - pc[pj]).T) if len(prev["pc"]) else np.array([0.0])
            v[9] = abs(v[8] - float(pprev.min()) / height)
    state["near"] = near_obj
    return np.clip(v, -CLIP, CLIP), state


def clip_sequence(persons: dict[str, np.ndarray], objects: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
    p = group_by_frame(persons["frame"], persons["box"])
    o = group_by_frame(objects["frame"], objects["box"], objects["conf"])
    seq, prev = np.zeros((n_frames, len(CHANNELS)), np.float32), None
    empty4, empty1 = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    for f in range(n_frames):
        pb = p.get(f, (empty4,))[0]
        ob, oc = o.get(f, (empty4, empty1))
        seq[f], prev = frame_channels(pb, ob, oc, prev)
    return seq


def window_features(seq: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    win = np.stack([seq[s:s + WIN] for s in starts])                      # (창, 12, C)
    summary = np.concatenate([win.mean(1), win.max(1), win.min(1), win[:, -1] - win[:, 0]], axis=1)
    return win, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    out = ROOT / f"relwin{SUFFIX}"
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(f.stem for f in (ROOT / f"objdet{SUFFIX}").glob("*.npz") if not f.name.endswith(".tmp.npz"))[i::n]
    for k, clip in enumerate(clips, 1):
        target = out / f"{clip}.npz"
        pf, ff = ROOT / "persons" / f"{clip}.npz", ROOT / "fullscene" / f"{clip}.npz"
        if target.is_file() or not pf.is_file() or not ff.is_file():
            continue
        persons, objects, starts = np.load(pf), np.load(ROOT / f"objdet{SUFFIX}" / f"{clip}.npz"), np.load(ff)["start"]
        seq = clip_sequence(persons, objects, int(persons["n_frames"]))
        win, summary = window_features(seq, starts)
        np.savez_compressed(target, seq=win, summary=summary, start=starts)
        if k % 100 == 0:
            print(f"[{k}/{len(clips)}] {clip}", flush=True)
    print("RELWIN_DONE", flush=True)


def _self_check() -> None:
    """사람이 물건을 들고 있다가(박스 안) 물건을 두고 떠나면: 처음엔 near_inside=1, 끝엔 사람이 없고 남겨진 물건(n_orphans)>0, 떠나는 프레임에 person_gone=1."""
    n = 12
    person = {"frame": np.arange(0, 8, dtype=np.int32), "box": np.tile(np.array([[100, 100, 160, 300]], np.float32), (8, 1))}  # 0~7프레임만 있음
    obj = {"frame": np.arange(n, dtype=np.int32), "box": np.tile(np.array([[120, 180, 140, 200]], np.float32), (n, 1)), "conf": np.full(n, 0.8, np.float32)}
    seq = clip_sequence(person, obj, n)
    c = {name: i for i, name in enumerate(CHANNELS)}
    assert seq[0, c["near_inside"]] == 1 and seq[0, c["n_persons"]] == 1
    assert seq[8, c["person_gone"]] == 1 and seq[8, c["n_persons"]] == 0
    assert seq[11, c["n_orphans"]] == 1 and seq[11, c["orphan_conf"]] > 0.7          # 사람이 없는 구간에서 물건은 남겨진 후보
    win, summary = window_features(seq, np.array([0]))
    assert win.shape == (1, WIN, len(CHANNELS)) and summary.shape == (1, 4 * len(CHANNELS))
    empty = clip_sequence({"frame": np.zeros(0, np.int32), "box": np.zeros((0, 4), np.float32)}, {"frame": np.zeros(0, np.int32), "box": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32)}, 5)
    assert (empty[:, c["n_objects"]] == 0).all() and np.isfinite(empty).all()
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
