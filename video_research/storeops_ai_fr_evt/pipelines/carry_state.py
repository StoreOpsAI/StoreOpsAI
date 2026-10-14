"""물건 '운반됨' 상태 규칙(오프라인 평가): 사람에게 닿은 물건이 가려지면 그 사람이 들고 있다고 보고, 그 사람이 화면 밖으로 나가는지로 유기·절도 후보를 만든다.

근거: 가려짐·운반 상태를 구분해 물체 영속성을 추적하는 TCOW(CVPR 2023), 낮은 점수 탐지까지 궤적에 잇는 ByteTrack(ECCV 2022), 운반 물건을 사람과 묶는 추적(ECCV 2012 계열).
우리는 박스·사람 추적 규칙으로 단순화(근거 약함).
- 입력: C:/data/features/objdet30(물건 탐지, 3fps 표본), persons(사람 추적). 손 위치가 없어 '손에 닿음'은 물건 중심이 사람 박스(여유 margin) 안인 것으로 근사한다.
- 절도 후보: 사람과 겹치던 물건 궤적이 겹친 채로 끝나고(gap표본보다 오래 안 보임) 그 사람이 화면 가장자리(border 비율 안)에서 exit표본 이상 사라짐. 시각 = 사라진 표본.
- 유기 후보: 사람과 겹치던 물건이 겹치지 않는 상태로 static표본 이상 남아 있고(궤적 유지) 그 사람이 같은 방식으로 사라짐.
- 평가: 정답 물건 박스(XML) 프레임 범위 [첫 프레임-PRE, 끝+POST표본]에 후보 시각이 들어오면 적중. 그 밖의 후보는 오경보. 정상 = 매장이동 영상(걷기·서 있기).
- 선택: 날짜 fold {0,1,2}에서 오경보/시간 최소(재현율 하한 RECALL_FLOOR), {3,4}에서 보고. 공식 validation은 쓰지 않는다.
"""
from __future__ import annotations

import csv
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT, AIHUB_ORGANIZED_ROOT
from pipelines.global_split import SPLIT_PATH
from pipelines.object_dataset2 import read_boxes

ROOT = Path(BEHAVIOR_FEATURE_ROOT)
LABEL = {"유기": "object_abandon", "절도": "object_article"}
PRE, POST, RECALL_FLOOR = 6, 30, 0.5
GRID = {"conf": (0.10, 0.25), "margin": (0.0, 0.15), "gap": (1, 3), "static": (2, 4), "exit": (2, 4), "border": (0.05, 0.15)}
W, H = 1920.0, 1080.0


def iou(a, b) -> float:
    iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def build_tracks(obj: dict, n: int, conf: float, gap: int) -> list[dict]:
    """물건 박스를 IoU로 이어 궤적을 만든다(탐욕 매칭, gap표본까지 끊겨도 같은 궤적)."""
    tracks: list[dict] = []
    by: dict[int, list] = {}
    for f, b, c in zip(obj["frame"], obj["box"], obj["conf"]):
        if c >= conf:
            by.setdefault(int(f), []).append(b)
    for f in range(n):
        alive = [t for t in tracks if f - t["last"] <= gap + 1]
        pairs = sorted(((iou(t["box"], b), ti, bi) for ti, t in enumerate(alive) for bi, b in enumerate(by.get(f, []))), reverse=True)
        used_t, used_b = set(), set()
        for v, ti, bi in pairs:
            if v < 0.2 or ti in used_t or bi in used_b:
                continue
            t = alive[ti]
            t["frames"][f] = by[f][bi]
            t["last"], t["box"] = f, by[f][bi]
            used_t.add(ti), used_b.add(bi)
        for bi, b in enumerate(by.get(f, [])):
            if bi not in used_b:
                tracks.append({"frames": {f: b}, "last": f, "box": b})
    return tracks


def carrier_at(box, persons: dict[int, list], f: int, margin: float):
    """물건 중심이 사람 박스(여유 margin) 안이면 그 사람(tid, 박스)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for tid, b in persons.get(f, []):
        mx, my = (b[2] - b[0]) * margin, (b[3] - b[1]) * margin
        if b[0] - mx <= cx <= b[2] + mx and b[1] - my <= cy <= b[3] + my:
            return tid, b
    return None


def exit_frame(pos_box, f0: int, persons: dict[int, list], n: int, exit_n: int, border: float):
    """f0 이후 그 사람(가까운 박스로 이어 감)이 화면 가장자리에서 exit_n표본 이상 사라지는 첫 표본. 없으면 None."""
    last = ((pos_box[0] + pos_box[2]) / 2, (pos_box[1] + pos_box[3]) / 2)
    ph, missing = max(pos_box[3] - pos_box[1], 1.0), 0
    for f in range(f0 + 1, n):
        cands = [b for _, b in persons.get(f, []) if np.hypot((b[0] + b[2]) / 2 - last[0], (b[1] + b[3]) / 2 - last[1]) <= ph]
        if cands:
            b = min(cands, key=lambda b: np.hypot((b[0] + b[2]) / 2 - last[0], (b[1] + b[3]) / 2 - last[1]))
            last, missing = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 0
        else:
            missing += 1
            if missing >= exit_n:
                near = last[0] < border * W or last[0] > (1 - border) * W or last[1] < border * H or last[1] > (1 - border) * H
                return (f - missing + 1) if near else None
    return None


def candidates(obj: dict, persons: dict[int, list], n: int, p: dict) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"유기": [], "절도": []}
    for t in build_tracks(obj, n, p["conf"], p["gap"]):
        fr = sorted(t["frames"])
        flags = [carrier_at(t["frames"][f], persons, f, p["margin"]) for f in fr]
        held = [i for i, c in enumerate(flags) if c]
        if not held or len(fr) < 2:
            continue
        i_last = held[-1]
        f_held, pbox = fr[i_last], flags[i_last][1]
        if i_last == len(fr) - 1 and (n - 1 - t["last"]) > p["gap"]:             # 사람과 겹친 채로 궤적이 끝남 = 가려짐(운반됨)
            x = exit_frame(pbox, f_held, persons, n, p["exit"], p["border"])
            if x is not None:
                out["절도"].append(x)
        detached = [f for f, c in zip(fr[i_last + 1:], flags[i_last + 1:]) if c is None]
        if len(detached) >= p["static"]:                                          # 떨어져서 가만히 남음
            x = exit_frame(pbox, f_held, persons, n, p["exit"], p["border"])
            if x is not None and x <= fr[-1] + p["gap"] + 1:
                out["유기"].append(x)
    return out


def load_clip(r: dict) -> dict:
    o, pz = np.load(ROOT / "objdet30" / f"{r['clip']}.npz"), np.load(ROOT / "persons" / f"{r['clip']}.npz")
    persons: dict[int, list] = {}
    for f, tid, b in zip(pz["frame"], pz["tid"], pz["box"]):
        persons.setdefault(int(f), []).append((int(tid), b))
    span = None
    if r["behavior"] in LABEL:
        gt = read_boxes(Path(AIHUB_ORGANIZED_ROOT) / r["behavior"] / "train" / "labels" / f"{r['clip']}.xml", LABEL[r["behavior"]])
        span = (min(gt) - PRE, max(gt) + POST) if gt else None
    return {"r": r, "obj": {k: o[k] for k in ("frame", "box", "conf")}, "persons": persons, "n": int(pz["n_frames"]), "span": span}


def run_params(args) -> dict:
    clips, p = args
    res = {k: {"tp": 0, "events": 0, "fa": 0, "fa_normal": 0} for k in ("유기", "절도")}
    for c in clips:
        cand = candidates(c["obj"], c["persons"], c["n"], p)
        beh = c["r"]["behavior"]
        for kind in ("유기", "절도"):
            in_span = lambda x: c["span"] is not None and beh == kind and c["span"][0] <= x <= c["span"][1]
            if beh == kind and c["span"] is not None:
                res[kind]["events"] += 1
                res[kind]["tp"] += int(any(in_span(x) for x in cand[kind]))
            fa = sum(1 for x in cand[kind] if not in_span(x))
            res[kind]["fa"] += fa
            if beh == "매장이동":
                res[kind]["fa_normal"] += fa
    return {"p": p, "res": res}


def main() -> None:
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")) if r["official_split"] == "train" and r["q_fall"] != "True" and r["behavior"] in ("유기", "절도", "매장이동")
            and (ROOT / "objdet30" / f"{r['clip']}.npz").is_file() and (ROOT / "persons" / f"{r['clip']}.npz").is_file()]
    with Pool(12) as pool:
        clips = pool.map(load_clip, rows)
    print(f"클립 {len(clips)}개(유기 {sum(c['r']['behavior'] == '유기' for c in clips)}, 절도 {sum(c['r']['behavior'] == '절도' for c in clips)}, 매장이동 {sum(c['r']['behavior'] == '매장이동' for c in clips)})", flush=True)
    sel = [c for c in clips if int(c["r"]["fold"]) in (0, 1, 2)]
    tst = [c for c in clips if int(c["r"]["fold"]) in (3, 4)]
    hrs = lambda cs: sum(c["n"] for c in cs if c["r"]["behavior"] == "매장이동") / 3 / 3600
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    with Pool(12) as pool:
        a = pool.map(run_params, [(sel, p) for p in combos])
        b = pool.map(run_params, [(tst, p) for p in combos])
    for kind in ("유기", "절도"):
        rows_ = []
        for ra, rb in zip(a, b):
            x, y = ra["res"][kind], rb["res"][kind]
            rows_.append((ra["p"], x["tp"] / max(x["events"], 1), x["fa_normal"] / max(hrs(sel), 1e-9), y["tp"] / max(y["events"], 1), y["fa_normal"] / max(hrs(tst), 1e-9), x["fa"], y["fa"]))
        ok = [r for r in rows_ if r[1] >= RECALL_FLOOR]
        print(f"\n[{kind}] 정상(매장이동) 시간 선택 {hrs(sel):.1f}h / 보고 {hrs(tst):.1f}h | 재현율 하한 {RECALL_FLOOR} 이상 설정 {len(ok)}/{len(rows_)}개", flush=True)
        for p, rs, fs, rt, ft, fa_s, fa_t in (sorted(ok, key=lambda r: r[2])[:4] or sorted(rows_, key=lambda r: -r[1])[:3]):
            print(f"  {p} | 선택 재현율 {rs:.2f} 정상 오경보 {fs:.1f}/시간 | 보고 재현율 {rt:.2f} 정상 오경보 {ft:.1f}/시간 | 전체 영상 오경보 후보 {fa_s}/{fa_t}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
