"""유기 역방향 확인: S3D 유기 점수가 오른 창(트리거)에서 되짚어 (1) 창 끝에 보이는 물건이 직전 N표본에는 그 자리에 없었고(새 물건)
(2) 창 뒤 H표본 동안 사람이 없다(떠남)일 때만 경보로 인정한다. N·H·물건 점수 하한·점수 임계값은 OOF에서 탐색한다(재현율>=0.9에서 오경보 최소).
입력: 합친 OOF 로짓(ensemble_eval.merged, A안), persons/·objdet/ 캐시(표본 단위 3fps). 공식 validation은 읽지 않는다.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.ensemble_eval import merged
from pipelines.eval_finetune import evaluate, log_odds
from pipelines.fusion_eval import clips_for
from pipelines.oof_search import candidates_from_values

ROOT, WIN = Path(BEHAVIOR_FEATURE_ROOT), 12
SUFFIX = os.getenv("OBJDET_SUFFIX", "")
GRID_N, GRID_H, GRID_CONF = (3, 6, 12, 24), (0, 2, 4, 8), (0.1, 0.25, 0.5)
TARGET = 0.90


def overlaps(box: np.ndarray, others: np.ndarray) -> bool:
    """box와 한 칸이라도 겹치는 박스가 있는가."""
    return bool(len(others) and ((others[:, 0] < box[2]) & (others[:, 2] > box[0]) & (others[:, 1] < box[3]) & (others[:, 3] > box[1])).any())


def gates_for_clip(clip: str, starts: np.ndarray) -> dict[tuple, np.ndarray]:
    """(conf, N, H) -> 창별 bool: 새 물건 있음 AND 사람 떠남(H=0이면 떠남 조건 없음)."""
    o, p = np.load(ROOT / f"objdet{SUFFIX}" / f"{clip}.npz"), np.load(ROOT / "persons" / f"{clip}.npz")
    n = int(p["n_frames"])
    persons = np.bincount(p["frame"], minlength=n + 1)[:n + 1]
    out = {}
    for conf in GRID_CONF:
        keep = o["conf"] >= conf
        fr, box = o["frame"][keep], o["box"][keep]
        by = {int(f): box[fr == f] for f in np.unique(fr)}
        for N in GRID_N:
            new = np.zeros(len(starts), bool)
            for i, s in enumerate(starts):
                last = by.get(s + WIN - 1, np.zeros((0, 4)))
                prev = np.concatenate([by[f] for f in range(max(0, s - N), s) if f in by] or [np.zeros((0, 4))])
                new[i] = any(not overlaps(b, prev) for b in last)
            for H in GRID_H:
                gone = np.array([H == 0 or (s + WIN + H <= n and persons[s + WIN:s + WIN + H].sum() == 0) for s in starts])
                out[(conf, N, H)] = new & gone
    return out


def main() -> None:
    d = merged()
    z0, valid = log_odds(d["la"], 2), d["label"] != -1
    gates = {k: np.zeros(len(z0), bool) for k in itertools.product(GRID_CONF, GRID_N, GRID_H)}
    clips = np.unique(d["clip"])
    for n_done, clip in enumerate(clips, 1):
        if n_done % 50 == 0:
            print(f"[{n_done}/{len(clips)}] 창 게이트 계산", flush=True)
        m = np.flatnonzero(d["clip"] == clip)
        try:
            g = gates_for_clip(clip, d["start"][m])
        except FileNotFoundError:
            continue
        for k, v in g.items():
            gates[k][m] = v
    print(f"창 {len(z0)}개, 유기 점수 단독:")
    cand = candidates_from_values(z0[valid], 60)
    def best(z):
        ok = [(t, r) for t, r in ((t, evaluate(clips_for(d, z, 2), t)) for t in cand) if r["recall"] >= TARGET]
        return min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"])) if ok else None
    b = best(z0)
    print("  ", f"thr {b[0]:.2f} 재현율 {b[1]['recall']:.3f} 오경보 {b[1]['fa_per_hour']:.2f}/시간" if b else "만족 없음")
    rows = []
    for n_done, (k, g) in enumerate(gates.items(), 1):
        print(f"[{n_done}/{len(gates)}] 조합 평가", flush=True)
        r = best(np.where(g, z0, -99.0))
        if r:
            rows.append((r[1]["fa_per_hour"], k, r[0], r[1]["recall"]))
    for fa, k, t, rc in sorted(rows)[:8]:
        print(f"  conf>={k[0]} N={k[1]} H={k[2]} | thr {t:.2f} 재현율 {rc:.3f} 오경보 {fa:.2f}/시간")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
