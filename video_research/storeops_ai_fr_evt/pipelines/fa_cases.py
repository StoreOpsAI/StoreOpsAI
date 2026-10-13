"""오경보 사례 시트: 평가에서 오경보로 센 경보 구간을 점수 높은 순으로 뽑아, 구간 한가운데 창의 크롭 프레임(4장)을 한 줄씩 붙인 사진 시트를 만든다.

사용: FA_CLASS=2(유기)/3(절도)/1(전도) FA_N=사례 수 python -m pipelines.fa_cases. A안 OOF 5개 fold, 재현율>=0.9가 되는 기준에서.
출력: output/fa_cases/<행동>_cases.csv, <행동>_sheet_NN.jpg (한 장 6줄). 사진은 사람 중심 크롭(crops_c1)에서 가져온다.
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_finetune import CLASSES, evaluate, load, per_clip, runs
from pipelines.oof_search import candidates_from_values

OUT = APP_ROOT / "output" / "fa_cases"
ROWS_PER_SHEET, TILE = 6, 224


def main() -> None:
    cls, n = int(os.getenv("FA_CLASS", "2")), int(os.getenv("FA_N", "18"))
    d = load()
    clips = per_clip(d, cls)
    starts = {c: np.sort(d["start"][d["clip"] == c]) for c in np.unique(d["clip"])}
    beh = {c: b for c, b in zip(d["clip"], d["behavior"])}
    z_all = np.concatenate([x["z"][x["y"] != -1] for x in clips])
    cand = candidates_from_values(z_all, 120)
    ok = [(t, evaluate(clips, t)) for t in cand]
    t, _ = min(((t, r) for t, r in ok if r["recall"] >= 0.9), key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"]))
    cases = []
    for x in clips:
        for a, e in runs(x["z"], t):
            if not (x["y"][a:e + 1] != 0).any():
                peak = a + int(np.argmax(x["z"][a:e + 1]))
                cases.append({"clip": x["clip"], "behavior": beh[x["clip"]], "start": int(starts[x["clip"]][peak]), "len_windows": e - a + 1, "score": float(x["z"][peak])})
    cases.sort(key=lambda c: -c["score"])
    OUT.mkdir(parents=True, exist_ok=True)
    name = CLASSES[cls]
    print(f"[{name}] 기준 {t:.2f}, 오경보 구간 {len(cases)}건. 영상 종류별:", {b: sum(c['behavior'] == b for c in cases) for b in sorted({c['behavior'] for c in cases})})
    # 영상 종류가 한쪽으로 쏠리지 않게 종류별 상위를 돌려 가며 고른다
    picked, pools = [], {b: [c for c in cases if c["behavior"] == b] for b in sorted({c["behavior"] for c in cases})}
    while len(picked) < min(n, len(cases)):
        for b in pools:
            if pools[b] and len(picked) < n:
                picked.append(pools[b].pop(0))
    for i, c in enumerate(picked):
        c["id"] = f"F{i:02d}"
    with open(OUT / f"{name}_cases.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(picked[0]))
        w.writeheader(), w.writerows(picked)
    for s in range(0, len(picked), ROWS_PER_SHEET):
        rows = []
        for c in picked[s:s + ROWS_PER_SHEET]:
            data = pickle.loads((Path(BEHAVIOR_FEATURE_ROOT) / "crops_c1" / f"{c['clip']}.pkl").read_bytes())
            jp = data[c["start"]]
            tiles = [cv2.imdecode(np.frombuffer(jp[k], np.uint8), cv2.IMREAD_COLOR) for k in (0, 4, 8, 11)]
            label = f"{c['id']} {c['behavior']} t={c['start'] / 3:.0f}s s={c['score']:.1f}"
            cv2.putText(tiles[0], label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            rows.append(np.hstack(tiles))
        ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        (OUT / f"{name}_sheet_{s // ROWS_PER_SHEET:02d}.jpg").write_bytes(buf.tobytes())
    print(f"저장: {OUT} (사례 {len(picked)}개)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
