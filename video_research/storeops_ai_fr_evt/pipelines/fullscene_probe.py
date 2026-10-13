"""화면 전체 4초 창 S3D 특징(얼림)으로 전도 선형 OOF 점검: 미세조정 전에 신호가 있는지 본다.

데이터: 전 행동 공통 분할표의 전도 헤드 학습 가능 클립(격리 제외), 날짜 fold 5개. 라벨: 전도 창 1/0(애매한 -1 제외), 다른 행동·정상 풀 창은 전부 0.
오경보 예산별 재현율(정상 창 시간당)과 AP를 낸다. 공식 validation은 읽지 않는다.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines.compare_backbones import _fit_linear
from pipelines.fullscene_windows import window_label
from pipelines.global_split import SPLIT_PATH
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.window_head_oof import BUDGETS_PER_HOUR, average_precision, oof_scores, recall_at_budgets

STRIDE_HOURS = 2.0 / 3600  # 창 간격 2초


def load() -> dict[str, np.ndarray]:
    falls = {c["clip"]: c["falls"] for s in ("train", "validation") for c in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))[s]}
    parts = {k: [] for k in ("x", "y", "fold", "behavior", "group")}
    for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8")):
        p = Path(BEHAVIOR_FEATURE_ROOT) / "fullscene" / f"{r['clip']}.npz"
        if r["official_split"] != "train" or r["q_fall"] == "True" or not p.is_file():
            continue
        z = np.load(p)
        y = z["label"].astype(int) if r["behavior"] == "전도" else np.zeros(len(z["start"]), int)
        keep = y >= 0
        parts["x"].append(z["s3d"][keep]), parts["y"].append(y[keep])
        parts["fold"].append(np.full(int(keep.sum()), int(r["fold"]))), parts["behavior"].append(np.array([r["behavior"]] * int(keep.sum())))
        parts["group"].append(np.array([f"{r['site']}_{r['date']}"] * int(keep.sum())))
    return {k: np.concatenate(v) for k, v in parts.items()}


def main() -> None:
    d = load()
    x, y, fold = d["x"].astype(np.float32), d["y"].astype(np.int64), d["fold"]
    print("창", len(y), "양성", int(y.sum()), "행동별", dict(Counter(d["behavior"])), "fold별", dict(sorted(Counter(fold.tolist()).items())))
    s = oof_scores(x, y, fold, _fit_linear)
    neg_hours = (y == 0).sum() * STRIDE_HOURS
    print(f"AP {average_precision(y, s):.3f} | 정상 창 시간 {neg_hours:.1f}시간(창 간격 2초 기준)")
    for b, v in recall_at_budgets(y, s).items():
        # recall_at_budgets는 창 시간을 4초(ABANDON_WIN_STRIDE)로 환산하므로 여기서는 창 수로 직접 표시한다
        print(f"  오경보 창 {v['false_windows']}개 허용 -> 재현율 {v['recall']:.3f} (예산 {b}/시간 기준 환산 값은 참고용)")
    for beh in ("전도", "유기", "절도", "반품", "매장이동"):
        m = d["behavior"] == beh
        if beh != "전도" and m.any():
            thr = recall_at_budgets(y, s)["1.0"]["thr"]
            print(f"  {beh} 창 중 점수가 기준(1.0/시간)을 넘는 비율: {(s[m] > thr).mean():.4f}")
    out = Path("output/window_head"); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "fullscene_fall_oof.npz", score=s, y=y, fold=fold, behavior=d["behavior"], group=d["group"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
