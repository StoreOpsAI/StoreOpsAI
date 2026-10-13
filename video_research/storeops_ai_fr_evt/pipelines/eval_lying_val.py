"""전도 최종 평가: 동결한 모델(analysis/weights/lying_mlp.pt)을 공식 validation에서 **딱 한 번** 평가한다.

- 전도 val 81클립 + 매장이동 val 101클립(정상). 학습·설정 선택에 쓰지 않았다(AGENTS.md 규칙 5·7)
- 결과 파일(analysis/val_result.json)이 이미 있으면 다시 돌리지 않는다. 결과를 보고 설정·임계값·데이터를 고치면 안 된다
- 기록: 클래스별 precision·recall·F1, 혼동행렬, train-validation F1 차이, 클립 단위 적중·오경보(전도 영상 정상 / 매장이동 정상)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from config.config import ANALYSIS_DIR, LYING_HOLD_SEC as HOLD_SEC, LYING_TRANSITION_FRAMES
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_lying import FPS, clip_alarms, judge_alarms, labeled
from pipelines.lying_models import LyingMLP
from pipelines.lying_pose_dataset import load_dataset, load_validation
from pipelines.train_abandonment import _metrics


def main():
    parser = argparse.ArgumentParser(description="전도 최종 모델 공식 validation 평가")
    parser.add_argument("--tag", default="", help="결과 파일 꼬리표. 비우면 1회차(val_result.json). 꼬리표가 있으면 validation 재사용으로 표시한다")
    parser.add_argument("--weights", default="lying_mlp.pt", help="analysis/weights 안의 가중치 파일")
    args = parser.parse_args()
    out = (APP_ROOT / ANALYSIS_DIR).resolve() / f"val_result{args.tag}.json"
    if out.exists():
        raise SystemExit(f"이미 평가했습니다({out}). 같은 설정·꼬리표로 다시 돌리지 않는다.")
    model = LyingMLP.load(out.parent / "weights" / args.weights)
    val = load_validation()
    _, y = labeled(val, "feat")
    scores = np.concatenate([model(c["feat"])[c["label"] >= 0] for c in val])
    person = _metrics(y.tolist(), scores.tolist())
    train = load_dataset(include_normal=True)
    xt, yt = labeled(train, "feat")
    train_person = _metrics(yt.tolist(), model(xt).tolist())
    hits = fall_clips = transition = 0
    false = {"전도 영상 정상": 0, "매장이동 정상": 0}
    minutes = {"전도 영상 정상": 0.0, "매장이동 정상": 0.0}
    for c in val:
        hit, n_false, n_transition = judge_alarms(c, clip_alarms(c, model(c["feat"])))
        domain = "전도 영상 정상" if c["falls"] else "매장이동 정상"
        fall_clips += bool(c["falls"])
        hits, transition, false[domain] = hits + hit, transition + n_transition, false[domain] + n_false
        minutes[domain] += (max(min(f["start"] for f in c["falls"]) - LYING_TRANSITION_FRAMES, 0) if c["falls"] else c["frames"]) / FPS / 60
    result = {"note": "공식 validation 재사용(2회차 이상): 이전 평가 결과를 본 뒤 설정을 바꾼 모델의 평가이므로 독립 시험이 아니다" if args.tag else "공식 validation 1회 평가",
              "weights": args.weights, "hold_sec": HOLD_SEC, "person_val": person, "train_person": {k: train_person[k] for k in ("precision", "recall", "f1")},
              "train_val_f1_gap": round(train_person["f1"] - person["f1"], 3), "fall_clips": fall_clips, "normal_clips": len(val) - fall_clips,
              "clips_hit": int(hits), "clip_recall": round(hits / fall_clips, 3),
              "fp": {d: {"false_alarms": false[d], "minutes": round(minutes[d], 1), "fp_per_min": round(false[d] / minutes[d], 3) if minutes[d] else None} for d in false},
              "transition_alarms_not_judged": transition}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
