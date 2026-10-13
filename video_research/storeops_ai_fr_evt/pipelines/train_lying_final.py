"""전도 최종 모델: 설정을 동결한 뒤 전도 first_pass 321 + 매장이동 정상 32클립 전체로 관절 MLP를 학습해 배포 폴더(analysis/)에 저장한다.

동결한 설정(config.py 사전 값): 관절 특징 56차원, MLP 128-64, 60에폭, 시드 0, 점수 임계값 0.5, 같은 자리(IoU 0.3) 2초 유지, 3fps.
공식 validation은 쓰지 않는다. dev 5-fold 결과(output/lying_eval/mlp_normal.json)를 학습 보고서에 함께 복사한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np

from config.config import ANALYSIS_DIR, LYING_HOLD_SEC, LYING_POSITIVE_IOU, POSE_IMGSZ, POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.eval_lying import labeled
from pipelines.lying_models import fit_mlp
from pipelines.lying_pose_dataset import KPT_CONF, load_dataset

PERSON_CONF = 0.25


def main():
    parser = argparse.ArgumentParser(description="전도 최종 모델 학습(관절 MLP)")
    parser.add_argument("--with-aug", action="store_true", help="영상 증강 클립(train 전용)을 학습에 추가")
    parser.add_argument("--dev-report", default="mlp_normal.json", help="학습 보고서에 복사할 dev 5-fold 결과(output/lying_eval)")
    args = parser.parse_args()
    out = (APP_ROOT / ANALYSIS_DIR).resolve()
    (out / "weights").mkdir(parents=True, exist_ok=True)
    old = out / "weights" / "lying_mlp.pt"
    if old.exists():  # 이전 가중치는 덮어쓰기 전에 output/에 보관한다(배포 폴더에는 새 것만 둔다)
        backup = APP_ROOT / "output" / "lying_weights_backup"
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, backup / f"lying_mlp_{int(old.stat().st_mtime)}.pt")
    ds = load_dataset(include_normal=True, include_aug=args.with_aug)
    x, y = labeled(ds, "feat")
    model = fit_mlp(x, y)
    model.save(out / "weights" / "lying_mlp.pt")
    shutil.copy2(APP_ROOT / POSE_WEIGHTS, out / "weights" / "yolo11s-pose.pt")
    (out / "label_map.json").write_text(json.dumps({"normal": 0, "fall": 1}, indent=2), encoding="utf-8")
    config = {"fps": 3.0, "pose_weights": "weights/yolo11s-pose.pt", "lying_weights": "weights/lying_mlp.pt", "pose_imgsz": POSE_IMGSZ,
              "person_conf": PERSON_CONF, "kpt_conf": KPT_CONF, "score_threshold": 0.5, "hold_sec": LYING_HOLD_SEC,
              "same_spot_iou": 0.3, "gap_samples": 1}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dev = json.loads((APP_ROOT / "output" / "lying_eval" / args.dev_report).read_text(encoding="utf-8"))
    report = {"video_augmentation": args.with_aug, "hold_sec": LYING_HOLD_SEC, "train_clips": len(ds), "fall_clips": sum(bool(c["falls"]) for c in ds if not c.get("aug")), "normal_clips": sum(not c["falls"] for c in ds if not c.get("aug")), "aug_clips": sum(bool(c.get("aug")) for c in ds),
              "person_frames": int(len(y)), "lying": int((y == 1).sum()), "normal": int((y == 0).sum()), "feature_dim": int(x.shape[1]),
              "dev_5fold": {k: dev[k] for k in ("person_dev", "train_dev_f1_gap", "clip_recall", "clips_hit", "fall_clips", "fp", "transition_alarms_not_judged")}}
    (out / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dev_5fold"}, ensure_ascii=False))
    assert np.isfinite(model(x[:10])).all()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
