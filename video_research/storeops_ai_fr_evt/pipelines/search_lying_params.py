"""전도 경보 파라미터를 OOF 탐색으로 정한다: 점수 기준(thr), 같은 자리 IoU(link_iou), 유지 시간(hold, 사용자 지정 후보 1~5초).

후보는 데이터에서 만든다: thr = OOF 사람 점수의 고유값/분위수, link_iou = 라벨 누운 사람의 연속 표본 간 박스 IoU 분포 분위수.
공식 validation·quarantine 미사용. 출력: output/lying_search/{threshold_search.csv, selected_params.json}
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from config.config import LYING_GRID_HOLDS, LYING_RECALL_TARGETS
from models.new_object_detector import _iou
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_pose_dataset import load_dataset
from pipelines.oof_search import candidates_from_values, search_params
from pipelines.event_metrics import event_stats
from pipelines.select_lying_operating_point import oof_clip_scores


def consecutive_ious(clips):
    """누운 사람(라벨 1)의 같은 클립 연속 표본 사이 박스 IoU (표본마다 가장 겹치는 박스)."""
    values = []
    for c in clips:
        by_frame = defaultdict(list)
        for f, box, y in zip(c["frame"], c["box"], c["label"]):
            if y == 1:
                by_frame[int(f)].append(box)
        order = sorted(by_frame)
        for a, b in zip(order, order[1:]):
            values += [max(_iou(x, y) for y in by_frame[b]) for x in by_frame[a]]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="mlp")
    parser.add_argument("--with-return", action="store_true", help="구매행동 반품 정상 풀을 포함(촬영 날짜 그룹 fold)")
    parser.add_argument("--tag", default="", help="출력 폴더 꼬리표(증강 설정 비교용). 증강 강도는 LYING_AUG_* 환경변수로 준다")
    parser.add_argument("--link-iou", type=float, default=0.0, help="0이 아니면 같은 자리 IoU를 이 값으로 고정(사용자 확정), 0이면 탐색")
    parser.add_argument("--max-thr", type=int, default=60, help="점수 후보 개수 상한(계산량 때문. 값은 점수 분위수)")
    args = parser.parse_args()
    clips = [c for c in load_dataset(include_normal=True, include_aug=False, include_return=args.with_return)]
    out_dir = APP_ROOT / "output" / (("lying_search_return" if args.with_return else "lying_search") + args.tag)
    scores = oof_clip_scores(clips, args.method)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "oof_scores.npz", **scores)  # 탐색을 다시 돌릴 때 학습을 반복하지 않도록
    pos = np.concatenate([scores[c["clip"]][c["label"] == 1] for c in clips])
    neg = np.concatenate([scores[c["clip"]][c["label"] == 0] for c in clips])
    floor = np.quantile(pos, 0.01)  # 누운 사람 점수 하위 1% 아래에서는 재현율이 변하지 않는다
    thr_space = candidates_from_values(np.concatenate([pos, neg[neg >= floor]]), args.max_thr)
    iou_space = [args.link_iou] if args.link_iou else candidates_from_values(consecutive_ious(clips), 8)

    def evaluate(thr, link_iou, hold):
        return event_stats(clips, scores, thr, hold, link_iou)

    from pipelines.calibration import fit_temperature, reliability  # 점수가 로짓일 때 확률 보정(dev OOF 전용)
    people = np.concatenate([np.stack([scores[c['clip']], c['label']], 1) for c in clips]); people = people[people[:, 1] >= 0]
    temp = fit_temperature(people[:, 0], people[:, 1])
    cal = {"method": "temperature_scaling", "temperature": temp, **reliability(1 / (1 + np.exp(-people[:, 0] / temp)), people[:, 1])}
    meta = {"calibration": cal, "score_mode": __import__("os").environ.get("LYING_SCORE", "prob"), "event": "fall", "method": args.method, "hold_candidates_by_user": list(LYING_GRID_HOLDS), "clips": len(clips), "with_return_pool": args.with_return, "aug_env": {k: v for k, v in __import__("os").environ.items() if k.startswith("LYING_AUG")},
            "candidate_source": {"thr": "OOF scores of lying persons + non-lying persons above lying-score 1st percentile (quantiles)", "link_iou": "label lying persons consecutive-sample IoU quantiles"}}
    report = search_params(evaluate, {"thr": thr_space, "link_iou": iou_space, "hold": list(LYING_GRID_HOLDS)}, LYING_RECALL_TARGETS, out_dir, meta)
    for target, best in report["selected_by_recall_target"].items():
        print(f"재현율 >= {target}:", "만족 조합 없음" if best is None else
              f"thr {best['thr']:.4f} link_iou {best['link_iou']:.3f} 유지 {best['hold']:g}초 -> 재현율 {best['recall']:.3f} ({best['hits']}/{best['fall_clips']}) 정밀도 {best['precision']:.3f} F1 {best['f1']:.3f} 지연 {best['delay_median_sec']:.1f}초, 오경보 {best['false_alarms']}건/{best['normal_minutes']}분 = {best['false_alarm_per_hour']:.2f}/시간")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
