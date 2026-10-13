"""파손 = 물건을 들고(집기) → 던지고(손목 속도 급증) → 바닥에 떨어짐(새 물건). 근거 조합별 재현율·오경보.

- 던지기: YOLO-pose 손목 속도(eval_broken_pose.wrist_speeds) >= BROKEN_THROW_SPEED. 3초 안의 던지기는 하나로 묶는다
- 떨어짐: 던진 뒤 BROKEN_AFTER_SEC 안에 새로 놓인 물건(added)이 생김 (eval_broken_evidence 프레임 캐시)
- 집기: 던지기 전후 BROKEN_PICK_SEC 안에 원래 물건이 치워진 자리(removed)가 생김
- 적중: 던지기 시각이 파손 사건 [시작-2초, 종료+2초] 안. 오경보: 사건 전후 여유(MARGIN) 밖 경보. 전환 구간은 세지 않음
- 음성: 파손 영상의 사건 밖 구간, 반품, 유기·절도·방화 train. 공식 validation 미사용
임계값은 라벨 관절 분석(8-1절)에서 정한 값으로 고정하고 여기서 고르지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from config.config import BROKEN_AFTER_SEC, BROKEN_PICK_SEC, BROKEN_THROW_SPEED
from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT, parse_annotation
from pipelines.eval_broken_evidence import MARGIN_AFTER_SEC, MARGIN_BEFORE_SEC
from pipelines.eval_broken_pose import load_pose_tracks, wrist_speeds
from pipelines.prepare_behaviors import list_videos

RULES = {
    "throw": lambda fell, picked: True,
    "throw+fell1": lambda fell, picked: fell >= 1,
    "throw+fell2": lambda fell, picked: fell >= 2,
    "throw+(pick|fell1)": lambda fell, picked: picked or fell >= 1,
    "throw+pick+fell1": lambda fell, picked: picked and fell >= 1,
}


def throws(pose_tracks, fps):
    frames = sorted(f for track in pose_tracks.values() for f, s in wrist_speeds(track) if s >= BROKEN_THROW_SPEED)
    merged = []
    for f in frames:
        if not merged or f - merged[-1] > 3 * fps:
            merged.append(f)
    return merged


def video_alarms(video, fps):
    """{규칙: 경보 프레임 목록}."""
    motion, added, removed, _ = json.loads(
        (APP_ROOT / "output" / "broken_evidence" / "frames" / f"{Path(video).stem}.json").read_text(encoding="utf-8"))
    alarms = {name: [] for name in RULES}
    for t in throws(load_pose_tracks(video, APP_ROOT / "output" / "broken_pose"), fps):
        fell = sum(t <= f <= t + BROKEN_AFTER_SEC * fps for f in added)
        picked = any(abs(f - t) <= BROKEN_PICK_SEC * fps for f in removed)
        for name, rule in RULES.items():
            if rule(fell, picked):
                alarms[name].append(t)
    return alarms, len(motion)


def main():
    parser = argparse.ArgumentParser(description="파손 집기·던지기·떨어짐 평가")
    parser.add_argument("--output", type=Path, default=APP_ROOT / "output" / "broken_throw.json")
    args = parser.parse_args()
    broken = [(mp4, parse_annotation(xml, "broken"))
              for xml, mp4, split in list_videos(WORKSPACE_ROOT / "파손", "파손")[0] if split == "train"]
    negatives = {"반품": [(mp4, []) for mp4 in sorted((WORKSPACE_ROOT / "구매행동" / "videos").glob("*.mp4"))]}
    for name in ("유기", "절도", "방화"):
        negatives[name] = [(mp4, []) for _, mp4, split in list_videos(WORKSPACE_ROOT / name, name)[0] if split == "train"]

    summary = {}
    for source, videos in {"파손": broken, **negatives}.items():
        stats = {name: {"events": 0, "matched": 0, "false": 0} for name in RULES}
        minutes = 0.0
        for video, events in videos:
            cache = APP_ROOT / "output" / "broken_evidence" / "frames" / f"{Path(video).stem}.json"
            if not cache.is_file():
                continue  # 새 물건 캐시가 없는 영상(eval_broken_evidence 미실행)은 건너뛴다
            fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 3.0
            alarms, frames = video_alarms(video, fps)
            hit_spans = [(e["start_frame"] - 2 * fps, e["end_frame"] + 2 * fps) for e in events]
            margins = [(e["start_frame"] - MARGIN_BEFORE_SEC * fps, e["end_frame"] + MARGIN_AFTER_SEC * fps) for e in events]
            minutes += sum(not any(a <= f <= b for a, b in margins) for f in range(frames)) / fps / 60
            for name, times in alarms.items():
                stats[name]["events"] += len(events)
                stats[name]["matched"] += sum(any(a <= t <= b for t in times) for a, b in hit_spans)
                stats[name]["false"] += sum(not any(a <= t <= b for a, b in margins) for t in times)
        summary[source] = {"minutes": round(minutes, 1), **{
            name: {"recall": round(s["matched"] / s["events"], 2) if s["events"] else None,
                   "false_alarms": s["false"], "fp_per_min": round(s["false"] / minutes, 3) if minutes else None}
            for name, s in stats.items()}}
        print(source, json.dumps(summary[source], ensure_ascii=False))
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
