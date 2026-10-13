"""시연 점검: 서비스 CLI(path1_behavior)에 우리 체크포인트를 꽂아 영상을 돌리고, 발급된 사건을 정답 행동과 나란히 보여 준다.

처음 보는 영상(그 체크포인트를 학습하지 않은 fold)만 쓴다. 서비스 output/에 사건 파일이 생기므로 끝난 뒤 새로 생긴 파일을 알려 준다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipelines.aihub_abandonment import APP_ROOT
from pipelines.fullscene_windows import video_path

SERVICE = APP_ROOT.parent / "storeops_ai"
WEIGHT = SERVICE / "training" / "runs" / "ours_a_fold0" / "best.pt"
# OOF(A안, 재현율 0.9, 다른 행동 사건 제외)에서 정한 로그오즈 기준 -> 서비스가 쓰는 확률로 환산
THRESHOLDS = {"FALL_EVENT_THRESHOLD": 6.23, "LITTERING_EVENT_THRESHOLD": 4.70, "THEFT_EVENT_THRESHOLD": 2.15}
PICK = {"전도": 3, "유기": 3, "절도": 3, "매장이동": 2}


def pick_clips() -> list[tuple[str, str]]:
    z = np.load(APP_ROOT / "output" / "finetune_s3d" / "oof_fold0.npz")
    by: dict[str, list[str]] = {}
    for c, b in zip(z["clip"], z["behavior"]):
        by.setdefault(str(b), [])
        if str(c) not in by[str(b)]:
            by[str(b)].append(str(c))
    out = []
    for b, n in PICK.items():
        names = sorted(by[b])
        out += [(b, names[i]) for i in np.linspace(0, len(names) - 1, n).astype(int)]
    return out


def main() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **{k: str(1 / (1 + np.exp(-v))) for k, v in THRESHOLDS.items()}}
    before = {p for p in (SERVICE / "output").rglob("*") if p.is_file()}
    seen = set((SERVICE / "output" / "events").glob("E*.json"))  # 이번 실행 전에 있던 사건 파일은 제외
    for b, clip in pick_clips():
        video = next(p for s in ("train", "validation") if (p := video_path(clip, b, s)).is_file())
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pipelines.path1_behavior", "--video", str(video), "--i3d-weight", str(WEIGHT)],
                           cwd=SERVICE, env=env, capture_output=True, text=True, encoding="utf-8")
        events = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((SERVICE / "output" / "events").glob("E*.json")) if p not in seen]
        seen |= set((SERVICE / "output" / "events").glob("E*.json"))
        got = [(e["category"], round(e["clip_start_time_sec"] or 0), round(e["confidence"], 4)) for e in events]
        print(f"[정답 {b}] {clip[-34:]} -> 사건 {len(events)}건 {got}" + ("" if r.returncode == 0 else f" (종료코드 {r.returncode}: {r.stderr[-150:].strip()})"), flush=True)
    new = sorted({str(p.relative_to(SERVICE)) for p in (SERVICE / "output").rglob("*") if p.is_file()} - {str(p.relative_to(SERVICE)) for p in before})
    print(f"\n서비스 output에 새로 생긴 파일 {len(new)}개 (정리 대상):", new[:6], "...")
    Path(APP_ROOT / "output" / "demo_new_files.json").write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
