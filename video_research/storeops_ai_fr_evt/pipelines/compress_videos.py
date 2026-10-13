"""유기·절도 원본 영상을 H.264(CRF 23)로 재인코딩해 용량을 줄인다. 해상도·fps·프레임 수는 그대로 둔다(작은 물건 탐지를 위해 해상도 유지).

안전장치: 임시 파일로 인코딩 -> 프레임 수·해상도·fps가 원본과 같을 때만 원본 교체(os.replace). 실패하면 원본을 그대로 둔다.
이미 압축한 파일은 마커 목록(compressed.txt)으로 건너뛴다(이어받기). 캐시 작업이 같은 영상을 읽는 동안 돌리지 않는다.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg

from config.config import AIHUB_ORGANIZED_ROOT

CRF = 23  # 사용자 요청(압축). 시험: 절도 41.5MB -> 12.9MB(31%), 해상도·프레임 수 동일


def info(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    out = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS), 2), int(cap.get(3)), int(cap.get(4)))
    cap.release()
    return out


def compress(path: Path, ffmpeg: str) -> tuple[bool, int, int]:
    tmp = path.with_suffix(".tmp.mp4")
    before = path.stat().st_size
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-an", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode or not tmp.is_file() or info(tmp) != info(path) or tmp.stat().st_size >= before:
        tmp.unlink(missing_ok=True)
        return False, before, before
    after = tmp.stat().st_size
    os.replace(tmp, path)
    return True, before, after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors", nargs="+", default=["유기", "절도"])
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    done_file = Path(AIHUB_ORGANIZED_ROOT) / f"compressed_{i}of{n}.txt"
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.is_file() else set()
    videos = sorted(v for b in args.behaviors for s in ("train", "validation") for v in (Path(AIHUB_ORGANIZED_ROOT) / b / s / "videos").glob("*.mp4") if not v.name.endswith(".tmp.mp4"))
    saved = total = 0
    for k, v in enumerate(videos[i::n], 1):
        if str(v) in done:
            continue
        ok, before, after = compress(v, ffmpeg)
        if ok:
            saved, total = saved + before - after, total + 1
            with open(done_file, "a", encoding="utf-8") as fh:
                fh.write(str(v) + "\n")
        else:
            print("건너뜀(검증 실패 또는 이득 없음):", v.name, flush=True)
        if k % 20 == 0:
            print(f"[{k}] 압축 {total}개, 절약 {saved / 1e9:.2f}GB", flush=True)
    print(f"COMPRESS_DONE 압축 {total}개, 절약 {saved / 1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
