from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np


def save_frames(frames, path, fps):
    if not frames:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    temporary_path = path.with_suffix(".opencv.mp4")
    writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {path}")
    try:
        for frame in frames:
            # 사람 crop처럼 프레임마다 크기가 다르면 VideoWriter가 프레임을 조용히 버린다. 첫 프레임 크기로 맞춘다.
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        writer.release()
    encoded_path = path.with_suffix(".h264.mp4")
    try:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError("H.264 MP4 saving requires ffmpeg or imageio-ffmpeg") from exc
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded_path)],
            check=True,
        )
        encoded_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
        encoded_path.unlink(missing_ok=True)
    return path


def representative_indices(n):
    if n <= 0: return []
    if n == 1: return [0, 0, 0]
    if n == 2: return [0, 1, 1]
    return [0, n // 2, n - 1]


def save_all_images(frames, out_dir, stem, limit=0):
    """VLM 확인용: 클립 프레임을 시간 순서대로 전부(limit>0이면 고른 간격으로 limit장) JPEG로 저장한다."""
    if not frames:
        raise ValueError("VLM 확인용 프레임이 없습니다.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = range(len(frames)) if limit <= 0 or len(frames) <= limit else np.linspace(0, len(frames) - 1, limit).round().astype(int)
    paths = []
    for i, idx in enumerate(idxs, 1):
        p = out_dir / f"{stem}_vlm_{i:02d}.jpg"
        cv2.imwrite(str(p), frames[int(idx)])
        paths.append(str(p))
    return paths


def save_representative_images(frames, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = representative_indices(len(frames))
    if not idxs:
        raise ValueError("FR-EVT-13을 위해 대표 이미지 3장이 필요합니다.")
    paths = []
    for i, idx in enumerate(idxs, 1):
        p = out_dir / f"{stem}_{i}.jpg"
        cv2.imwrite(str(p), frames[idx])
        paths.append(str(p))
    return paths
