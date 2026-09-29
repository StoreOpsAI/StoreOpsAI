from pathlib import Path
import subprocess
import cv2


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
    try:
        # OpenCV 기본 mp4v는 브라우저에서 재생되지 않을 수 있어 H.264로 표준화한다.
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ],
            check=True,
        )
        temporary_path.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 로컬 개발 환경에 FFmpeg가 없으면 OpenCV 결과를 그대로 보존한다.
        temporary_path.replace(path)
    return path


def representative_indices(n):
    if n <= 0: return []
    if n == 1: return [0, 0, 0]
    if n == 2: return [0, 1, 1]
    return [0, n // 2, n - 1]


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
