from pathlib import Path
import cv2


def save_frames(frames, path, fps):
    if not frames:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def representative_indices(n):
    if n <= 0: return []
    if n == 1: return [0]
    if n == 2: return [0, 1]
    return [0, n // 2, n - 1]


def save_representative_images(frames, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = representative_indices(len(frames))
    if len(idxs) < 3:
        raise ValueError("FR-EVT-13을 위해 대표 이미지 3장이 필요합니다.")
    paths = []
    for i, idx in enumerate(idxs, 1):
        p = out_dir / f"{stem}_{i}.jpg"
        cv2.imwrite(str(p), frames[idx])
        paths.append(str(p))
    return paths
