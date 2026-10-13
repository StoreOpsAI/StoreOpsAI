"""전도 클립의 사람 박스·관절(YOLO-pose)을 클립마다 npz로 저장한다. 영상은 이 캐시를 만든 뒤 지워도 된다.

저장(output/lying_pose/<클립>.npz): frame(사람마다 프레임 번호), box(x1,y1,x2,y2), box_conf, kpt(17×3: x,y,관절 신뢰도), n_frames.
lying_manifest.json의 첫 학습 클립(--set first_pass) 또는 전체 train/validation을 처리하고, 이미 있는 캐시는 건너뛴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from config.config import POSE_IMGSZ, POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.video_sampling import is_sample
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.normal_manifest import NORMAL_MANIFEST_PATH

OUTPUT_DIR = APP_ROOT / "output" / "lying_pose"
PERSON_CONF = 0.25


def select_clips(manifest, which):
    if which == "normal":  # 매장이동 정상 클립(전도 없음). manifest는 무시하고 normal_manifest를 읽는다
        normal = json.loads(NORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
        return [c for c in normal["train"] if not c["quarantine"]]
    if which == "normal_validation":  # 매장이동 공식 validation(정상). 최종 시험용이라 설정 동결 뒤에만 쓴다
        return json.loads(NORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))["validation"]
    if which == "first_pass":
        return [c for c in manifest["train"] if c.get("first_pass")]
    if which == "validation":
        return manifest["validation"]
    return [c for c in manifest["train"] if not c["quarantine"]]


def extract_clip(model, clip, transform=None):
    frames, boxes, confs, kpts = [], [], [], []
    cap = cv2.VideoCapture(clip["video"])
    native_fps = cap.get(cv2.CAP_PROP_FPS)  # 3fps보다 빠른 영상은 정확히 3fps 표본만 쓴다. frame은 표본 번호
    n = raw = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        raw += 1
        if not is_sample(raw - 1, native_fps):
            continue
        image = transform(image) if transform else image  # 영상 증강(train 전용)이면 열화한 프레임으로 포즈를 뽑는다
        result = model.predict(image, imgsz=POSE_IMGSZ, conf=PERSON_CONF, half=True, verbose=False)[0]
        for box, conf, kpt in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.keypoints.data.cpu().numpy()):
            frames.append(n), boxes.append(box), confs.append(conf), kpts.append(kpt)
        n += 1
    cap.release()
    return {"frame": np.array(frames, np.int32), "box": np.array(boxes, np.float32).reshape(-1, 4),
            "box_conf": np.array(confs, np.float32), "kpt": np.array(kpts, np.float32).reshape(-1, 17, 3), "n_frames": np.int32(n), "native_fps": np.float32(native_fps)}


def main():
    parser = argparse.ArgumentParser(description="전도 클립 사람 박스·관절 캐시")
    parser.add_argument("--set", choices=["first_pass", "train", "validation", "normal", "normal_validation"], default="first_pass")
    parser.add_argument("--limit", type=int, default=0, help="시험용: 앞 N개만")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(APP_ROOT / POSE_WEIGHTS))
    clips = select_clips(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), args.set)
    clips = clips[:args.limit] if args.limit else clips
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for i, clip in enumerate(clips, 1):
        path = OUTPUT_DIR / f"{clip['clip']}.npz"
        if path.is_file():
            continue
        data = extract_clip(model, clip)
        assert data["n_frames"] == clip["frames"], f"프레임 수 불일치 {clip['clip']}: {data['n_frames']} != {clip['frames']}"
        np.savez(path, **data)
        print(f"{i}/{len(clips)} {clip['clip']} 사람 {len(data['frame'])}건 {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
