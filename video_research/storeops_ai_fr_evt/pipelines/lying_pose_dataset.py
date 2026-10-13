"""전도 사람 단위 샘플: 포즈 캐시(output/lying_pose) + 라벨(fall_start/fall_end) → 관절 특징·라벨.

라벨 규칙(3fps, 전환 구간 T=LYING_TRANSITION_FRAMES):
- 누움(1): fall_start+T 이후 프레임에서 라벨 전도 박스(fall_end 위치)와 IoU >= LYING_POSITIVE_IOU인 사람
- 정상(0): 첫 fall_start-T 이전의 모든 사람, 그리고 fall_start 이후에도 어떤 전도 박스와도 겹치지 않는(IoU < 0.1) 다른 사람(주변 사람)
- 그 밖(전환 구간, 애매하게 겹침)은 쓰지 않는다(-1)
같은 장면의 앞부분이 정상, 뒷부분이 누움이라 촬영 환경으로 라벨을 맞힐 수 없다.
"""

from __future__ import annotations

import json

import numpy as np

from config.config import LYING_POSITIVE_IOU, LYING_TRANSITION_FRAMES, POSE_MIN_CONF
from models.new_object_detector import _iou
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.lying_pose_extract import OUTPUT_DIR
from pipelines.lying_video_augment import AUG_DIR
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.normal_manifest import NORMAL_MANIFEST_PATH

KPT_CONF = 0.3
OTHER_IOU = 0.1


def pose_features(box, kpt):
    """관절을 사람 박스 중심·큰 변 기준으로 정규화한 특징(박스 위치는 넣지 않는다: 카메라 배치 편향 방지)."""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    scale, cx, cy = max(w, h), (x1 + x2) / 2, (y1 + y2) / 2
    ok = kpt[:, 2] >= KPT_CONF
    xy = np.where(ok[:, None], (kpt[:, :2] - [cx, cy]) / scale, 0.0)
    torso = np.zeros(3)
    if ok[[5, 6]].all() and ok[[11, 12]].all():  # 어깨·엉덩이가 모두 보일 때만 몸통 방향
        v = kpt[[5, 6], :2].mean(0) - kpt[[11, 12], :2].mean(0)
        norm = np.linalg.norm(v) + 1e-6
        torso = np.array([v[0] / norm, v[1] / norm, 1.0])
    return np.concatenate([xy.ravel(), kpt[:, 2], [h / w, w / h], torso]).astype(np.float32)


def person_label(frame, box, falls):
    starts = [f["start"] for f in falls]
    if frame <= min(starts) - LYING_TRANSITION_FRAMES:
        return 0
    if frame < min(starts) + LYING_TRANSITION_FRAMES:
        return -1
    overlaps = [_iou(box, f["box_end"]) for f in falls if frame >= f["start"] + LYING_TRANSITION_FRAMES]
    if overlaps and max(overlaps) >= LYING_POSITIVE_IOU:
        return 1
    return 0 if all(_iou(box, f["box_end"]) < OTHER_IOU and _iou(box, f["box_start"]) < OTHER_IOU for f in falls) else -1


def load_clip(clip, pose_dir=OUTPUT_DIR):
    data = np.load(pose_dir / f"{clip['clip']}.npz")
    feats = np.array([pose_features(b, k) for b, k in zip(data["box"], data["kpt"])]).reshape(len(data["box"]), -1)
    if clip["falls"]:
        labels = np.array([person_label(int(f), b, clip["falls"]) for f, b in zip(data["frame"], data["box"])], np.int8)
    else:  # 정상 데이터(매장이동): 클립 안의 모든 사람이 정상
        labels = np.zeros(len(data["box"]), np.int8)
    return {"frame": data["frame"], "box": data["box"], "kpt": data["kpt"], "feat": feats, "label": labels}


RETURN_MANIFEST_PATH = APP_ROOT / "data" / "return_manifest.json"
RETURN_POSE_DIR = APP_ROOT / "output" / "lying_pose_return"


def load_dataset(include_normal=False, include_aug=False, include_return=False):
    """first_pass 전도 클립(+ include_normal이면 매장이동 정상 클립). [{clip 정보, frame, box, feat, label}] (사람 단위 배열)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    clips = [c for c in manifest["train"] if c.get("first_pass")]
    if include_normal:
        clips += [c for c in json.loads(NORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))["train"] if not c["quarantine"]]
    ds = [{**clip, **load_clip(clip)} for clip in clips]
    if include_return:  # 구매행동 반품 정상 클립(정상 풀 확장). 촬영 날짜 그룹 fold는 return_manifest가 정한다
        ds += [{**c, **load_clip(c, RETURN_POSE_DIR)} for c in json.loads(RETURN_MANIFEST_PATH.read_text(encoding="utf-8")) if not c["quarantine"]]
    if include_aug:  # 영상 증강 클립(train 전용): 원본과 같은 fold, aug=True로 표시해 평가에서 뺀다
        ds += [{**clip, **load_clip(clip, AUG_DIR), "aug": True} for clip in clips if (AUG_DIR / f"{clip['clip']}.npz").is_file()]
    return ds


def load_validation():
    """공식 validation(전도 81 + 매장이동 정상 101). 최종 시험용이라 설정 동결 뒤 eval_lying_val만 부른다."""
    clips = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["validation"]
    clips += json.loads(NORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))["validation"]
    return [{**clip, **load_clip(clip)} for clip in clips]


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.stdout.reconfigure(encoding="utf-8")
    ds = load_dataset()
    labels = np.concatenate([c["label"] for c in ds])
    print("클립", len(ds), "사람-프레임", len(labels), dict(Counter(labels.tolist())), "특징 차원", ds[0]["feat"].shape[1])
    assert set(labels.tolist()) <= {-1, 0, 1} and (labels == 1).sum() > 0 and (labels == 0).sum() > 0
