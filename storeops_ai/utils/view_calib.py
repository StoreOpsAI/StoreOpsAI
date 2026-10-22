"""자동 보정 보조: 빈 화면 색 통계 맞춤(Lab)과 걷는 사람으로 초점거리 추정. 연구 쪽 eval_photometric.apply_lab_match, focal_from_walk와 같은 식.

- 색: Reinhard, Ashikhmin, Gooch, Shirley, "Color Transfer between Images", IEEE CG&A 21(5):34-41, 2001 (검색으로 존재 확인, 본문 읽지 않음).
  Lab 채널마다 (값 - 우리 빈 화면 평균) / 우리 표준편차 x 학습 표준편차 + 학습 평균. 원 논문은 l-alpha-beta 공간, 여기는 OpenCV Lab(단순화, 근거 약함).
  통계는 학습 캐시와 같게 빈 화면을 224x224로 줄여 잰다(학습 기준값: config TRAIN_EMPTY_LAB_MEAN/STD, 봉인 장소 제외 6곳).
- 초점거리: Lv, Zhao, Nevatia, "Camera Calibration from Video of a Walking Human", IEEE TPAMI 28(9), 2006 (검색으로 존재 확인, 본문 읽지 않음).
  한 사람의 두 시각 어깨-어깨 선과 발목-발목 선의 교차점은 지평선 위에 있고, (수직 소실점 - 중심)·(지평선 점 - 중심) = -f^2.
  화면에 곧게 선 사람이 한 명일 때만 짝짓는다(추적 번호 없이). 현재는 기록만 하고 영상 변환에는 쓰지 않는다.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

LabStats = Tuple[np.ndarray, np.ndarray]
PAIR_SEC = (1.0, 3.0)   # 짝지을 시간 간격(연구: 3fps 표본 3~9개)
MIN_MOVE, MAX_DIST = 0.05, 20.0


def lab_stats(frames: Sequence[np.ndarray]) -> LabStats:
    """BGR 프레임들 -> Lab 채널별 (평균, 표준편차). 224x224로 줄여 잰다(학습 캐시와 같은 크기)."""
    px = np.concatenate([cv2.cvtColor(cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0,
                                      cv2.COLOR_BGR2LAB).reshape(-1, 3) for f in frames])
    return px.mean(0), px.std(0)


def lab_match(img: np.ndarray, src: LabStats, ref: LabStats) -> np.ndarray:
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab = (lab - src[0]) / np.maximum(src[1], 1e-3) * ref[1] + ref[0]
    lab[..., 0] = lab[..., 0].clip(0, 100)
    return (cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)


def horizon_points(samples: Sequence[Tuple[float, np.ndarray]], height: float) -> np.ndarray:
    """(시각, [어깨 x, y, 발목 x, y]) 목록 -> 짝마다 어깨 선과 발목 선의 교차점."""
    out = []
    for i, (ti, a) in enumerate(samples):
        for tj, b in samples[i + 1:]:
            if tj - ti > PAIR_SEC[1]:
                break
            if tj - ti < PAIR_SEC[0] or np.hypot(b[2] - a[2], b[3] - a[3]) < MIN_MOVE * height:
                continue
            x = np.cross(np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0]), np.cross([a[2], a[3], 1.0], [b[2], b[3], 1.0]))
            if abs(x[2]) > 1e-9:
                out.append(x[:2] / x[2])
    return np.array(out).reshape(-1, 2)


def focal_from(vp: Optional[np.ndarray], hpts: np.ndarray, w: int, h: int) -> dict:
    """수직 소실점(동차)과 지평선 점들로 f. 못 구하면 이유만."""
    if vp is None or abs(vp[2]) < 1e-9 * np.linalg.norm(vp[:2]):
        return {"focal_note": "수직 소실점 없음"}
    p = np.array([w / 2, h / 2])
    v = vp[:2] / vp[2] - p
    hp = hpts[np.linalg.norm(hpts - p, axis=1) < MAX_DIST * h] if len(hpts) else hpts
    d = (hp - p) @ (v / np.linalg.norm(v)) if len(hp) else np.zeros(0)
    if len(d) < 10 or np.median(d) >= 0:
        return {"focal_note": f"짝 {len(d)}개(최소 10, 한 사람만 화면에서 걸어야 함)"}
    f = float(np.sqrt(np.linalg.norm(v) * -np.median(d)))
    return {"focal_over_h": f / h, "hfov_deg": float(np.degrees(2 * np.arctan(w / 2 / f))), "pitch_down_deg": float(np.degrees(np.arctan(f / np.linalg.norm(v)))),
            "focal_pairs": int(len(d))}


def _self_check() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    s = lab_stats([img])
    assert np.abs(lab_match(img, s, s).astype(int) - img).max() <= 2                       # 같은 통계면 그대로
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    sat = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV)[..., 1].mean()
    faded = (img * 0.4 + gray * 0.6).astype(np.uint8)
    assert sat(lab_match(faded, lab_stats([faded]), s)) > sat(faded) * 1.5                 # 바랜 색은 학습 쪽 퍼짐으로 되살아난다
    W, H, f, pitch = 1920.0, 1080.0, 1.1 * 1080.0, np.radians(15)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    R = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    proj = lambda X: (lambda x: x[:2] / x[2])(K @ (R @ (X - np.array([0.0, -1.5, 0.0]))))
    samples = []
    for t in range(20):                                                                    # 한 사람이 0.5초마다 0.3m씩 비스듬히 걷는다
        foot = np.array([-2.0 + 0.25 * t, 0.0, 6.0 - 0.15 * t])
        samples.append((0.5 * t, np.r_[proj(foot + [0, -1.4, 0]), proj(foot)]))
    vp = K @ R @ np.array([0.0, 1.0, 0.0])                                                 # 세상 수직(아래) 방향의 소실점
    r = focal_from(vp, horizon_points(samples, H), int(W), int(H))
    assert abs(r["focal_over_h"] - 1.1) < 0.02 and abs(r["pitch_down_deg"] - 15) < 0.5, r
    print("ok", r)


if __name__ == "__main__":
    _self_check()
