"""버튼 한 번으로 하는 자동 보정(사용자 결정 2026-10-19: 어디서든 버튼을 누르면 값을 자동 계산한다. 손으로 정하는 값 없음).

순서: 빈 화면 empty_sec초(사람이 보이면 세지 않는다) -> 사람이 화면을 걸어 다니는 walk_sec초.
1) 밝기: 빈 화면 평균 밝기를 학습 장소 빈 화면 평균(TRAIN_EMPTY_LUMA)에 맞추는 감마. C1 크롭 입력에 건다.
2) 기울기(소실점): 서 있는 사람의 어깨 중점->발목 중점 선(세상에서 수직선)을 모아 수직 소실점을 RANSAC으로 찾는다.
   학습 CCTV의 소실점(TRAIN_VP)과 비교해, 사람 위치마다 "학습 카메라였다면 이만큼 기울어 보였을" 각도 차이를 구하고
   C1 크롭과 P 관절을 그 차이만큼 평면 회전한다. 카메라 높이·내려다보는 시점 자체(원근)는 회전으로 만들 수 없어 보정하지 않고 안내만 한다.
   걷는 동안 한 사람만 보이면 초점거리·내려다보는 각도도 구해 기록한다(utils/view_calib.py, Lv 외 2006). 영상 변환에는 아직 쓰지 않는다.
4) 색: 빈 화면 Lab 평균·표준편차를 학습 빈 화면 기준에 맞춘다(Reinhard 외 2001). C1 크롭 입력에 감마 대신 건다(색 통계가 있을 때).
3) 물건 판정 색 모델: 빈 화면 프레임 n_frames장(1/4 축소)으로 픽셀별 통계와 문턱을 정한다(utils/color_model.py, Horprasert 외 1999).
근거: 보행자 수직선으로 소실점을 구하는 보정(Huang 외 CVPR 2016 [v3 R9] 등)의 단순화. 효과는 검증하지 않았다(학습 기준과의 차이를 보여 주는 진단 + 보정).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.color_model import ColorModel, fit
from utils.view_calib import LabStats, focal_from, horizon_points, lab_match, lab_stats

KPT_CONF, MAX_BEND_DEG, MAX_TILT_DEG, MIN_LEN_FRAC = 0.5, 10.0, 30.0, 0.12   # vertical_vp.py(연구)와 같은 선 선택 기준
INLIER_DEG, RANSAC_ITERS, MIN_LINES, MIN_SPREAD = 3.0, 300, 30, 0.3
MAX_ROTATE_DEG = 20.0


def lines_from_persons(persons: Sequence[Tuple[np.ndarray, Sequence[float]]], height: float) -> np.ndarray:
    """곧게 선 사람의 [x_top, y_top, x_bot, y_bot] (어깨 중점 -> 발목 중점)."""
    out = []
    for kpt, _ in persons:
        k = np.asarray(kpt, np.float32)
        if (k[[5, 6, 11, 12, 15, 16], 2] < KPT_CONF).any():
            continue
        sh, hip, ank = k[[5, 6], :2].mean(0), k[[11, 12], :2].mean(0), k[[15, 16], :2].mean(0)
        a, b, c = hip - sh, ank - hip, ank - sh
        cosb = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6)
        if sh[1] < hip[1] < ank[1] and cosb > math.cos(math.radians(MAX_BEND_DEG)) and math.degrees(math.atan2(abs(c[0]), c[1])) < MAX_TILT_DEG \
                and np.linalg.norm(c) > MIN_LEN_FRAC * height:
            out.append([*sh, *ank])
    return np.asarray(out, np.float64).reshape(-1, 4)


def _angle_err(lines: np.ndarray, v: np.ndarray) -> np.ndarray:
    mid, d = (lines[:, :2] + lines[:, 2:]) / 2, lines[:, 2:] - lines[:, :2]
    to_v = v[:2] - v[2] * mid
    cos = np.abs((d * to_v).sum(1)) / np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(to_v, axis=1), 1e-9)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def fit_vp(lines: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    p1, p2 = np.c_[lines[:, :2], np.ones(len(lines))], np.c_[lines[:, 2:], np.ones(len(lines))]
    L = np.cross(p1, p2)
    L /= np.maximum(np.linalg.norm(L[:, :2], axis=1, keepdims=True), 1e-9)
    best, best_n = None, -1
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(L), 2, replace=False)
        v = np.cross(L[i], L[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        v /= np.linalg.norm(v)
        n = int((_angle_err(lines, v) < INLIER_DEG).sum())
        if n > best_n:
            best, best_n = v, n
    inl = _angle_err(lines, best) < INLIER_DEG
    _, vecs = np.linalg.eigh(L[inl].T @ L[inl])
    return vecs[:, 0], float(inl.mean())


def tilt_at(v: np.ndarray, x: float, y: float) -> float:
    """소실점 v(동차)에서 화면 점 (x, y)의 수직선이 기운 각도(도). 위쪽이 오른쪽으로 기울면 +."""
    dx, dy = v[0] - v[2] * x, v[1] - v[2] * y
    if dy < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(-dx, dy))


class Calibration:
    """보정값을 입력에 적용한다. 값이 없으면 아무것도 바꾸지 않는다."""

    def __init__(self, train_vp: Tuple[float, float], gamma: float = 1.0, vp: Optional[Sequence[float]] = None, frame_size: Optional[Tuple[int, int]] = None,
                 lab_src: Optional[LabStats] = None, lab_ref: Optional[LabStats] = None):
        self.train_vp, self.gamma, self.frame_size = train_vp, float(gamma), frame_size
        self.lab = (lab_src, lab_ref) if lab_src is not None and lab_ref is not None else None   # 있으면 감마 대신 색 통계 맞춤
        self.vp = None if vp is None else np.asarray(vp, np.float64)
        self._lut = (255.0 * (np.arange(256) / 255.0) ** self.gamma).clip(0, 255).astype(np.uint8)

    def delta_deg(self, x: float, y: float) -> float:
        """학습 카메라에서의 기울기 - 우리 카메라에서의 기울기. 소실점이 없으면 0."""
        if self.vp is None or self.frame_size is None:
            return 0.0
        w, h = self.frame_size
        tv = np.array([w / 2 + self.train_vp[0] * w, h / 2 + self.train_vp[1] * h, 1.0])
        return float(np.clip(tilt_at(tv, x, y) - tilt_at(self.vp, x, y), -MAX_ROTATE_DEG, MAX_ROTATE_DEG))

    def photometric(self, img: np.ndarray) -> np.ndarray:
        if self.lab is not None:
            return lab_match(img, *self.lab)
        return img if self.gamma == 1.0 else self._lut[img]

    @staticmethod
    def rotation(cx: float, cy: float, delta_deg: float) -> np.ndarray:
        """(cx, cy) 중심으로 기울기를 delta만큼(위쪽이 오른쪽으로 +) 바꾸는 2x3 변환. cv2 각도는 반시계가 +라 부호를 뒤집는다."""
        return cv2.getRotationMatrix2D((float(cx), float(cy)), -float(delta_deg), 1.0)

    def rotated_crop(self, frame: np.ndarray, x0: float, y0: float, side: float, delta_deg: float) -> np.ndarray:
        """정사각형 영역을 중심 기준으로 회전해 자른다. 영역 밖 원본 화소를 쓰므로 모서리가 비지 않는다(화면 가장자리는 복제)."""
        if side <= 0:
            return frame
        m = self.rotation(x0 + side / 2, y0 + side / 2, delta_deg)
        m[:, 2] -= (x0, y0)
        return cv2.warpAffine(frame, m, (int(side), int(side)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def rotate_keypoints(self, kpt: np.ndarray, box: Sequence[float]) -> np.ndarray:
        d = self.delta_deg((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if d == 0.0:
            return kpt
        m = self.rotation((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, d)
        out = np.array(kpt, np.float32, copy=True)
        out[:, :2] = out[:, :2] @ m[:, :2].T + m[:, 2]
        return out


class AutoCalibrator:
    """버튼(키)으로 시작해 빈 화면 -> 걷기 순서로 값을 모으고, 끝나면 Calibration과 결과 dict를 만든다."""

    def __init__(self, empty_sec: float, walk_sec: float, max_sec: float, train_luma: float, train_vp: Tuple[float, float],
                 train_vp_y_range: Tuple[float, float], save_path: Path, n_frames: int = 100, detection_rate: float = 0.9999,
                 dark_min: float = 0.4, train_lab: Optional[LabStats] = None):
        self.empty_sec, self.walk_sec, self.max_sec = empty_sec, walk_sec, max_sec
        self.n_frames, self.detection_rate, self.dark_min, self.train_lab = n_frames, detection_rate, dark_min, train_lab
        self.train_luma, self.train_vp, self.vp_range, self.save_path = train_luma, train_vp, train_vp_y_range, Path(save_path)
        self.state, self.message, self.result = "idle", "", None
        self.camera_settings: Optional[dict] = None   # 카메라에서 고정한 노출·화이트밸런스(camera_open.lock_camera). 보정 파일에 같이 저장
        self._reset()

    def _reset(self):
        self.t0 = self.t_prev = None
        self.empty_time, self.lumas, self.lines, self.walk_t0 = 0.0, [], [], None
        self.walk_pts: list = []             # (시각, 어깨·발목 선): 화면에 곧게 선 사람이 한 명일 때만(초점거리용)
        self.empty_small: list = []          # 빈 화면 1/4 축소 프레임(empty_sec 동안 고르게 n_frames장): 물건 판정 색 모델을 만든다
        self.next_small = 0.0                # 다음 프레임을 모을 빈 화면 누적 시각
        self.color_model: Optional[ColorModel] = None

    def start(self, t: float) -> None:
        self._reset()
        self.state, self.t0, self.t_prev = "empty", t, t
        self.message = f"보정 1/2: 화면을 비워 주세요 ({self.empty_sec:.0f}초)"

    def update(self, t: float, frame: np.ndarray, persons: Sequence[Tuple[np.ndarray, Sequence[float]]], n_people: int) -> Optional[Calibration]:
        """보정 중이면 값을 모은다. 끝나는 순간 Calibration을 돌려준다(그 외에는 None)."""
        if self.state not in ("empty", "walk"):
            return None
        dt, self.t_prev = t - self.t_prev, t
        if t - self.t0 > self.max_sec:
            if self.state == "walk":   # 선이 모자라도 빈 화면 보정(감마·색 모델)은 살리고 소실점 없이 끝낸다
                return self._finish(frame.shape[1], frame.shape[0])
            self.state, self.message = "failed", f"보정 실패: {self.max_sec:.0f}초 안에 끝나지 않았습니다. 다시 눌러 주세요"
            return None
        if self.state == "empty":
            if n_people:
                self.message = f"보정 1/2: 사람이 보입니다. 화면을 비워 주세요 ({self.empty_time:.0f}/{self.empty_sec:.0f}초)"
                return None
            self.empty_time += dt
            self.lumas.append(float(cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).mean()))
            if self.empty_time >= self.next_small:
                self.next_small += self.empty_sec / self.n_frames
                self.empty_small.append(cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4), interpolation=cv2.INTER_AREA))
            self.message = f"보정 1/2: 빈 화면 측정 중 ({self.empty_time:.0f}/{self.empty_sec:.0f}초)"
            if self.empty_time >= self.empty_sec:
                self.state, self.walk_t0 = "walk", t
            return None
        found = lines_from_persons(persons, frame.shape[0])
        self.lines.extend(found.tolist())
        if len(found) == 1 and n_people == 1:
            self.walk_pts.append((t, found[0]))
        self.message = f"보정 2/2: 전신이 보이게 화면 왼쪽~오른쪽을 걸어 주세요 ({t - self.walk_t0:.0f}/{self.walk_sec:.0f}초, 선 {len(self.lines)}/{MIN_LINES}개)"
        if t - self.walk_t0 < self.walk_sec or len(self.lines) < MIN_LINES:   # 소실점에 필요한 선이 모일 때까지 계속 걷기(최대 max_sec)
            return None
        return self._finish(frame.shape[1], frame.shape[0])

    def _finish(self, w: int, h: int) -> Calibration:
        luma = float(np.mean(self.lumas))
        gamma = float(np.clip(math.log(self.train_luma / 255.0) / math.log(min(max(luma, 1.0), 254.0) / 255.0), 0.5, 2.0))
        lines = np.asarray(self.lines, np.float64).reshape(-1, 4)
        vp, vp_y, note = None, None, ""
        spread = (np.ptp((lines[:, 0] + lines[:, 2]) / 2) / w) if len(lines) else 0.0
        if len(lines) >= MIN_LINES and spread >= MIN_SPREAD:
            v, frac = fit_vp(lines, np.random.default_rng(0))
            if abs(v[2]) > 1e-9 * np.linalg.norm(v[:2]):
                vp, vp_y = v, float((v[1] / v[2] - h / 2) / h)
                lo, hi = self.vp_range
                note = "학습 범위 안" if lo <= vp_y <= hi else ("학습보다 수평에 가까움(카메라를 더 아래로 숙이면 학습 조건에 가까워짐)" if vp_y > hi or vp_y < 0 else "학습보다 더 내려다봄")
            else:
                vp, vp_y, note = v, float("inf"), "선이 평행(카메라가 거의 수평). 학습보다 수평에 가까움"
        else:
            note = f"기울기 보정 생략: 선 {len(lines)}개, 좌우 폭 {spread:.2f} (최소 {MIN_LINES}개, {MIN_SPREAD})"
        self.result = {"gamma": gamma, "empty_luma": luma, "train_luma": self.train_luma, "lines": int(len(lines)), "spread": float(spread),
                       "vp": None if vp is None else [float(x) for x in vp], "vp_y_over_h": vp_y, "train_vp": list(self.train_vp),
                       "train_vp_y_range": list(self.vp_range), "frame_size": [w, h], "note": note,
                       **focal_from(vp, horizon_points(self.walk_pts, h), w, h)}
        if self.camera_settings:
            self.result["camera"] = self.camera_settings
        lab = lab_stats(self.empty_small) if self.empty_small else None
        if lab is not None:
            self.result["empty_lab"] = {"mean": lab[0].tolist(), "std": lab[1].tolist()}
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(json.dumps(self.result, ensure_ascii=False, indent=1), encoding="utf-8")
        if len(self.empty_small) >= 4:   # 5초 남짓 걸린다(270x480, 100장)
            self.color_model = fit(self.empty_small, self.detection_rate, self.dark_min)
            self.color_model.save(color_model_path(self.save_path))
            self.result["color_model"] = {"frames": len(self.empty_small), "tau_cd": self.color_model.tau_cd, "tau_a1": self.color_model.tau_a1,
                                          "tau_a2": self.color_model.tau_a2, "b_min": self.color_model.b_min}
            self.save_path.write_text(json.dumps(self.result, ensure_ascii=False, indent=1), encoding="utf-8")
        self.empty_small = []
        self.state = "done"
        self.message = f"보정 완료: 감마 {gamma:.2f}(밝기 {luma:.0f}->{self.train_luma:.0f}), 소실점 y={vp_y if vp_y is not None else '없음'} {note}"
        return Calibration(self.train_vp, gamma, vp, (w, h), lab, self.train_lab)


def color_model_path(path: Path) -> Path:
    return Path(path).with_suffix(".color.npz")


def load_color_model(path: Path) -> Optional[ColorModel]:
    """보정 때 저장한 물건 판정 색 모델(1/4 축소). 없으면 None(예전 .empty.png만 있으면 다시 보정해야 한다)."""
    return ColorModel.load(color_model_path(path))


def load_calibration(path: Path, train_vp: Tuple[float, float], train_lab: Optional[LabStats] = None) -> Optional[Calibration]:
    """저장된 보정값. 없으면 None(보정 없이 동작)."""
    path = Path(path)
    if not path.is_file():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    lab = r.get("empty_lab")   # 예전 보정 파일에는 없다(그때는 감마)
    src = (np.asarray(lab["mean"], np.float32), np.asarray(lab["std"], np.float32)) if lab else None
    return Calibration(train_vp, r["gamma"], r["vp"], tuple(r["frame_size"]), src, train_lab)


def _self_check() -> None:
    import tempfile
    W, H = 1920, 1080
    train_vp = (0.0, 1.54)
    tv = np.array([W / 2, H / 2 + 1.54 * H, 1.0])
    assert tilt_at(tv, W / 2, 500) == 0.0 and tilt_at(tv, 200, 500) < -10 and tilt_at(tv, 1700, 500) > 10   # 학습 카메라: 왼쪽은 왼쪽으로, 오른쪽은 오른쪽으로
    flat = np.array([0.0, 1.0, 0.0])                                                        # 평행(수평 카메라)
    assert abs(tilt_at(flat, 200, 500)) < 1e-9
    cal = Calibration(train_vp, 1.0, flat, (W, H))
    d = cal.delta_deg(600, 500)
    assert abs(d - tilt_at(tv, 600, 500)) < 1e-9 and d < 0 and cal.delta_deg(200, 500) == -MAX_ROTATE_DEG   # 큰 차이는 상한으로 자른다
    k = np.zeros((17, 3), np.float32)
    k[[5, 6], :2], k[[15, 16], :2], k[:, 2] = (600, 300), (600, 700), 1.0                    # 수직으로 선 사람
    r = cal.rotate_keypoints(k, (550, 280, 650, 720))
    top, bot = r[[5, 6], :2].mean(0), r[[15, 16], :2].mean(0)
    assert abs(math.degrees(math.atan2(top[0] - bot[0], bot[1] - top[1])) - d) < 1e-3       # 회전 후 기울기 = 학습 카메라의 그 위치 기울기
    img = np.zeros((H, W, 3), np.uint8)
    cv2.line(img, (960, 300), (960, 700), (255, 255, 255), 9)
    crop = cal.rotated_crop(img, 760, 300, 400, 15.0)
    ys, xs = np.nonzero(crop[..., 0] > 128)
    slope = np.polyfit(ys, xs, 1)[0]                                                        # x = a*y + b, 위쪽이 오른쪽이면 a < 0
    assert crop.shape == (400, 400, 3) and abs(math.degrees(math.atan(-slope)) - 15.0) < 2.0  # 크롭 회전 방향이 관절 회전과 같다
    g = Calibration(train_vp, 0.5)
    assert g.photometric(np.full((2, 2, 3), 64, np.uint8)).mean() > 64 and Calibration(train_vp).photometric(img) is img
    rng = np.random.default_rng(0)
    persons = []
    for x in np.linspace(200, 1700, 60):                                                    # 학습 카메라와 같은 소실점을 갖는 사람들
        bot = np.array([x, 800.0]); top = bot + (tv[:2] - bot) * -0.35
        kk = np.zeros((17, 3), np.float32); kk[:, 2] = 1.0
        kk[[5, 6], :2] = top; kk[[11, 12], :2] = (top + bot) / 2; kk[[15, 16], :2] = bot
        persons.append((kk, (x - 50, top[1], x + 50, bot[1])))
    out = Path(tempfile.mkdtemp()) / "cal.json"
    ac = AutoCalibrator(1.0, 1.0, 30.0, 138.5, train_vp, (1.19, 1.85), out, n_frames=10)
    ac.start(0.0)
    frame = np.full((H, W, 3), 100, np.uint8)
    noisy = lambda: np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
    t = 0.0
    while ac.state == "empty":
        t += 0.1; ac.update(t, noisy(), [], 1 if t < 0.5 else 0)                               # 처음 0.5초는 사람이 있어 세지 않는다
    assert t >= 1.35                                                                         # 사람이 있던 0.4초는 빼고 1초를 센다
    res = None
    while res is None:
        t += 0.1; res = ac.update(t, frame, persons, len(persons))
    assert ac.state == "done" and abs(ac.result["vp_y_over_h"] - 1.54) < 0.05 and ac.result["gamma"] < 1.0   # 밝기 100 -> 138.5로 밝히는 감마
    assert abs(res.delta_deg(300, 600)) < 0.5 and load_calibration(out, train_vp) is not None                # 학습 카메라와 같으면 회전이 거의 없다
    cm = load_color_model(out)
    assert "empty_lab" in ac.result and "focal_note" in ac.result                                             # 걷는 사람이 여러 명이면 초점거리는 못 구한다고 기록
    assert cm is not None and cm.mu.shape == (H // 4, W // 4, 3) and ac.result["color_model"]["frames"] == 10   # 1초에 고르게 10장(n_frames=10)
    ac, t, res = AutoCalibrator(1.0, 1.0, 5.0, 138.5, train_vp, (1.19, 1.85), out, n_frames=10), 0.0, None   # 걷는 동안 선이 하나도 없으면
    ac.start(0.0)
    while res is None and t < 10: t += 0.1; res = ac.update(t, noisy(), [], 0)
    assert res is not None and ac.state == "done" and t > 5.0 and ac.result["vp"] is None   # 걷기 1초에 안 끝나고 제한 시간까지 기다린 뒤 소실점 없이 저장
    print("ok")


if __name__ == "__main__":
    _self_check()
