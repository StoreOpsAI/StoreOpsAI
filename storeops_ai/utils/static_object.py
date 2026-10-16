"""새로 멈춘 물건 영역 검출 (유기 경보 확인용).

방법: Porikli, Ivanov, Haga, "Robust Abandoned Object Detection Using Dual Foregrounds",
EURASIP J. Adv. Signal Process. 2008, 197875. 근거·변경점은 행동인식_분석.md "새로 멈춘 물건" 절.
느린 배경(BL)과 빠른 배경(BS)을 따로 두고, FL=1 이면서 FS=0 인 픽셀(잠시 멈춘 물체)의 증거를 시간에 따라 쌓는다.
논문과 다른 점: 가우시안 혼합 모델 대신 OpenCV MOG2, 사람 박스 안은 증거에서 제외(멈춰 선 사람을 물건으로 보지 않는다).
놓인 것/없어진 것 구분: Tian, Feris, Liu, Hampapur, Sun, "Robust Detection of Abandoned and Removed Objects in Complex
Surveillance Videos", IEEE Trans. SMC-C 41(5), 2011. 영역 경계 안쪽을 씨앗으로 현재 화면과 오래된 배경에서 각각 영역을 키워
더 크게 자란 쪽(주변과 어울리는 쪽)으로 판정한다. 색 허용 오차와 "비슷함" 기준은 논문에 수치가 없어 설정값으로 둔다.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


class StaticObjectDetector:
    def __init__(self, scale: float, proc_fps: float, long_period_sec: float, long_alpha: float, ratio: float,
                 max_evidence: int, decay: float, min_area_frac: float, hold_sec: float, person_pad: float,
                 grow_tol: int = 10, similar_ratio: float = 1.5):
        self.scale, self.proc_dt, self.long_dt = scale, 1.0 / proc_fps, long_period_sec
        self.alpha_l, self.alpha_s = long_alpha, min(1.0, long_alpha * ratio)
        self.max_e, self.k, self.min_area_frac = float(max_evidence), decay, min_area_frac
        self.hold_sec, self.person_pad = hold_sec, person_pad
        self.grow_tol, self.similar_ratio = grow_tol, similar_ratio
        self.bg_long = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.bg_short = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.evidence: np.ndarray | None = None
        self.next_proc = self.next_long = 0.0
        self.last_time = {"abandoned": None, "removed": None}   # 종류별 마지막 확인 시각
        self.owner: dict[str, int | None] = {"abandoned": None, "removed": None}   # 그 물건 영역에 가장 가까웠던 손목의 사람 번호
        self.last_region: dict | None = None   # 가장 최근 물건 영역과 가장 가까운 손목(거리·반경·인정 여부). 반경을 정하는 측정용 기록
        self.boxes: list[tuple[int, int, int, int]] = []   # 최근 검출된 물건 영역(원본 좌표)

    def update(self, frame: np.ndarray, t: float, person_boxes: Sequence[Sequence[float]],
               trail_points: Sequence[tuple[int, float, float, float]] | None = None) -> None:
        """trail_points: 최근 손목 위치 (사람 번호, x, y, 반경 픽셀). None이면 주인 확인을 하지 않는다."""
        if t < self.next_proc:
            return
        self.next_proc = t + self.proc_dt
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        update_long = t >= self.next_long
        if update_long:
            self.next_long = t + self.long_dt
        fl = self.bg_long.apply(small, learningRate=self.alpha_l if update_long else 0) > 0
        fs = self.bg_short.apply(small, learningRate=self.alpha_s) > 0
        if self.evidence is None:
            self.evidence = np.zeros(fl.shape, np.float32)
        static = fl & ~fs
        h, w = fl.shape
        for x1, y1, x2, y2 in person_boxes:
            px, py = (x2 - x1) * self.person_pad, (y2 - y1) * self.person_pad
            a, b = max(0, int((x1 - px) * self.scale)), max(0, int((y1 - py) * self.scale))
            c, d = min(w, int((x2 + px) * self.scale) + 1), min(h, int((y2 + py) * self.scale) + 1)
            static[b:d, a:c] = False
        self.evidence = np.where(static, self.evidence + 1, self.evidence - self.k).clip(0, self.max_e)
        flagged = (self.evidence >= self.max_e).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(flagged)
        min_area = self.min_area_frac * h * w
        found = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= min_area]
        if found:
            background = self.bg_long.getBackgroundImage()
            for stat in found:
                kind = self._heal_type(flagged, stat, small, background)
                box = (int(stat[0] / self.scale), int(stat[1] / self.scale),
                       int((stat[0] + stat[2]) / self.scale), int((stat[1] + stat[3]) / self.scale))
                nearest = self._nearest(box, trail_points) if trail_points is not None else None
                owner = nearest[0] if nearest and nearest[1] <= nearest[2] else None
                if kind and trail_points is not None:
                    self.last_region = {"t": t, "kind": kind, "box": box, "tid": nearest[0] if nearest else None,
                                        "dist": nearest[1] if nearest else None, "r": nearest[2] if nearest else None, "ok": owner is not None}
                if kind and (trail_points is None or owner is not None):
                    self.last_time[kind], self.owner[kind] = t, owner
                    self.boxes = [box]

    @staticmethod
    def _nearest(box: tuple[int, int, int, int], points: Sequence[tuple[int, float, float, float]]) -> tuple[int, float, float] | None:
        """물건 영역 중심에서 가장 가까운 손목 (사람 번호, 거리, 그 사람의 반경). 거리가 반경 이하여야 주인으로 인정한다."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best = None
        for tid, x, y, r in points:
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if best is None or dist < best[1]:
                best = (tid, dist, r)
        return best

    def _grown_area(self, image: np.ndarray, seeds: np.ndarray) -> int:
        """씨앗에서 색이 이어지는 영역을 키워 넓이를 센다(Tian 외 2011의 영역 키우기)."""
        image = cv2.GaussianBlur(image, (5, 5), 0)   # 센서 잡음에 영역 키우기가 끊기지 않게 한다
        h, w = image.shape[:2]
        grown = np.zeros((h + 2, w + 2), np.uint8)
        tol = (self.grow_tol,) * 3
        for x, y in seeds:
            if not grown[y + 1, x + 1]:
                cv2.floodFill(image, grown, (int(x), int(y)), 0, tol, tol, cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        return int(np.count_nonzero(grown))

    def _heal_type(self, flagged: np.ndarray, stat, current: np.ndarray, background: np.ndarray | None) -> str | None:
        if background is None:
            return None
        x, y, w, h = int(stat[0]), int(stat[1]), int(stat[2]), int(stat[3])
        mask = np.zeros_like(flagged)
        mask[y:y + h, x:x + w] = flagged[y:y + h, x:x + w]
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        edge = cv2.morphologyEx(eroded, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(edge if edge.any() else mask)
        if len(xs) == 0:
            return None
        step = max(1, len(xs) // 16)
        seeds = np.stack([xs[::step], ys[::step]], axis=1)
        cur, bg = self._grown_area(current, seeds), self._grown_area(background, seeds)
        if bg > cur * self.similar_ratio:
            return "abandoned"   # 배경 쪽이 주변과 더 이어진다 = 현재 화면에 물건이 생겼다
        if cur > bg * self.similar_ratio:
            return "removed"     # 현재 화면 쪽이 더 이어진다 = 배경에 있던 물건이 없어졌다
        return None              # 논문의 unclear

    def recently_flagged(self, t: float, kind: str = "abandoned") -> bool:
        last = self.last_time[kind]
        return last is not None and t - last <= self.hold_sec


def _self_check() -> None:
    rng = np.random.default_rng(0)
    noise = lambda img: np.clip(img + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    empty = rng.integers(90, 110, (240, 320, 3), dtype=np.uint8)
    with_obj = empty.copy(); with_obj[100:140, 150:200] = 250
    make = lambda: StaticObjectDetector(1.0, 5.0, 5.0, 0.002, 30, max_evidence=20, decay=1, min_area_frac=0.001, hold_sec=4, person_pad=0.1)

    def run(det, start, end, boxes=()):
        t = 0.0
        for _ in range(400):   # 80초 학습
            det.update(noise(start), t, []); t += 0.2
        for _ in range(60):    # 12초 변화 유지
            det.update(noise(end), t, boxes); t += 0.2
        return t

    d = make(); t = run(d, empty, empty)
    assert not d.recently_flagged(t) and not d.recently_flagged(t, "removed")
    d = make(); t = run(d, empty, with_obj)                       # 물건이 놓임
    assert d.recently_flagged(t, "abandoned") and not d.recently_flagged(t, "removed") and d.boxes
    d = make(); t = run(d, with_obj, empty)                       # 물건이 없어짐
    assert d.recently_flagged(t, "removed") and not d.recently_flagged(t, "abandoned")
    d = make(); t = run(d, empty, with_obj, [(140, 90, 210, 150)])   # 사람 박스 안의 변화는 물건이 아니다
    assert not d.recently_flagged(t, "abandoned")
    near, far = [(7, 175.0, 120.0, 60.0)], [(7, 5.0, 5.0, 60.0)]
    for trail, expect in ((near, 7), (far, None), ([], None)):    # 놓은 사람의 손목이 근처에 없으면 유기가 아니다
        d = make(); t = 0.0
        for _ in range(400):
            d.update(noise(empty), t, [], trail); t += 0.2
        for _ in range(60):
            d.update(noise(with_obj), t, [], trail); t += 0.2
        assert d.recently_flagged(t, "abandoned") == (expect is not None) and d.owner["abandoned"] == expect, (trail, expect)
    print("ok")


if __name__ == "__main__":
    _self_check()
