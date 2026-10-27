"""새로 멈춘 물건 영역 검출 (유기 경보 확인용).

방법: Porikli, Ivanov, Haga, "Robust Abandoned Object Detection Using Dual Foregrounds",
EURASIP J. Adv. Signal Process. 2008, 197875. 근거·변경점은 행동인식_분석.md "새로 멈춘 물건" 절.
느린 배경(BL)과 빠른 배경(BS)을 따로 두고, FL=1 이면서 FS=0 인 픽셀(잠시 멈춘 물체)의 증거를 시간에 따라 쌓는다.
논문과 다른 점: 가우시안 혼합 모델 대신 OpenCV MOG2, 사람 박스 안은 증거에서 제외(멈춰 선 사람을 물건으로 보지 않는다).
놓인 것/없어진 것 구분: Tian, Feris, Liu, Hampapur, Sun, "Robust Detection of Abandoned and Removed Objects in Complex
Surveillance Videos", IEEE Trans. SMC-C 41(5), 2011. 영역 경계 안쪽을 씨앗으로 현재 화면과 오래된 배경에서 각각 영역을 키워
더 크게 자란 쪽(주변과 어울리는 쪽)으로 판정한다. 색 허용 오차와 "비슷함" 기준은 논문에 수치가 없어 설정값으로 둔다.
그림자: MOG2의 그림자 판정(배경보다 일정 비율 어둡고 색은 비슷한 픽셀, 마스크 값 127)을 켜고 그림자 픽셀은 전경에서 뺀다(255만 전경).
그림자 검출 방법 비교 근거: Prati, Mikic, Trivedi, Cucchiara, "Detecting Moving Shadows: Algorithms and Evaluation", IEEE TPAMI 25(7), 2003
(본문 읽음, 실내는 통계 방식이 믿을 만하다고 결론).
빈 화면 기준(자동 보정 후): Horprasert, Harwood, Davis, ICCV'99 FRAME-RATE Workshop의 통계 색 모델(utils/color_model.py, 본문 읽음).
픽셀별 정규화 밝기·색 왜곡과 빈 화면에서 자동으로 고른 문턱으로 물건/배경/그림자/빛 얼룩을 나누고, 물건만 전경으로 쓴다(빛 얼룩 상한 없음, 논문대로).
보정 전(MOG2 경로)은 그림자만 빼고 빛 얼룩은 빼지 않는다.
빛 변화·장면 변화 따라가기(보정 후 경로, Tian, Feris, Liu, Hampapur, Sun 2011 본문 읽음 + Tian, Lu, Hampapur CVPR 2005 본문 읽음):
- 빠른 빛 변화: 색이 다르다고 나온 화소라도 배경과 기울기 무늬가 같으면(S(X) >= 0.7, Tian 2005 식 5, Sobel, 문턱 논문값) 뺀다.
  논문에 창 크기 M x N 수치가 없어 설정값(근거 약함). 두 화면 모두 무늬가 없는 곳(잡음 이하)은 판단하지 않고 색 판정을 둔다.
- 느린 빛 변화: 사람·움직임·물건 후보가 아닌 화소만 빈 화면 평균 mu를 학습률로 따라간다(Stauffer-Grimson 갱신, Tian 2011은
  alpha=0.005·시각 기준 갱신). 시간 상수(초)로 바꾼 값은 논문에 없음(근거 약함). 잡음 sigma·a·b와 문턱은 보정값 그대로 둔다.
- 흡수(healing): 물건 영역이 heal_sec 동안 계속 확정 상태면 그 자리 mu를 현재 화면으로 바꾼다(Tian 2011 "push the static region
  to the background"). 이후 그 물건을 가져가면 배경과 달라진 자리가 '없어짐'으로 잡힌다. 논문은 영역이 가장 클 때 바로 흡수하고
  경보는 따로 추적하지만, 우리는 경보·주인 판정(hold_sec)이 끝난 뒤에 흡수한다. 논문의 배경 이력(덮인 자리 저장) 비교는 생략(영역 키우기로 판정).
한계: 색이 없는(흰·회·검) 물건이 비슷한 무채색 바닥에 놓이면 그림자·빛과 구분되지 않아 빠질 수 있다(색 모델 방식의 공통 한계).
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import cv2
import numpy as np

from utils.color_model import ColorModel, fit
from utils.hand_box import box_gap

TEXTURE_MIN_ENERGY = 48.0   # 화소당 |g|^2+|gb|^2 하한 = 잡음 sigma 1일 때 Sobel 두 축·두 화면 기대값(12 x 4). 근거 약함


class StaticObjectDetector:
    def __init__(self, scale: float, proc_fps: float, long_period_sec: float, long_alpha: float, ratio: float,
                 max_evidence: int, decay: float, min_area_frac: float, hold_sec: float,
                 grow_tol: int = 10, similar_ratio: float = 1.5, detect_shadows: bool = True,
                 bg_tau_sec: float = 0.0, heal_sec: float = 0.0, texture_sim: float = 0.0, texture_win: int = 5,
                 edge_ratio: float = 0.0):
        self.scale, self.proc_dt, self.long_dt = scale, 1.0 / proc_fps, long_period_sec
        self.alpha_l, self.alpha_s = long_alpha, min(1.0, long_alpha * ratio)
        self.max_e, self.k, self.min_area_frac = float(max_evidence), decay, min_area_frac
        self.hold_sec = hold_sec
        self.grow_tol, self.similar_ratio, self.edge_ratio = grow_tol, similar_ratio, edge_ratio   # edge_ratio 0이면 윤곽선 보조 판정 끔
        self.bg_long = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=detect_shadows)
        self.bg_short = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=detect_shadows)
        self.evidence: np.ndarray | None = None
        self.next_proc = self.next_long = 0.0
        self.last_time = {"abandoned": None, "removed": None}   # 종류별 마지막 확인 시각
        self.owner: dict[str, int | None] = {"abandoned": None, "removed": None}   # 그 물건 영역에 가장 가까웠던 손목의 사람 번호
        self.last_region: dict | None = None   # 가장 최근 물건 영역과 가장 가까운 손목(거리·반경·인정 여부). 반경을 정하는 측정용 기록
        self.boxes: list[tuple[int, int, int, int]] = []   # 최근 검출된 물건 영역(원본 좌표)
        self.kind_box: dict[str, tuple[int, int, int, int]] = {}   # 종류별 마지막 확인 영역(같은 자리 모순 판정 검사용)
        self._absorbed: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []   # absorb 기록(상자, 화소, 덮기 전 기준값)
        self.detect_shadows = detect_shadows
        self.bg_tau, self.heal_sec, self.texture_sim, self.texture_win = bg_tau_sec, heal_sec, texture_sim, texture_win   # 0이면 끔
        self.flag_age: np.ndarray | None = None   # 화소별 확정 상태가 이어진 시간(흡수용)
        self.cover_tid: np.ndarray | None = None   # 화소별 마지막으로 덮은 사람 번호(손·몸 상자). 주인 역추적용(Lin 외 TIFS 2015 back-tracing)
        self.cover_step: np.ndarray | None = None  # 그 처리 번호(클수록 최근)
        self.cover_t: np.ndarray | None = None     # 그 처리 시각(초). 손·몸이 그 자리를 마지막으로 덮은 시각(손 기록이 없을 때 VLM 클립 기준)
        self.hand_tid: np.ndarray | None = None    # 손 상자만: 화소별 마지막으로 닿은 사람 번호와 시각. 놓기·집기 순간 = VLM 클립 기준
        self.hand_t: np.ndarray | None = None      # (몸은 놓은 뒤 옆에 서 있거나 떠날 때도 덮어 시각이 늦어진다, 2026-10-27 시연 E005)
        self.touch_time: dict[str, float | None] = {"abandoned": None, "removed": None}   # 종류별 주인이 마지막으로 덮은 시각
        self.full_step: np.ndarray | None = None   # 화소별 증거가 가득 찬 처리 번호(Lin 2015의 t, 역추적 기준 시점)
        self._step = 0
        self._taken = None   # 마지막으로 가져간 흡수 물건 (상자, 화소, 물건 모습)
        self.model: ColorModel | None = None   # 보정 때 만든 빈 화면 색 모델(축소). 있으면 느린 배경 대신 이것과 대조한다
        self._recent: deque = deque(maxlen=max(1, int(round(proc_fps))))   # 약 1초 전 화면(움직임 판정용)

    def set_model(self, model: ColorModel, frame_shape: tuple[int, ...]) -> bool:
        """빈 화면 색 모델을 고정한다(자동 보정 버튼). 이후 '새 물건'은 기준에 없던 것, '없어진 물건'은 기준에 있던 것이다.
        모델 크기가 이 카메라 해상도 x scale과 다르면 쓰지 않고 False(다시 보정해야 한다).
        ponytail: 기준은 다시 보정할 때까지 고정이라 의자 이동 등 장면이 바뀌면 그 자리가 계속 잡힌다. 다시 보정하면 풀린다."""
        h, w = round(frame_shape[0] * self.scale), round(frame_shape[1] * self.scale)   # update()의 cv2.resize(fx=scale)와 같은 크기
        if model.mu.shape[:2] != (h, w):
            return False
        self.model = model
        self.evidence = self.flag_age = self.cover_tid = self.cover_step = self.cover_t = self.full_step = None
        self.hand_tid = self.hand_t = None
        self.touch_time = {"abandoned": None, "removed": None}
        self.last_time = {"abandoned": None, "removed": None}
        self.owner = {"abandoned": None, "removed": None}
        self._recent.clear()
        self._absorbed = []
        self._taken = None
        return True

    def update(self, frame: np.ndarray, t: float, person_boxes: Sequence[Sequence[float]],
               trail_points: Sequence[tuple[int, float, float, float, float]] | None = None, n_hands: int = 0) -> None:
        """trail_points: 이번 프레임의 주인 후보 상자 (사람 번호, x1, y1, x2, y2; 손 상자 + 사람 박스, utils.hand_box.reach_boxes).
        None이면 주인 확인을 하지 않는다."""
        if t < self.next_proc:
            return
        self.next_proc += self.proc_dt          # 평균 간격을 지킨다(t + dt로 두면 입력이 처리 속도와 비슷할 때 흔들림 때문에 절반을 건너뛴다)
        if self.next_proc <= t:                 # 한 간격 넘게 밀렸으면 몰아서 처리하지 않고 지금부터 다시 센다
            self.next_proc = t + self.proc_dt
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        update_long = t >= self.next_long
        if update_long:
            self.next_long = t + self.long_dt
        if self.model is not None:
            fl = self.model.classify(small)[0]   # 빈 화면 기준과 색이 다르거나 너무 어두움(그림자·빛 얼룩·노출 변화는 빠진다)
            if self.texture_sim > 0:
                fl &= ~self._same_texture(small)   # 색은 바뀌었어도 무늬가 같으면 빛 변화
            # 지금 움직이는 것 = 약 1초 전 화면과 다름(프레임 차분). 짧은 배경 모델은 놓인 물건을 흡수할 때까지 십수 초가 걸려
            # 손목 기록(당시 20초)이 지난 뒤에야 물건이 잡혔다(합성 시험: 10초에 놓은 물건을 29.6초에 잡음).
            # 논문 밖 확장: 같은 빈 화면 잡음 모델로 두 장을 비교한다. 두 장 모두 잡음이 있어 차이의 잡음은 sqrt(2)배.
            past = self._recent[0] if self._recent else small
            fs = self.model.classify(small, expected=past, noise_gain=np.sqrt(2.0))[0]
            self._recent.append(small)
        else:
            fl = self.bg_long.apply(small, learningRate=self.alpha_l if update_long else 0) == 255   # 127 = 그림자, 전경이 아니다
            fs = self.bg_short.apply(small, learningRate=self.alpha_s) == 255   # 지금 움직이는 것(멈춘 것만 물건 후보)
        if self.evidence is None:
            self.evidence = np.zeros(fl.shape, np.float32)
        h, w = fl.shape
        people = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in person_boxes:
            people[max(0, int(y1 * self.scale)):int(y2 * self.scale) + 1, max(0, int(x1 * self.scale)):int(x2 * self.scale) + 1] = True
        self._step += 1
        if trail_points is not None:
            self._paint_cover(trail_points, fs, t, n_hands)
        # Lin 외 TIFS 2015: 사람 영역을 미리 지우지 않고 멈춘 전경 전체에서 정지 후보를 뽑은 뒤, 사람인 후보를 사람 검출로 거른다(아래 영역 단계)
        static = fl & ~fs
        self.evidence = np.where(static, self.evidence + 1, self.evidence - self.k).clip(0, self.max_e)
        if self.full_step is None or self.full_step.shape != fl.shape:
            self.full_step = np.full(fl.shape, -1, np.int64)
        self.full_step = np.where(self.evidence >= self.max_e, np.where(self.full_step < 0, self._step, self.full_step), -1)
        if self.model is not None:
            self._maintain(small, fl | fs | people, fl & ~people, people)
        flagged = (self.evidence >= self.max_e).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(flagged)
        min_area = self.min_area_frac * h * w
        found = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= min_area]
        self.debug_regions = []   # 진단용: 이번 처리에서 잡힌 영역마다 크기·판정·손목 거리
        # 진단용: 단계별 마스크(기준과 다름 / 움직임 / 멈춘 후보 / 증거)를 한 장으로 남긴다
        self.debug_view = np.hstack([small, *(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in (
            fl.astype(np.uint8) * 255, fs.astype(np.uint8) * 255, static.astype(np.uint8) * 255,
            (self.evidence / self.max_e * 255).astype(np.uint8)))])
        self.debug_counts = (int(fl.sum()), int(fs.sum()), int(static.sum()), int(flagged.sum()), len(found))
        if found:
            background = self.model.background_image() if self.model is not None else self.bg_long.getBackgroundImage()
            for label in (i for i in range(1, count) if stats[i][cv2.CC_STAT_AREA] >= min_area):
                stat = stats[label]
                box = (int(stat[0] / self.scale), int(stat[1] / self.scale),
                       int((stat[0] + stat[2]) / self.scale), int((stat[1] + stat[3]) / self.scale))
                if any(self._same_spot(box, tuple(p)) for p in person_boxes):
                    continue   # 사람인 정지 후보(가만히 선 사람 등)는 버린다(Lin 2015: 사람 검출기로 사람 모양 정지 전경을 거름)
                kind = self._heal_type(flagged, stat, small, background)
                nearest = self._nearest(box, trail_points) if trail_points is not None else None   # 진단 출력용(지금 가장 가까운 후보)
                owner, touch = self._last_cover(labels == label) if trail_points is not None else (None, None)   # 그 자리를 마지막으로 덮은 사람·시각
                self.debug_regions.append({"t": t, "box": box, "area": int(stat[cv2.CC_STAT_AREA]), "kind": kind or "unclear",
                                           "wrists": None if trail_points is None else len(trail_points),
                                           "hand_gap": None if not nearest else round(nearest[1], 1)})
                if trail_points is not None:   # unclear도 남긴다(진단: 왜 경보로 안 이어졌는지)
                    self.last_region = {"t": t, "kind": kind or "unclear", "heal": getattr(self, "last_heal", None), "box": box, "tid": nearest[0] if nearest else None,
                                        "dist": nearest[1] if nearest else None, "r": nearest[2] if nearest else None, "ok": owner is not None}
                if kind == "removed":
                    self._restore_absorbed(box)   # 흡수했던 물건을 가져가면 기준을 빈 화면으로 되돌린다. 없어짐 판정은 그대로 둔다(MVP 시연: 가져가면 절도, 사용자 결정 2026-10-21)
                other = {"abandoned": "removed", "removed": "abandoned"}.get(kind)
                if kind and self.recently_flagged(t, other) and self._same_spot(box, self.kind_box.get(other)):
                    # 같은 자리가 놓임과 없어짐으로 번갈아 판정됨 = 판정 불가(Tian 2011은 영역마다 한 종류). 둘 다 지워 경보하지 않는다(시연 run16 E002·E003, E005·E006 동시 경보)
                    self.last_time[kind] = self.last_time[other] = None
                    self.owner[kind] = self.owner[other] = None
                    continue
                if kind and (trail_points is None or owner is not None):
                    self.last_time[kind], self.owner[kind], self.touch_time[kind] = t, owner, touch
                    self.boxes = [box]
                    self.kind_box[kind] = box

    def _same_texture(self, img: np.ndarray) -> np.ndarray:
        """Tian 외 2005 식 5: S = sum 2 g.gb / sum(|g|^2 + |gb|^2) (창 안 합). S >= 문턱이면 무늬가 같다(빛 변화)."""
        def grad(x):
            g = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        (gx, gy), (bx, by) = grad(img), grad(self.model.mu)
        k = (self.texture_win, self.texture_win)
        num = cv2.boxFilter(2 * (gx * bx + gy * by), -1, k, normalize=False)
        den = cv2.boxFilter(gx * gx + gy * gy + bx * bx + by * by, -1, k, normalize=False)
        textured = den > TEXTURE_MIN_ENERGY * self.texture_win ** 2   # 둘 다 무늬 없음(평평한 바닥 위 평평한 물건)은 판단 안 함
        return textured & (num >= self.texture_sim * den)

    def _paint_cover(self, boxes: Sequence[tuple[int, float, float, float, float]], moving: np.ndarray, t: float = 0.0, n_hands: int = 0) -> None:
        """이번 처리에서 손·몸 상자 안의 움직이는 화소(그 사람의 전경)에 사람 번호와 처리 번호를 남긴다.
        Lin 2015는 물건 자리의 사람 전경 덩어리를 주인으로 본다. 상자만 겹치고 몸이 그 자리를 지나지 않았거나 가만히 선 경우는 남기지 않는다."""
        if self.cover_tid is None or self.cover_tid.shape != moving.shape:
            self.cover_tid, self.cover_step = np.full(moving.shape, -1, np.int32), np.full(moving.shape, -1, np.int64)
            self.cover_t = np.full(moving.shape, np.nan, np.float64)
            self.hand_tid, self.hand_t = np.full(moving.shape, -1, np.int32), np.full(moving.shape, np.nan, np.float64)
        for i, (tid, x1, y1, x2, y2) in enumerate(boxes):   # 앞 n_hands개는 손 상자(reach_boxes 순서)
            ys = slice(max(0, int(y1 * self.scale)), max(0, int(y2 * self.scale) + 1))
            xs = slice(max(0, int(x1 * self.scale)), max(0, int(x2 * self.scale) + 1))
            hit = moving[ys, xs]
            self.cover_tid[ys, xs][hit], self.cover_step[ys, xs][hit], self.cover_t[ys, xs][hit] = tid, self._step, t
            if i < n_hands:
                self.hand_tid[ys, xs][hit], self.hand_t[ys, xs][hit] = tid, t

    def _last_cover(self, mask: np.ndarray) -> tuple[int | None, float | None]:
        """물건이 멈춘 시점 근처에 그 자리를 덮은(움직인) 사람 번호. 없으면 None(주인 없음 = 경보 안 함).
        Lin 외 TIFS 2015 back-tracing: 정지 판정 시점 t에서 t0 = t - Ts(Ts = 정지 판정에 걸리는 시간)로 돌아가 그 근처 창에서 주인을 찾는다.
        우리 Ts = 증거가 차는 처리 수(max_e). 창은 t0 앞뒤 Ts로 둔다(손이 덮은 마지막 움직임은 t0 직전이라 논문의 [t0, t0+δ]를 앞으로 넓힘, δ=Ts는 근거 약함).
        몇 분 전 지나간 사람이 주인이 되던 문제(run27) 대응. ponytail: 멈춘 뒤 그 자리를 가리고 지나가 증거가 다시 차면 그 사람이 주인(논문도 같은 한계, 막으려면 색 추적)."""
        if self.cover_step is None or self.full_step is None:
            return None, None
        near = (self.full_step >= 0) & (np.abs(self.cover_step - (self.full_step - self.max_e)) <= self.max_e)
        steps = np.where(mask & near, self.cover_step, -1)
        i = int(np.argmax(steps))
        if steps.flat[i] < 0:
            return None, None
        owner = int(self.cover_tid.flat[i])
        # 클립 기준: 그 주인 손이 그 자리에 마지막으로 닿은 시각(놓기·집기). 손 기록이 없으면(손목 안 보임) 몸까지 센 시각
        # ponytail: 같은 주인 번호의 오래된 손 기록도 센다. 번호가 다른 사람에게 재사용되면 예전 시각이 잡힐 수 있다
        hand = np.where(mask & (self.hand_tid == owner), self.hand_t, np.nan) if self.hand_t is not None else None
        touch = float(np.nanmax(hand)) if hand is not None and not np.isnan(hand).all() else float(self.cover_t.flat[i])
        return owner, touch

    def _maintain(self, small: np.ndarray, busy: np.ndarray, changed: np.ndarray, people: np.ndarray) -> None:
        """빈 화면 평균을 빛 변화에 맞춰 천천히 갱신하고, 오래 확정된 물건 영역은 배경으로 흡수한다(Tian 외 2011)."""
        mu, cur = self.model.mu, small.astype(np.float32)
        if self.bg_tau > 0:
            quiet = ~busy
            mu[quiet] += (1.0 - np.exp(-self.proc_dt / self.bg_tau)) * (cur[quiet] - mu[quiet])
        if self.heal_sec > 0:
            full = (self.evidence >= self.max_e) & ~people   # 멈춰 있는 사람은 배경으로 흡수하지 않는다
            self.flag_age = np.where(full, (self.flag_age if self.flag_age is not None else 0) + self.proc_dt, 0.0)
            aged = self.flag_age >= self.heal_sec
            if aged.any():   # 화소별이 아니라 이어진 영역 전체를 흡수한다(논문: 조각 남김 방지)
                _, lab = cv2.connectedComponents((changed | aged).astype(np.uint8))
                heal = np.isin(lab, np.unique(lab[aged]))
                mu[heal] = np.mean(self._recent, axis=0, dtype=np.float32)[heal]   # 약 1초 평균(한 장이면 그 잡음이 기준에 박힌다)
                self.evidence[heal] = 0
                self.flag_age[heal] = 0

    @staticmethod
    def _nearest(box: tuple[int, int, int, int], points: Sequence[tuple[int, float, float, float, float]]) -> tuple[int, float, float] | None:
        """물건 영역과 가장 가까운 주인 후보 상자 (사람 번호, 빈 거리, 반 변). 후보 = 손 상자(OpenPose, Simon 외 CVPR 2017)와 여백 붙인 사람 박스
        (utils.hand_box.reach_boxes). 빈 거리 0(닿음)이어야 주인으로 인정한다."""
        best = None
        for tid, x1, y1, x2, y2 in points:
            dist = box_gap((x1, y1, x2, y2), box)
            if best is None or dist < best[1]:
                best = (tid, dist, (x2 - x1) / 2)
        return best

    def _grown_area(self, image: np.ndarray, seeds: np.ndarray) -> int:
        """씨앗에서 색이 이어지는 영역을 키워 넓이를 센다(Tian 외 2011의 영역 키우기).
        씨앗 색에서 grow_tol 안인 화소만 자란다(고정 범위). IBM 특허 US9129402(같은 저자들의 A/R 판정)의 "같은 색 범위" 기준.
        이전에는 이웃끼리 비교(유동 범위)라 흐림(5x5)·압축으로 부드러워진 물건 경계를 타고 바닥 전체로 번져 현재·배경 둘 다 화면
        대부분을 채워 늘 '애매'였다(2026-10-20 시연 실측 12.4만/13만 화소)."""
        image = cv2.GaussianBlur(image, (5, 5), 0)   # 센서 잡음에 영역 키우기가 끊기지 않게 한다
        h, w = image.shape[:2]
        grown = np.zeros((h + 2, w + 2), np.uint8)
        tol = (self.grow_tol,) * 3
        for x, y in seeds:
            if not grown[y + 1, x + 1]:
                cv2.floodFill(image, grown, (int(x), int(y)), 0, tol, tol, cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
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
        self.last_heal = (cur, bg)   # 진단용: 현재·배경 쪽 영역 키우기 넓이
        if bg > cur * self.similar_ratio:
            return "abandoned"   # 배경 쪽이 주변과 더 이어진다 = 현재 화면에 물건이 생겼다
        if cur > bg * self.similar_ratio:
            return "removed"     # 현재 화면 쪽이 더 이어진다 = 배경에 있던 물건이 없어졌다
        return self._edge_type(mask, current, background)   # 영역 키우기가 애매(물건과 배경 색이 비슷해 번짐이 샘)일 때만

    def _edge_type(self, mask: np.ndarray, current: np.ndarray, background: np.ndarray) -> str | None:
        """윤곽선 에너지 비교(Connell 외 2004, Tian 외 2011 2절): 영역 경계에서 현재 화면의 윤곽선이 더 강하면 놓임, 약하면 없어짐.
        단순한 배경에서 잘 되고 어지러운 배경에서는 안 된다(Tian 2011). 영역 키우기와 서로 보완(IBM 특허 US8483481).
        조명만 바뀌면 윤곽선 위치·세기가 비슷해 애매로 남는다. 비율 edge_ratio는 논문에 수치 없음(측정으로 재검토)."""
        if self.edge_ratio <= 0:
            return None
        band = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0   # 영역 경계 안팎 한 화소
        if not band.any():
            return None
        def energy(img):
            g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))[band].mean())
        cur, bg = energy(current), energy(background)
        self.last_edge = (cur, bg)   # 진단용
        if cur > bg * self.edge_ratio:
            return "abandoned"
        if bg > cur * self.edge_ratio:
            return "removed"
        return None              # 논문의 unclear

    def absorb(self, box: tuple[int, int, int, int], kind: str = "abandoned") -> None:
        """MVP용(2026-10-21 사용자 결정): 경보를 낸 물건 영역을 바로 배경으로 흡수해 같은 물건으로 다시 경보하지 않는다.
        Tian 2011의 healing을 경보 직후로 앞당긴 것(논문은 영역이 가장 클 때 흡수). 영역 상자에 닿은 증거 덩어리 화소만 바꾼다.
        ponytail: 물건에 붙어 있는 다른 변화도 함께 흡수될 수 있다."""
        if self.model is None or self.evidence is None or not self._recent:
            return
        h, w = self.evidence.shape
        a, b = max(0, int(box[0] * self.scale)), max(0, int(box[1] * self.scale))
        c, d = min(w, int(box[2] * self.scale) + 1), min(h, int(box[3] * self.scale) + 1)
        # 상자에 닿은 증거 덩어리 전체(아직 증거가 덜 찬 같은 물건의 나머지 부분 포함). 상자 안만 흡수하면 남은 부분이 놓임/없어짐으로 다시 잡힌다(시연 run15)
        _, labels = cv2.connectedComponents((self.evidence > 0).astype(np.uint8))
        touched = np.unique(labels[b:d, a:c][self.evidence[b:d, a:c] > 0])
        sel = np.isin(labels, touched[touched > 0])
        ys, xs = np.nonzero(sel)
        if not len(xs):
            return
        span = (int(xs.min() / self.scale), int(ys.min() / self.scale), int((xs.max() + 1) / self.scale), int((ys.max() + 1) / self.scale))
        self._absorbed.append((span, sel, self.model.mu[sel].copy()))
        self.model.mu[sel] = np.mean(self._recent, axis=0, dtype=np.float32)[sel]
        self.evidence[sel] = 0
        if self.flag_age is not None:
            self.flag_age[sel] = 0
        self.last_time[kind], self.owner[kind], self.touch_time[kind] = None, None, None

    def _restore_absorbed(self, box: tuple[int, int, int, int]) -> bool:
        """없어진 영역의 중심이 흡수했던 물건 상자 안이면 그 화소의 기준을 흡수 전(빈 화면)으로 되돌리고 True."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for i, (b, sel, old) in enumerate(self._absorbed):
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                self._taken = (b, sel, self.model.mu[sel].copy())   # 가져간 물건의 모습(still_there에서 '아직 없음' 확인용)
                self.model.mu[sel] = old
                self.evidence[sel] = 0
                if self.flag_age is not None:
                    self.flag_age[sel] = 0
                del self._absorbed[i]
                return True
        return False

    @staticmethod
    def _same_spot(a: tuple[int, int, int, int], b: tuple[int, int, int, int] | None) -> bool:
        """한 상자의 중심이 다른 상자 안이면 같은 자리."""
        if b is None:
            return False
        inside = lambda p, q: q[0] <= (p[0] + p[2]) / 2 <= q[2] and q[1] <= (p[1] + p[3]) / 2 <= q[3]
        return inside(a, b) or inside(b, a)

    def recently_flagged(self, t: float, kind: str = "abandoned") -> bool:
        last = self.last_time[kind]
        return last is not None and t - last <= self.hold_sec

    def still_there(self, kind: str) -> bool:
        """그 종류의 마지막 확정 자리가 지금도 정지 물체인가. Porikli 2008: 증거가 최대인 화소만 정지 물체이고, 배경으로 돌아오면 증거가 깎인다.
        영역 검출과 같은 최소 면적(min_area_frac)을 쓴다. 지나간 그림자·발이 확정된 뒤 사라졌는데 hold_sec 동안 경보가 나가던 문제(run31 E009·E011)."""
        box = self.kind_box.get(kind)
        if box is None or self.evidence is None:
            return False
        taken = self._taken
        if kind == "removed" and taken is not None and self._same_spot(box, taken[0]) and self._recent:
            # 흡수했던 물건을 가져감: 기준을 빈 화면으로 되돌려 증거가 지워졌으므로, 저장한 물건 모습과 지금이 다른지로 본다(Tian 2011 배경 이력 비교)
            _, sel, look = taken
            expected = self.model.mu.copy()
            expected[sel] = look
            gone = self.model.classify(self._recent[-1], expected=expected)[0] & sel
            return int(gone.sum()) >= self.min_area_frac * gone.size
        x1, y1, x2, y2 = (int(v * self.scale) for v in box)
        return int((self.evidence[y1:y2 + 1, x1:x2 + 1] >= self.max_e).sum()) >= self.min_area_frac * self.evidence.size
