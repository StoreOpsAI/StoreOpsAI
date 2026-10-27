"""정지 물건 검출 자체 검사(utils/static_object.py에서 옮김: 모듈 300줄 규칙). 실행: python -m unittest discover -s tests -p test_static_object.py"""
import unittest

import numpy as np

from utils.color_model import fit
from utils.static_object import StaticObjectDetector


def _self_check() -> None:
    rng = np.random.default_rng(0)
    noise = lambda img: np.clip(img + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    empty = rng.integers(90, 110, (240, 320, 3), dtype=np.uint8)
    with_obj = empty.copy(); with_obj[100:140, 150:200] = 250
    make = lambda: StaticObjectDetector(1.0, 5.0, 5.0, 0.002, 30, max_evidence=20, decay=1, min_area_frac=0.001, hold_sec=4)

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
    near, far = [(7, 115.0, 60.0, 235.0, 180.0)], [(7, -55.0, -55.0, 65.0, 65.0)]   # 주인 후보 상자 (번호, x1, y1, x2, y2)
    for trail, expect in ((near, 7), (far, None), ([], None)):    # 놓은 사람의 손목이 근처에 없으면 유기가 아니다
        d = make(); t = 0.0
        for _ in range(400):
            d.update(noise(empty), t, [], trail); t += 0.2
        for _ in range(60):
            d.update(noise(with_obj), t, [], trail); t += 0.2
        assert d.recently_flagged(t, "abandoned") == (expect is not None) and d.owner["abandoned"] == expect, (trail, expect)
    shadow = (empty.astype(np.float32) * 0.6).astype(np.uint8)          # 같은 색, 60% 밝기 = 그림자
    shaded = empty.copy(); shaded[100:140, 150:200] = shadow[100:140, 150:200]
    d = make(); t = run(d, empty, shaded)
    assert not d.recently_flagged(t, "abandoned") and not d.recently_flagged(t, "removed")   # 그림자는 물건이 아니다
    d = StaticObjectDetector(1.0, 5.0, 5.0, 0.002, 30, max_evidence=20, decay=1, min_area_frac=0.001, hold_sec=4, detect_shadows=False)
    t = run(d, empty, shaded)
    assert d.recently_flagged(t, "abandoned") or d.recently_flagged(t, "removed")            # 그림자 처리를 끄면 물건으로 잡힌다(점검이 의미 있는지 확인)

    def run_ref(ref, scene, steps=80):   # 빈 화면 색 모델 대조: 긴 학습 없이 바로 쓴다
        det = make(); assert det.set_model(fit([noise(ref) for _ in range(40)], 0.9999, 0.4), ref.shape); t = 0.0
        for _ in range(steps):
            det.update(noise(scene), t, []); t += 0.2
        return det, t
    assert not make().set_model(fit([noise(empty) for _ in range(4)], 0.9999, 0.4), (480, 640, 3))           # 해상도가 다르면 쓰지 않는다
    d, t = run_ref(empty, empty)
    assert not d.recently_flagged(t) and not d.recently_flagged(t, "removed")
    d, t = run_ref(empty, with_obj)                                                         # 유기 경보 뒤 흡수(MVP): 같은 물건을 다시 잡지 않는다
    assert d.recently_flagged(t) and d.boxes
    d.absorb(d.boxes[0])
    assert not d.recently_flagged(t)
    for _ in range(80):
        d.update(noise(with_obj), t, []); t += 0.2
    assert not d.recently_flagged(t) and not d.recently_flagged(t, "removed")
    hist = []
    for _ in range(80):                                                                     # 흡수한 물건을 가져감 = 없어짐(MVP: 절도), 기준은 빈 화면으로 복귀
        d.update(noise(empty), t, []); t += 0.2
        hist.append(d.recently_flagged(t, "removed"))
        if hist[-1]:
            assert d.still_there("removed")                                                 # 흡수했던 물건을 가져감: 증거는 지워져도 물건은 여전히 없다
    assert any(hist) and not d._absorbed and not hist[-1] and not d.recently_flagged(t)    # 한 번 잡힌 뒤 계속 잡히지 않는다
    d, t = run_ref(empty, with_obj)                                                         # 상자가 물건 일부만 덮어도 붙은 증거 덩어리 전체를 흡수
    d.absorb((150, 120, 200, 140))
    for _ in range(80):
        d.update(noise(with_obj), t, []); t += 0.2
        assert not d.recently_flagged(t) and not d.recently_flagged(t, "removed")
    assert StaticObjectDetector._same_spot((0, 0, 10, 10), (4, 4, 30, 30)) and not StaticObjectDetector._same_spot((0, 0, 10, 10), (20, 20, 30, 30))
    body = (140, 90, 210, 150)                                                               # 사람 박스(물건 자리를 덮음)
    person = empty.copy(); person[90:150, 140:210] = 30                                      # 사람 몸(어두운 덩어리)
    def scenario(ref, steps):   # (장면, 사람 박스 목록) 순서대로. 주인 후보 = 사람 박스(번호 9)
        det = make(); assert det.set_model(fit([noise(ref) for _ in range(40)], 0.9999, 0.4), ref.shape); t = 0.0
        for scene, boxes, n in steps:
            for _ in range(n):
                det.update(noise(scene), t, boxes, [(9, *b) for b in boxes]); t += 0.2
        return det, t
    d, t = scenario(with_obj, [(with_obj, [], 20), (person, [body], 10), (empty, [], 60)])   # 절도: 사람이 덮었다가 떠난 뒤 물건이 없음
    assert d.recently_flagged(t, "removed") and d.owner["removed"] == 9, d.owner                # 들고 나간 사람 = 마지막으로 덮은 사람
    assert 4.0 <= d.touch_time["removed"] <= 5.8 < d.last_time["removed"] - 2.0, (d.touch_time, d.last_time)   # 클립 기준 = 사람이 덮은 4.0~5.8초(비킨 뒤 확정 - Ts가 아님)
    d = make(); assert d.set_model(fit([noise(with_obj) for _ in range(40)], 0.9999, 0.4), with_obj.shape); t = 0.0   # 손은 처음 3번만 닿고 몸은 계속 덮음(놓은 뒤 옆에 섬)
    for i, (scene, boxes) in enumerate([(with_obj, [])] * 20 + [(person, [body])] * 10 + [(empty, [])] * 60):
        hands = [(9, *body)] if 20 <= i < 23 else []
        d.update(noise(scene), t, boxes, hands + [(9, *b) for b in boxes], n_hands=len(hands)); t += 0.2
    assert d.owner["removed"] == 9 and 4.0 <= d.touch_time["removed"] <= 4.4 + 1e-6, d.touch_time   # 클립 기준 = 손이 닿은 시각(몸이 떠난 5.8초가 아님)
    d, t = scenario(empty, [(empty, [], 20), (person, [body], 10), (with_obj, [], 60)])        # 유기: 사람이 덮었다가 떠난 뒤 물건이 남음
    assert d.recently_flagged(t, "abandoned") and d.owner["abandoned"] == 9, d.owner
    d, t = scenario(empty, [(person, [body], 80)])                                            # 가만히 선 사람은 물건이 아니다(사람 박스로 거름)
    assert not d.recently_flagged(t) and not d.recently_flagged(t, "removed")
    mask = np.zeros((240, 320), np.uint8); mask[100:140, 150:200] = 1                       # 윤곽선 보조 판정(영역 키우기가 애매할 때)
    flat = np.full((240, 320, 3), 100, np.uint8); boxed = flat.copy(); boxed[100:140, 150:200] = 160
    e = StaticObjectDetector(1.0, 5.0, 5.0, 0.002, 30, max_evidence=20, decay=1, min_area_frac=0.001, hold_sec=4, edge_ratio=1.5)
    assert e._edge_type(mask, boxed, flat) == "abandoned" and e._edge_type(mask, flat, boxed) == "removed"
    assert e._edge_type(mask, (flat * 1.3).astype(np.uint8), flat) is None                  # 밝기만 바뀌면 윤곽선이 없어 애매
    assert make()._edge_type(mask, boxed, flat) is None                                     # edge_ratio 0이면 끔
    d, t = run_ref(empty, with_obj)                                                         # 기준에 없던 물건 = 놓임
    assert d.recently_flagged(t, "abandoned") and not d.recently_flagged(t, "removed")
    d, t = run_ref(with_obj, empty)                                                         # 기준에 있던 물건이 없음 = 없어짐
    assert d.recently_flagged(t, "removed") and not d.recently_flagged(t, "abandoned") and d.still_there("removed")
    d, t = run_ref(empty, with_obj)                                                         # 확정 뒤 사라짐(지나간 그림자·발, run31 E009·E011)
    for _ in range(5):
        d.update(noise(empty), t, []); t += 0.2
    assert d.recently_flagged(t, "abandoned") and not d.still_there("abandoned")            # 확정 기록은 남아도 지금은 물건이 아니다
    gain = lambda img, k: np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)
    d, t = run_ref(empty, gain(empty, 1.2))                                                 # 화면 전체 밝기 변화(자동 노출 이득)는 물건이 아니다
    assert not d.recently_flagged(t) and not d.recently_flagged(t, "removed")
    d, t = run_ref(empty, shaded)                                                           # 그림자는 물건이 아니다
    assert d.debug_counts[3] == 0 and not d.recently_flagged(t)
    for k in (1.3, 2.0):                                                                    # 같은 색, 1.3배·2배 밝은 빛 얼룩(상한 없음, 논문대로)
        lit = empty.copy(); lit[100:140, 150:200] = gain(empty[100:140, 150:200], k)
        d, t = run_ref(empty, lit)
        assert d.debug_counts[3] == 0 and not d.recently_flagged(t), k                     # 물건 후보 마스크 단계에서 빠진다
    dark = empty.copy(); dark[100:140, 150:200] = gain(empty[100:140, 150:200], 0.3)       # 같은 색이라도 너무 어두우면(검은 물건) 물건
    d, t = run_ref(empty, dark)
    assert d.debug_counts[3] > 0
    tex = lambda: StaticObjectDetector(1.0, 5.0, 5.0, 0.002, 30, max_evidence=20, decay=1, min_area_frac=0.001, hold_sec=4,
                                       bg_tau_sec=20.0, heal_sec=8.0, texture_sim=0.7)
    stripes = empty.copy(); stripes[:, ::4] = 40                                            # 무늬 있는 배경
    tinted = np.clip(stripes.astype(np.float32) * (0.8, 1.0, 1.3), 0, 255).astype(np.uint8)   # 햇빛처럼 색이 바뀐 빛(무늬는 같음)
    def run_tex(ref, scenes, det=None, keep=False):   # keep: 이어서 돌린다(모델을 다시 넣지 않음)
        det = det or tex(); t = getattr(det, "_t", 0.0)
        if not keep:
            assert det.set_model(fit([noise(ref) for _ in range(40)], 0.9999, 0.4), ref.shape); t = 0.0
        hist = []
        for scene, n in scenes:
            for _ in range(n):
                det.update(noise(scene), t, []); t += 0.2
                hist.append((det.recently_flagged(t), det.recently_flagged(t, "removed")))
        det._t = t
        return det, hist
    d, _ = run_tex(stripes, [(tinted, 60)])                                                 # 빠른 색 빛 변화는 무늬가 같아 물건 아님
    assert d.debug_counts[3] == 0, d.debug_counts
    d0, _ = run_tex(stripes, [(tinted, 60)], StaticObjectDetector(1.0, 5.0, 5.0, 0.002, 30, max_evidence=20, decay=1, min_area_frac=0.001, hold_sec=4))
    assert d0.debug_counts[3] > 0                                                           # 무늬 판정을 끄면 잡힌다(점검이 의미 있음)
    drift = [(np.clip(empty.astype(np.float32) * (1 - 0.004 * i, 1.0, 1 + 0.004 * i), 0, 255).astype(np.uint8), 10) for i in range(60)]
    d, _ = run_tex(empty, drift)                                                            # 2분 동안 천천히 색이 24% 바뀜(무늬 없는 바닥)
    assert d.debug_counts[0] < 0.01 * empty.shape[0] * empty.shape[1], d.debug_counts        # 따라가서 기준과 다름이 거의 없다
    d0, _ = run_tex(empty, drift, StaticObjectDetector(1.0, 5.0, 5.0, 0.002, 30, max_evidence=20, decay=1, min_area_frac=0.001, hold_sec=4))
    assert d0.debug_counts[0] > 0.5 * empty.shape[0] * empty.shape[1]                       # 따라가기를 끄면 화면 대부분이 다름
    d, hist = run_tex(empty, [(with_obj, 80)])                                              # 놓인 물건: 잡힌 뒤 heal_sec(8초) 지나 흡수
    assert any(a for a, _ in hist) and d.debug_counts[4] == 0 and d.debug_counts[3] < 30, d.debug_counts   # 흩어진 잡음 화소만 남는다(다시 흡수됨)
    d, hist = run_tex(empty, [(empty, 40)], d, keep=True)                                   # 흡수된 물건을 가져감 = 없어짐
    assert any(r for _, r in hist) and not any(a for a, _ in hist[-5:]), hist[-5:]
    print("ok")


class TestStaticObject(unittest.TestCase):
    def test_self_check(self):
        _self_check()


if __name__ == "__main__":
    unittest.main()
