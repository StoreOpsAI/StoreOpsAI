"""유기 사건 판정(PETS 2006 방치 수하물 정의 응용): 발견 → 소유자 연결 → 관리 중 / 방치 → 경보 또는 취소.

PETS 2006: 소유자가 물건에서 a=2m 안이면 '관리 중', b=3m 밖이면 '방치'(그 사이는 직전 상태 유지).
방치가 t초 이어지고 그동안 소유자가 돌아오지 않고 다른 사람도 손대지 않으면 경보. 다른 사람이 손대면 유기가 아니다.
- 소유자: 물건이 생기기 직전 그 자리에 손이 있던 사람(placement_check.placer, Lin 외 TIFS 2015 역추적)
- 거리: 화면 픽셀을 소유자 키(관절 최상단~최하단)로 나눠 PERSON_HEIGHT_M을 곱한 근사 미터. 발(관절 최하단)과 물건 아래 가운데 사이
- 소유자가 화면에 없으면 방치(사용자 기준: 화면에서 사라지면 떠난 것)
- 취소: 경보 전 소유자 손이 물건에 다시 닿음(집어 감/관리) 또는 다른 사람 손이 닿음
"""
from config.config import ABANDON_ATTEND_M, ABANDON_GRACE_SEC, ABANDON_UNATTEND_M, PERSON_HEIGHT_M, POSE_MIN_CONF

LW, RW = 9, 10
SAMPLE_FPS = 3  # 관절 추정 표본 간격(eval_broken_pose.SAMPLE_FPS와 같게)


def _touches(k, box, margin):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return any(k[w][2] >= POSE_MIN_CONF and x1 - dx <= k[w][0] <= x2 + dx and y1 - dy <= k[w][1] <= y2 + dy for w in (LW, RW))


def owner_distance_m(k, box):
    """소유자 발과 물건 아래 가운데 사이 근사 거리(m). 관절이 부족하면 None."""
    seen = [p for p in k if p[2] >= POSE_MIN_CONF]
    if len(seen) < 3:
        return None
    top, feet = min(p[1] for p in seen), max(p[1] for p in seen)
    if feet - top < 10:
        return None
    foot_x = sum(p[0] for p in seen) / len(seen)
    ox, oy = (box[0] + box[2]) / 2, box[3]
    return ((foot_x - ox) ** 2 + (feet - oy) ** 2) ** 0.5 / (feet - top) * PERSON_HEIGHT_M


def abandon_alarm(detection, owner, pose_tracks, fps, last_frame, touch_margin=0.3, grace_sec=None):
    """경보 프레임 또는 None. 물건 확정(confirmed_frame) 뒤 표본 프레임을 차례로 보며 방치 시간을 센다.
    grace_sec: 유예 시간(없으면 ABANDON_GRACE_SEC). 평가에서 여러 값을 비교할 때만 넘긴다."""
    grace_sec = ABANDON_GRACE_SEC if grace_sec is None else grace_sec
    box, start = detection["box"], detection["confirmed_frame"]
    step = max(1, round(fps / SAMPLE_FPS))  # 관절은 이 간격 표본. 사람이 안 보여도 시간은 이 간격으로 센다
    frames = range(start - start % step, last_frame + 1, step)
    unattended_since, state = None, "attended"
    for f in frames:
        for tid, track in pose_tracks.items():
            if f in track and _touches(track[f], box, touch_margin) and (tid != owner or state == "unattended"):
                return None  # 다른 사람이 손댐, 또는 방치 뒤 소유자가 돌아와 손댐 → 유기 아님(취소)
        k = pose_tracks[owner].get(f)
        distance = owner_distance_m(k, box) if k is not None else float("inf")  # 화면에 없음 = 떠남
        if distance is not None:
            if distance > ABANDON_UNATTEND_M:
                state = "unattended"
            elif distance <= ABANDON_ATTEND_M:
                state = "attended"
        if state == "unattended":
            unattended_since = f if unattended_since is None else unattended_since
            if f - unattended_since >= grace_sec * fps:
                return f
        else:
            unattended_since = None
    return None


if __name__ == "__main__":
    import numpy as np

    def person(x, feet_y, height=340, hand=None):
        k = np.zeros((17, 3))
        k[0] = (x, feet_y - height, 0.9)
        k[15] = k[16] = (x, feet_y, 0.9)
        k[LW] = k[RW] = (*(hand or (x, feet_y - height / 2)), 0.9)
        return k

    fps, box = 3, (400, 560, 440, 600)  # 물건 아래 가운데 (420, 600). 키 340px ≈ 1.7m → 200px ≈ 1m
    det = {"box": box, "confirmed_frame": 0}
    walk_away = {"o": {f: person(420 + 100 * f, 600) for f in range(0, 10)}}      # 0.5m씩 멀어지다 f=9 뒤 화면 밖
    import config.config as c
    grace = round(c.ABANDON_GRACE_SEC * fps)
    assert abandon_alarm(det, "o", walk_away, fps, last_frame=60) == 7 + grace      # 3m 넘는 f=7(3.5m)부터 유예 뒤 경보
    assert abandon_alarm(det, "o", walk_away, fps, last_frame=7 + grace - 1) is None  # 유예가 끝나기 전 영상 끝
    stays = {"o": {f: person(460, 600) for f in range(0, 60)}}                    # 0.2m 옆에 계속 있음(관리 중)
    assert abandon_alarm(det, "o", stays, fps, last_frame=60) is None
    back = {"o": {**{f: person(420 + 100 * f, 600) for f in range(0, 10)},         # 멀어졌다가
                  **{f: person(420, 600, hand=(420, 580)) for f in range(12, 20)}}}  # 돌아와 집음 → 취소
    if grace > 5:  # 유예가 돌아오는 시점(f=12)보다 길 때만 취소가 먼저 일어난다
        assert abandon_alarm(det, "o", back, fps, last_frame=60) is None
    other = {"o": walk_away["o"], "x": {f: person(420, 600, hand=(420, 580)) for f in (8,)}}  # 다른 사람이 손댐
    assert abandon_alarm(det, "o", other, fps, last_frame=60) is None
    print("ok")
