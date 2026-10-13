"""유기 근거: 새 물건이 생긴 자리에 그 직전 사람 손이 있었는가 (사람이 내려놓은 물건인가).

Lin 외(IEEE TIFS 2015)의 역추적 검증을 관절로 응용한다: 물건이 '실제로 나타난 시점'까지 거슬러 올라가 그때 그 자리에 사람 손이 있었는지 본다.
- 가림 역추적: 새 물건 감지기는 사람 박스 안을 보지 않으므로, 사람이 물건을 가리는 동안은 물건이 늦게 잡힌다.
  그래서 first_frame에서 물건 자리를 사람 박스가 계속 가리던 구간을 거슬러 올라가고, 그 앞 ABANDON_HAND_BEFORE_SEC까지 본다.
- 손 위치: 그 구간에 손목이 물건 박스(폭·높이의 ABANDON_HAND_MARGIN만큼 넓힘) 안에 있었는가
- 주인이 떠남(사용자 기준 2026-09-28: 놓고 화면에서 사라지면 유기): 손이 있던 사람의 추적이 끝나고 다시 나타나지 않음
  (owner_left). Lin 외 2015의 소유자 궤적 확인 응용
발 위치로 바닥 깊이를 맞추는 조건은 사람이 물건을 발 앞쪽에 내려놓아 실제 유기의 절반을 걸러서(유기 train 48개 중 41→22) 쓰지 않는다.
"""
from config.config import ABANDON_HAND_BEFORE_SEC, ABANDON_HAND_MARGIN, OWNER_LEFT_SEC, POSE_MIN_CONF

LW, RW = 9, 10  # COCO 손목


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def occlusion_start(box, first_frame, person_boxes):
    """first_frame 직전부터 물건 자리를 사람 박스가 계속 가린 구간의 시작 프레임."""
    frame = first_frame
    while frame > 0 and any(_overlaps(box, p) for p in person_boxes[frame - 1]):
        frame -= 1
    return frame


def placer(detection, pose_tracks, fps, person_boxes):
    """물건 자리에 손이 있었던 사람의 추적 번호(가장 늦게 손이 있었던 사람). 없으면 None."""
    x1, y1, x2, y2 = detection["box"]
    dx, dy = (x2 - x1) * ABANDON_HAND_MARGIN, (y2 - y1) * ABANDON_HAND_MARGIN
    start = occlusion_start(detection["box"], detection["first_frame"], person_boxes) - ABANDON_HAND_BEFORE_SEC * fps
    best = None
    for tid, frames in pose_tracks.items():
        for frame_no, k in frames.items():
            if not start <= frame_no <= detection["first_frame"]:
                continue
            for w in (LW, RW):
                if k[w][2] >= POSE_MIN_CONF and x1 - dx <= k[w][0] <= x2 + dx and y1 - dy <= k[w][1] <= y2 + dy:
                    if best is None or frame_no > best[0]:
                        best = (frame_no, tid)
    return None if best is None else best[1]


def hand_placed(detection, pose_tracks, fps, person_boxes):
    return placer(detection, pose_tracks, fps, person_boxes) is not None


def owner_left(owner, pose_tracks, fps, last_frame):
    """주인이 화면에서 사라졌으면 사라진 프레임, 아니면 None(사용자 기준: 그냥 화면에서 사라짐).

    주인 추적이 끝났고, 그 뒤 OWNER_LEFT_SEC 이상 영상이 이어졌는데 다시 나타나지 않았으면 사라진 것이다.
    ponytail: 넘어짐처럼 추적 번호만 바뀌어도 사라진 것으로 본다. 오경보가 번호 전환에서 나오면 위치로 잇는 조건을 되살린다.
    """
    end = max(pose_tracks[owner])
    return end if end <= last_frame - OWNER_LEFT_SEC * fps else None


if __name__ == "__main__":
    import numpy as np

    def pose(wx, wy):
        k = np.zeros((17, 3))
        k[LW] = (wx, wy, 0.9)
        return k

    det = {"box": (100, 100, 140, 140), "first_frame": 60}
    empty = [[] for _ in range(100)]
    assert hand_placed(det, {"1": {45: pose(120, 120)}}, 10, empty)       # 1.5초 전 손이 물건 자리
    assert hand_placed(det, {"1": {45: pose(165, 90)}}, 10, empty)        # 물건 윗부분 옆(범위 1.0배 안)
    assert not hand_placed(det, {"1": {45: pose(400, 400)}}, 10, empty)   # 손이 먼 곳
    assert not hand_placed(det, {"1": {0: pose(120, 120)}}, 10, empty)    # 창 밖(6초 전)
    assert not hand_placed(det, {"1": {70: pose(120, 120)}}, 10, empty)   # 물건이 생긴 뒤
    occluded = [[] for _ in range(100)]
    for f in range(20, 60):
        occluded[f] = [(90, 0, 160, 160)]                                  # 20~59 프레임 동안 사람이 가림
    assert occlusion_start(det["box"], 60, occluded) == 20
    assert hand_placed(det, {"1": {15: pose(120, 120)}}, 10, occluded)    # 가림 역추적으로 창이 20-50=-30까지

    def body(x, y):
        k = np.zeros((17, 3))
        k[0], k[15] = (x, y - 100, 0.9), (x, y + 100, 0.9)
        return k

    walk = {"1": {f: body(100 + f, 300) for f in range(0, 60, 3)}}
    assert owner_left("1", walk, 10, last_frame=300) == 57                          # 57에서 사라지고 영상은 계속됨
    assert owner_left("1", walk, 10, last_frame=70) is None                        # 영상 끝 직전까지 보임(떠났는지 모름)
    print("ok")
