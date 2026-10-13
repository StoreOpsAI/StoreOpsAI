"""전도: 바닥에 누워 있는 사람이 같은 자리에서 LYING_HOLD_SEC 이상 이어지면 경보.

누움 여부는 사람 박스 사진 분류기(pipelines/train_lying, Lafuente-Arroyo 외 ESWA 2022 2단계 방식 응용)가 정한다.
추적 번호 대신 위치(박스 IoU)로 이어 붙인다: 넘어질 때 번호가 바뀌어도 된다. 서 있던 기록은 요구하지 않는다.
이전 몸통·다리 각도 규칙은 추적 번호 전환과 카메라 방향으로 누운 경우(2D 각도가 세로로 보임) 때문에 train 재현율 0.22로 폐기했다.
"""
from config.config import LYING_HOLD_SEC, LYING_LINK_IOU
from models.new_object_detector import _iou


def lying_alarms(frames, fps, gap=1, hold_sec=None, link_iou=None):
    """frames: [(프레임 번호, [누움으로 판정된 사람 박스])] (표본 순서). 경보 [(프레임, 박스)].

    같은 자리(IoU >= link_iou, 기본 LYING_LINK_IOU)의 누움이 이어진 시간이 LYING_HOLD_SEC 이상이면 한 번 경보. 표본 gap개까지 끊겨도 잇는다.
    """
    hold = LYING_HOLD_SEC if hold_sec is None else hold_sec  # None이면 설정값, 아니면 호출자가 정한 유지 시간(학습으로 고를 때)
    link = LYING_LINK_IOU if link_iou is None else link_iou
    runs, alarms = [], []  # run: {"box", "start", "last", "misses", "alarmed"}
    for frame_no, boxes in frames:
        for run in runs:
            run["misses"] += 1
        for box in boxes:
            run = max(runs, key=lambda r: _iou(r["box"], box), default=None)
            if run is not None and _iou(run["box"], box) >= link:
                run.update(box=box, last=frame_no, misses=0)
            else:
                run = {"box": box, "start": frame_no, "last": frame_no, "misses": 0, "alarmed": False}
                runs.append(run)
            if not run["alarmed"] and run["last"] - run["start"] >= hold * fps:
                run["alarmed"] = True
                alarms.append((frame_no, box))
        runs = [r for r in runs if r["misses"] <= gap]
    return alarms


if __name__ == "__main__":
    fps = 3
    box = (100, 100, 200, 150)
    still = [(f, [box]) for f in range(0, 30, 1)]
    assert len(lying_alarms(still, fps)) == 1                                   # 10초 누움 → 경보 한 번
    assert lying_alarms([(f, [box]) for f in range(3)], fps) == []             # 1초 미만(3표본=0.67초) → 경보 없음
    assert len(lying_alarms([(f, [box]) for f in range(4)], fps)) == 1         # 정확히 1초 → 경보
    flicker = [(f, [box] if f % 2 == 0 else []) for f in range(12)]            # 한 표본씩 끊겨도 이어짐
    assert len(lying_alarms(flicker, fps)) == 1
    moved = [(f, [(100 + 60 * f, 100, 200 + 60 * f, 150)]) for f in range(12)]  # 자리가 계속 바뀜(걷는 사람 오판정)
    assert lying_alarms(moved, fps) == []
    other = (1000, 100, 1100, 150)  # 여러 사람: 한쪽만 계속 누워 있고 다른 사람은 걸어다님 → 누운 쪽만 경보 한 번
    crowd = [(f, [box, (60 * f, 400, 60 * f + 100, 450)]) for f in range(12)]
    found = lying_alarms(crowd, fps)
    assert len(found) == 1 and found[0][1] == box
    both = [(f, [box, other]) for f in range(12)]  # 두 사람이 동시에 누움 → 각각 경보
    assert len(lying_alarms(both, fps)) == 2
    print("ok")
