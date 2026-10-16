"""한 사람만 나오는 영상에서 추적 번호가 몇 번 바뀌는지 잰다(라벨 없이 쓸 수 있는 조각남 측정).
사용: python tools/measure_track_fragments.py 영상 추적설정.yaml [track_buffer]
번호가 N개면 조각남 N-1회. 끊긴 뒤 새 번호가 생기기까지 간격(초)이 track_buffer를 정하는 근거다."""
import sys
import yaml
from ultralytics import YOLO


def main(video, tracker, buffer=None, model="yolo11n.pt"):
    cfg = tracker
    if buffer:
        spec = yaml.safe_load(open(tracker, encoding="utf-8"))
        spec["track_buffer"] = int(buffer)
        cfg = "_tmp_tracker.yaml"
        yaml.safe_dump(spec, open(cfg, "w", encoding="utf-8"))
    spans = {}
    fps = 30.0
    for n, r in enumerate(YOLO(model).track(video, tracker=cfg, classes=[0], stream=True, verbose=False)):
        if r.boxes.id is None:
            continue
        for tid in r.boxes.id.int().tolist():
            s = spans.setdefault(tid, [n, n])
            s[1] = n
    order = sorted(spans.items(), key=lambda kv: kv[1][0])
    print(f"번호 수={len(order)} (조각남 {max(0, len(order) - 1)}회)")
    for i, (tid, (a, b)) in enumerate(order):
        gap = (a - order[i - 1][1][1]) / fps if i else 0.0
        print(f"  id={tid} 프레임 {a}-{b} 이전 번호와 간격 {gap:.1f}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
