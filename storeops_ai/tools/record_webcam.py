"""웹캠을 N초 녹화한다(추적 설정 측정용 영상). 사용: python tools/record_webcam.py 출력.mp4 [초] [카메라 번호]
웹캠을 쓰는 다른 프로그램(path1_webcam)은 먼저 꺼야 한다."""
import sys
import time

import cv2


def main(out: str, seconds: float = 60.0, camera: int = 0) -> None:
    cap = cv2.VideoCapture(int(camera))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("웹캠을 열지 못했습니다(다른 프로그램이 쓰고 있지 않은지 확인).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
    start, n = time.time(), 0
    while time.time() - start < float(seconds):
        ok, frame = cap.read()
        if ok:
            writer.write(frame)
            n += 1
    writer.release(); cap.release()
    print(f"저장 {out}: {n}프레임, {time.time() - start:.1f}초, {n / (time.time() - start):.1f}fps")


if __name__ == "__main__":
    main(*sys.argv[1:4])
