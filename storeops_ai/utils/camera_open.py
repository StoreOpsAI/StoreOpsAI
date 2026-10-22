"""웹캠을 학습 영상과 같은 해상도·화면비로 연다.

A 스트림은 전체 화면을 224x224로 눌러 쓰므로 화면비가 학습(1920x1080, 16:9)과 다르면 사람 모양이 달라진다.
해상도를 지정하지 않으면 OpenCV가 640x480(4:3)으로 열 수 있어, 요청한 화면비로 열리지 않으면 조용히 넘어가지 않고 예외를 던진다.

lock_camera: 자동 노출·화이트밸런스·초점을 끄고 값을 고정한다. 정지 물건 색 모델(Horprasert 외 1999)은 카메라 설정이 고정돼 있다고
가정하고 빈 화면 20초의 화소 변동만 배운다. 자동 화이트밸런스가 사람이 들어올 때 채널 이득을 바꾸면 화면 대부분이 "색이 다름"으로
뜬다(2026-10-20 실측: 채널 비 B 0.89 / G 0.91 / R 0.95, 물건 마스크 76%). 처음에는 자동이 맞춘 값을 읽어 고정하고,
보정 때 고정한 값은 보정 파일에 저장해 다음 실행에서 같은 값으로 연다.
"""
from __future__ import annotations

import cv2
import numpy as np

from config.config import CAMERA_LOCK_WARMUP_FRAMES, CAPTURE_ALLOW_ASPECT, CAPTURE_HEIGHT, CAPTURE_WIDTH

ASPECT_TOLERANCE = 0.02
# (이름, 자동 속성, 수동 값, 고정할 속성들). DSHOW/V4L2는 AUTO_EXPOSURE 0.25가 수동이다(OpenCV 관례)
LOCK_PROPS = (("white_balance", cv2.CAP_PROP_AUTO_WB, 0.0, (cv2.CAP_PROP_WB_TEMPERATURE,)),
              ("exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, (cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_GAIN)),
              ("focus", cv2.CAP_PROP_AUTOFOCUS, 0.0, (cv2.CAP_PROP_FOCUS,)))


def open_camera(index: int, width: int = CAPTURE_WIDTH, height: int = CAPTURE_HEIGHT, allow_aspect: bool = CAPTURE_ALLOW_ASPECT) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():   # 일부 카메라/PC에서는 DSHOW가 안 맞을 수 있어 기본 backend로 재시도
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"웹캠을 열 수 없습니다. camera={index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # 기본 YUY2는 대역폭 때문에 1080p를 못 여는 카메라가 많다
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    got_w, got_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 카메라 해상도: 요청 {width}x{height} -> 실제 {got_w}x{got_h}", flush=True)
    if not aspect_ok(got_w, got_h, width, height) and not allow_aspect:
        cap.release()
        raise RuntimeError(f"화면비가 학습과 다릅니다({got_w}x{got_h}). 16:9를 지원하는 카메라·해상도를 쓰거나 CAPTURE_ALLOW_ASPECT=1로 알고 쓰세요.")
    return cap


def _luma(cap: cv2.VideoCapture, n: int = 3) -> float:
    vals = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).mean() for ok, f in (cap.read() for _ in range(n)) if ok]
    return float(np.mean(vals)) if vals else 0.0


def wait_auto_settled(cap: cv2.VideoCapture, min_frames: int, max_frames: int = 300, window: int = 15, tol: float = 0.01) -> float:
    """자동 노출이 다 맞춰질 때까지 읽는다: 최근 window장의 평균 밝기 변화가 tol(1%) 안이면 끝. 반환: 그때 평균 밝기."""
    lumas: list[float] = []
    for i in range(max_frames):
        lumas.append(_luma(cap, 1))
        recent = lumas[-window:]
        if i + 1 >= min_frames and len(recent) == window and (max(recent) - min(recent)) <= tol * max(np.mean(recent), 1.0):
            break
    return float(np.mean(lumas[-window:]))


def _match_exposure(cap: cv2.VideoCapture, target: float, start: float, span: int = 6, settle: int = 5) -> tuple[float, float]:
    """수동 노출에서 밝기가 target(자동이 맞춘 밝기)에 가장 가까운 노출값을 찾는다. 자동 모드에서 읽은 노출값은 카메라마다
    실제와 다를 수 있어(2026-10-20 C920/C925e: 드라이버가 이전 수동값을 보고) 값을 믿지 않고 밝기로 맞춘다. 반환: (노출, 밝기)."""
    best = (start, float("inf"), 0.0)
    for e in range(int(round(start)) - span, int(round(start)) + span + 1):
        if not cap.set(cv2.CAP_PROP_EXPOSURE, e):
            continue
        for _ in range(settle):
            cap.read()
        lum = _luma(cap)
        if abs(lum - target) < best[1]:
            best = (float(e), abs(lum - target), lum)
    cap.set(cv2.CAP_PROP_EXPOSURE, best[0])
    return best[0], best[2]


def lock_camera(cap: cv2.VideoCapture, saved: dict | None = None, warmup: int = CAMERA_LOCK_WARMUP_FRAMES) -> dict:
    """자동 조정을 끄고 값을 고정한다. saved(보정 때 값)가 있으면 그 값으로 연다.
    없으면: 자동 조정을 다시 켜고(드라이버가 이전 실행의 수동값을 기억한다) 밝기가 안정될 때까지 기다린 뒤, 화이트밸런스·초점은
    그 값으로, 노출은 같은 밝기가 나오는 수동값을 찾아 고정한다. 반환: {이름: {"values": [...], "locked": bool}}."""
    target = None
    if not saved:
        for _, auto, _, _ in LOCK_PROPS:
            cap.set(auto, 0.75 if auto == cv2.CAP_PROP_AUTO_EXPOSURE else 1.0)   # DSHOW/V4L2 관례: 0.75 = 자동 노출
        target = wait_auto_settled(cap, warmup)
    out = {}
    for name, auto, manual, props in LOCK_PROPS:
        vals = saved[name]["values"] if saved and name in saved else [cap.get(p) for p in props]
        ok = cap.set(auto, manual)
        for p, v in zip(props, vals):
            ok = cap.set(p, v) and abs(cap.get(p) - v) < 1e-3 and ok
        out[name] = {"values": [float(v) for v in vals], "locked": bool(ok)}
    if target is not None and out["exposure"]["locked"]:
        e, lum = _match_exposure(cap, target, out["exposure"]["values"][0])
        out["exposure"]["values"][0] = e
        out["exposure"]["luma"] = {"auto": round(target, 1), "locked": round(lum, 1)}
    bad = [n for n, r in out.items() if not r["locked"]]
    desc = ", ".join(f"{n}={r['values']}" for n, r in out.items())
    lum = out["exposure"].get("luma")
    print(f"[카메라] 자동 조정 고정: {desc}" + (f" 밝기 자동 {lum['auto']} -> 고정 {lum['locked']}" if lum else "")
          + (f" / 고정 실패: {bad} (물건 판정이 자동 조정에 흔들릴 수 있음)" if bad else ""), flush=True)
    return out


def aspect_ok(got_w: int, got_h: int, want_w: int, want_h: int) -> bool:
    return got_h > 0 and abs(got_w / got_h - want_w / want_h) <= ASPECT_TOLERANCE


def _self_check() -> None:
    assert aspect_ok(1920, 1080, 1920, 1080) and aspect_ok(1280, 720, 1920, 1080)
    assert not aspect_ok(640, 480, 1920, 1080) and not aspect_ok(0, 0, 1920, 1080)
    print("ok")


if __name__ == "__main__":
    _self_check()
