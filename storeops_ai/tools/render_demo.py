"""시연 영상에 서비스 판정(사람 박스, 행동별 점수, 경보)을 입혀 데모 영상으로 만든다.

실제 서비스 코드(Path1Webcam)에 영상 파일을 카메라 대신 넣고, 화면 오른쪽에 점수 패널을 붙인다.
- 서비스 출력 폴더와 실제 백엔드로는 아무것도 보내지 않는다(임시 폴더, 막힌 주소 사용, 끝나면 임시 폴더 삭제).
- --blur-head: 사람 박스 위쪽 22%를 흐리게 처리해 얼굴을 가린다(포트폴리오 공개용).
- 점수 기록은 <out>.scores.jsonl로 같이 남긴다.
사용(storeops_ai 폴더에서): python -m tools.render_demo --video output/demo_recordings/20261007_163155.mp4 --out demo_out.mp4 [--blur-head] [--max-sec 60]
미리보기(모델 없이 패널만): python -m tools.render_demo --video 영상.mp4 --out panel.jpg --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"
VIEW_W, VIEW_H, PANEL_W = 1280, 720, 440
BG, FG, DIM = (24, 28, 38), (236, 240, 247), (140, 150, 168)
OK_C, ALERT_C, BAR_C = (72, 199, 142), (235, 77, 75), (86, 156, 255)
SHOWN = ("쓰러짐", "쓰레기 투기", "절도")
ALARM_HOLD_SEC = 6.0


def _font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def blur_heads(frame: np.ndarray, detections) -> None:
    for _, x1, y1, x2, y2, _ in detections:
        y_end = int(y1 + 0.22 * (y2 - y1))
        roi = frame[max(y1, 0):max(y_end, 0), max(x1, 0):max(x2, 0)]
        if roi.size:
            frame[max(y1, 0):max(y_end, 0), max(x1, 0):max(x2, 0)] = cv2.GaussianBlur(roi, (0, 0), 18)


def render_panel(t: float, latest: dict | None, thresholds: dict, required: int, history: list[dict], alarm: dict | None) -> np.ndarray:
    """오른쪽 패널(BGR, PANEL_W x VIEW_H)."""
    img = Image.new("RGB", (PANEL_W, VIEW_H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 20), "StoreOps AI", font=_font(30, True), fill=FG)
    d.text((24, 62), f"영상 시간 {int(t // 60):02d}:{t % 60:04.1f}", font=_font(18), fill=DIM)
    if alarm:
        d.rounded_rectangle((20, 100, PANEL_W - 20, 182), 12, fill=ALERT_C)
        d.text((36, 108), f"경보 발생  {alarm['category']}", font=_font(26, True), fill=(255, 255, 255))
        d.text((36, 148), f"{alarm['event_id']}  확신도 {alarm['confidence']:.3f}", font=_font(18), fill=(255, 255, 255))
    else:
        d.rounded_rectangle((20, 100, PANEL_W - 20, 182), 12, fill=(36, 44, 60))
        d.text((36, 118), "감시 중" if latest else "영상 수집 중", font=_font(26, True), fill=OK_C)
        if latest:
            d.text((36, 154), f"현재 판정 {latest['category']}", font=_font(18), fill=DIM)
    d.text((24, 206), "행동별 점수 (점선 = 경보 임계값)", font=_font(18, True), fill=FG)
    y = 244
    scores = (latest or {}).get("scores", {})
    for name in SHOWN:
        v = float(scores.get(name, 0.0))
        d.text((24, y), name, font=_font(18), fill=FG)
        d.text((PANEL_W - 24 - 70, y), f"{v:.3f}", font=_font(18), fill=DIM)
        x0, x1 = 24, PANEL_W - 24
        d.rounded_rectangle((x0, y + 30, x1, y + 46), 6, fill=(44, 52, 70))
        if v > 0:
            d.rounded_rectangle((x0, y + 30, x0 + max(8, int((x1 - x0) * min(v, 1.0))), y + 46), 6, fill=ALERT_C if v > thresholds.get(name, 1.1) else BAR_C)
        tx = x0 + int((x1 - x0) * min(thresholds.get(name, 1.0), 1.0))
        for yy in range(y + 26, y + 50, 6):
            d.line((tx, yy, tx, yy + 3), fill=(255, 214, 102), width=2)
        y += 66
    if latest and latest.get("category") not in ("정상", "판정 보류"):
        d.text((24, y), f"연속 확인 {latest.get('confirm_count', 0)}/{required}", font=_font(18), fill=(255, 214, 102))
    d.text((24, 490), "경보 기록", font=_font(18, True), fill=FG)
    for i, ev in enumerate(history[-5:][::-1]):
        d.text((24, 526 + i * 34), f"{ev['event_id']}  {ev['category']}  {int(ev['t'] // 60):02d}:{ev['t'] % 60:04.1f}", font=_font(18), fill=DIM)
    if not history:
        d.text((24, 526), "아직 없음", font=_font(18), fill=DIM)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def compose(frame: np.ndarray, detections, panel: np.ndarray, alarm_active: bool) -> np.ndarray:
    view = frame.copy()
    color = (0, 0, 235) if alarm_active else (0, 220, 120)
    for tid, x1, y1, x2, y2, conf in detections:
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 4)
        cv2.putText(view, f"ID {tid}", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    view = cv2.resize(view, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
    return np.hstack([view, panel])


def selftest(video: str, out: str) -> None:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = cap.read()
    assert ok, "프레임을 읽지 못했다"
    latest = {"category": "쓰러짐", "confidence": 0.999, "scores": {"정상": 0.0, "쓰러짐": 0.999, "쓰레기 투기": 0.12, "절도": 0.40}, "confirm_count": 2}
    th = {"쓰러짐": 0.998, "쓰레기 투기": 0.991, "절도": 0.896}
    alarm = {"category": "쓰러짐", "event_id": "E001", "confidence": 0.999}
    panel = render_panel(12.3, latest, th, 2, [{"event_id": "E001", "category": "쓰러짐", "t": 11.0}], alarm)
    cv2.imencode(".jpg", compose(frame, [(1, 400, 300, 700, 900, 0.9)], panel, True))[1].tofile(out)
    print("미리보기 저장:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sec", type=float, default=0, help="앞에서부터 이 초만 처리(0 = 전체)")
    ap.add_argument("--blur-head", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.video, args.out)

    tmp = tempfile.mkdtemp(prefix="demo_render_")
    os.environ.update(STOREOPS_OUTPUT_DIR=tmp, ALERT_WEBHOOK_URL="http://127.0.0.1:9/none", EVENT_INGEST_TOKEN="demo", VLM_ON_EVENT="0", VLM_ENABLED="0")
    real_cap = cv2.VideoCapture
    probe = real_cap(args.video)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    limit = int(args.max_sec * fps) if args.max_sec else 10 ** 9
    probe.release()

    class Limited:
        def __init__(self, *_a, **_k):
            self.cap, self.n = real_cap(args.video), 0
        def read(self):
            self.n += 1
            return self.cap.read() if self.n <= limit else (False, None)
        def isOpened(self): return self.cap.isOpened()
        def get(self, prop): return self.cap.get(prop)
        def release(self): self.cap.release()

    cv2.VideoCapture = Limited
    cv2.waitKey = lambda *a, **k: -1
    cv2.namedWindow = cv2.destroyAllWindows = lambda *a, **k: None

    from config.config import CATEGORY_THRESHOLDS
    from pipelines.path1_webcam import Path1Webcam

    state = {"frame": 0, "seen_events": 0, "alarm": None, "history": [], "last_scores": None, "writer": None}
    log = open(Path(args.out).with_suffix(".scores.jsonl"), "w", encoding="utf-8")
    walker = Path1Webcam(camera_id="DEMO", debug=False)

    def draw(self, frame, detections, latest):
        t = state["frame"] / fps
        state["frame"] += 1
        if args.blur_head:
            blur_heads(frame, detections)
        for ev in self.events_issued[state["seen_events"]:]:
            state["alarm"] = {"category": ev.category, "event_id": ev.event_id, "confidence": ev.confidence, "t": t}
            state["history"].append({"event_id": ev.event_id, "category": ev.category, "t": t})
        state["seen_events"] = len(self.events_issued)
        if state["alarm"] and t - state["alarm"]["t"] > ALARM_HOLD_SEC:
            state["alarm"] = None
        if latest and latest["scores"] != state["last_scores"]:
            state["last_scores"] = dict(latest["scores"])
            log.write(json.dumps({"t": round(t, 2), "category": latest["category"], "scores": latest["scores"]}, ensure_ascii=False) + "\n")
        panel = render_panel(t, latest, CATEGORY_THRESHOLDS, self.confirmations_required, state["history"], state["alarm"])
        return compose(frame, detections, panel, state["alarm"] is not None)

    def show(_name, img):
        if state["writer"] is None:
            state["writer"] = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        state["writer"].write(img)

    Path1Webcam._draw = draw
    cv2.imshow = show
    try:
        walker.run(0)
    finally:
        if state["writer"]:
            state["writer"].release()
        log.close()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"완료: {args.out} (경보 {len(state['history'])}건, 점수 기록 {Path(args.out).with_suffix('.scores.jsonl')})")


if __name__ == "__main__":
    main()
