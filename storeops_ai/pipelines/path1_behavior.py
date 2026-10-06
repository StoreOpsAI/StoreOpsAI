from __future__ import annotations
import argparse
import cv2
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

import logging

from config.config import (
    YOLO_MODEL_PATH, TRACKER_CONFIG, CONF_THRESHOLD, PERSON_CLASS_ID,
    CLIP_MIN_SEC, CLIP_MAX_SEC, EVENT_VIDEO_SEC, EVENT_VIDEO_PRE_SEC, EVENT_VIDEO_POST_SEC, CATEGORY_THRESHOLDS,
    EXPANDED_CROP_CATEGORIES, EXPAND_X, EXPAND_Y, EVENT_DIR, CLIP_DIR, IMAGE_DIR,
    I3D_INPUT_MODE, I3D_WINDOW_SEC, I3D_INFER_INTERVAL_SEC, I3D_BUFFER_FPS, EVENT_COOLDOWN_SEC,
)
from models.yolo_detector import YOLODetector
from models.i3d_classifier import I3DClassifier
from models.vlm_analyzer import VLMAnalyzer
from services.realtime import event_hub
from schemas.event_schema import EventCandidate
from utils.clip_manager import save_frames, save_representative_images
from utils.event_manager import EventManager
from utils.notification import send_first_alert, send_vlm_result

logger = logging.getLogger("storeops_ai")


# cv2.putText 는 한글을 그리지 못하므로 tracked 영상에는 영문 라벨을 쓴다.
CATEGORY_EN = {"파손": "BROKEN", "쓰러짐": "FALL", "쓰레기 투기": "LITTERING",
               "싸움": "FIGHT"}


class Path1BehaviorPipeline:
    """FR-EVT-01~08 + FR-EVT-13~16 기능 연결.

    01~04: 기존 YOLO/ByteTrack/crop/2~4초 clip 흐름 유지
    05: 학습된 S3D 분류기(models/i3d_classifier.py)로 점수 산출
    06~08: 임계값 이상이면 사건 후보/번호/알림/10초 영상 저장
    13~15: 대표 이미지 3장 + VLM payload/비동기 연결
    """
    def __init__(self, camera_id="CAM001", i3d_weight_path=None, debug=False, vlm_transport=None):
        self.camera_id = camera_id
        self.debug = debug
        self.detector = YOLODetector(YOLO_MODEL_PATH, TRACKER_CONFIG, CONF_THRESHOLD, PERSON_CLASS_ID)
        self.i3d = I3DClassifier(i3d_weight_path)
        self.vlm = VLMAnalyzer(vlm_transport)
        self.events = EventManager(EVENT_DIR)
        self.track_buffers = defaultdict(lambda: deque(maxlen=240))
        self.issued_events = []
        self._pending_video_events = []
        # (범위, 카테고리) -> 마지막 사건 발급 시각(영상 시간, 초). 같은 사건의 중복 발급 방지.
        self._last_event_time = {}
        # frame_no -> (카테고리, 점수). 임계값을 넘은 분류 구간을 tracked 영상에 빨갛게 표시하기 위한 기록.
        self._alert_frames = {}

    @staticmethod
    def _box_from_result(result, i):
        box = result.boxes.xyxy[i].cpu().numpy().tolist()
        conf = float(result.boxes.conf[i].cpu().item())
        track_id = int(result.boxes.id[i].cpu().item()) if result.boxes.id is not None else -1
        x1, y1, x2, y2 = map(int, box)
        return track_id, x1, y1, x2, y2, conf

    @staticmethod
    def _expand_box(x1, y1, x2, y2, frame_shape):
        h, w = frame_shape[:2]
        bw, bh = x2 - x1, y2 - y1
        return (
            max(0, int(x1 - bw * EXPAND_X)), max(0, int(y1 - bh * EXPAND_Y)),
            min(w, int(x2 + bw * EXPAND_X)), min(h, int(y2 + bh * EXPAND_Y)),
        )

    def _crop(self, frame, box, expand=False):
        x1, y1, x2, y2 = box
        if expand:
            x1, y1, x2, y2 = self._expand_box(x1, y1, x2, y2, frame.shape)
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size else frame.copy()

    @staticmethod
    def _annotate(frame, detections, alert=None):
        """alert=(카테고리, 점수) 이면 빨강(사건 구간), 없으면 초록(정상/미판정)."""
        out = frame.copy()
        color = (0, 0, 255) if alert else (0, 255, 0)   # BGR: 빨강 / 초록
        thick = 3 if alert else 2
        for tid, x1, y1, x2, y2, conf in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            cv2.putText(out, f"ID:{tid} Person {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if alert:
            category, score = alert
            label = CATEGORY_EN.get(category, str(category))
            h, w = out.shape[:2]
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)   # 화면 테두리
            cv2.putText(out, f"EVENT: {label} {score:.2f}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        return out

    def _queue_event_video(self, event, all_frames, fps, trigger_frame):
        """정책: 판정 시점 전 5초/후 5초. 후행 프레임이 오면 10초 영상을 확정한다."""
        pre = max(1, int(EVENT_VIDEO_PRE_SEC * fps))
        post = max(1, int(EVENT_VIDEO_POST_SEC * fps))
        # 총 길이는 정확히 EVENT_VIDEO_SEC: trigger 직전/포함 5초, 이후 5초 미만 구간.
        start = max(0, trigger_frame - pre)
        target_end = trigger_frame + post - 1
        save_frames(all_frames[start:trigger_frame + 1], event.event_video_path, fps)
        event.event_video_complete = False
        updated = self.events.update(event.event_id, event_video_complete=False)
        event_hub.publish("event.updated", updated)
        self._pending_video_events.append({"event": event, "start": start, "end": target_end})

    def _finalize_due_event_videos(self, all_frames, fps, frame_no, force=False):
        remaining = []
        for pending in self._pending_video_events:
            if not force and frame_no < pending["end"]:
                remaining.append(pending)
                continue
            event = pending["event"]
            available_end = min(pending["end"], len(all_frames) - 1)
            save_frames(all_frames[pending["start"]:available_end + 1], event.event_video_path, fps)
            event.event_video_complete = available_end == pending["end"]
            updated = self.events.update(event.event_id, event_video_complete=event.event_video_complete)
            event_hub.publish("event.updated", updated)
        self._pending_video_events = remaining

    def _persist_vlm_result(self, event, future):
        try:
            result = future.result()
            status = result.pop("status", "completed").lower() if isinstance(result, dict) else "completed"
            if status == "vlm_not_configured":
                error = "QWEN_VLM_BASE_URL이 설정되지 않았습니다."
                updated = self.events.update(event.event_id, vlm={"status": "failed", "error": error})
                send_vlm_result(event, "failed", error=error)
            else:
                updated = self.events.update(event.event_id, vlm={"status": "completed", "result": result})
                send_vlm_result(event, "completed", result=result)
            event_hub.publish("event.updated", updated)
        except Exception as exc:
            logger.exception("[VLM FAILED] event=%s", event.event_id)
            updated = self.events.update(event.event_id, vlm={"status": "failed", "error": str(exc)})
            send_vlm_result(event, "failed", error=str(exc))
            event_hub.publish("event.updated", updated)

    def _make_event(self, category, confidence, scores, track_ids, clip_frames,
                    event_frames, fps, start_frame, end_frame, start_time, end_time):
        event_id = self.events.next_event_id()
        clip_path = CLIP_DIR / f"{event_id}_2to4sec.mp4"
        video_path = CLIP_DIR / f"{event_id}_10sec.mp4"
        save_frames(clip_frames, clip_path, fps)
        save_frames(event_frames, video_path, fps)
        try:
            images = save_representative_images(clip_frames, IMAGE_DIR, event_id)
        except ValueError:
            # 대표 이미지를 못 만들어도 사건 저장·알림은 계속 진행한다(NFR-03).
            logger.exception("[대표 이미지 실패] event=%s", event_id)
            images = []

        event = EventCandidate(
            event_id=event_id, camera_id=self.camera_id, event_type="행동",
            category=category, confidence=float(confidence),
            threshold=CATEGORY_THRESHOLDS[category], scores=scores,
            track_ids=list(track_ids), clip_start_frame=start_frame, clip_end_frame=end_frame,
            clip_start_time_sec=start_time, clip_end_time_sec=end_time,
            clip_path=str(clip_path), event_video_path=str(video_path),
            representative_images=images, event_video_complete=False, status="사건후보",
            alert_sent=False, vlm={"status": "PENDING"},
        )
        # FR-EVT-06: 저장 후 번호가 붙은 사건 후보
        self.events.save(event)
        # FR-EVT-07: VLM보다 먼저 즉시 알림
        event.alert_sent = send_first_alert(event)
        self.events.save(event)
        event_hub.publish("event.created", event.to_dict())
        # FR-EVT-15: 별도 작업. 실패/지연은 위 저장/알림에 영향 없음.
        try:
            future = self.vlm.analyze_async(
                images,
                {"event_id": event.event_id, "camera_id": event.camera_id},
            )
            future.add_done_callback(lambda done, event=event: self._persist_vlm_result(event, done))
            if self.debug:
                print(f"[VLM 비동기 시작] {event_id}")
        except Exception as exc:
            if self.debug:
                print(f"[VLM 시작 실패] {exc}")
            updated = self.events.update(event_id, vlm={"status": "failed", "error": str(exc)})
            send_vlm_result(event, "failed", error=str(exc))
            event_hub.publish("event.updated", updated)
        return event

    def create_event_from_scores(self, clip_frames, event_frames, fps, track_id,
                                 scores: Dict[str, float], start_frame=0, end_frame=0,
                                 start_time=0.0, end_time=0.0):
        """FR-EVT-05/06/07/08 테스트 및 향후 실제 I3D 연결용.

        실제 I3D가 연결되면 predict() 결과를 그대로 이 함수에 넘기면 된다.
        """
        category, confidence, scores = self.i3d.decide(scores)
        if category in ("정상", "판정 보류"):
            return None
        threshold = CATEGORY_THRESHOLDS[category]
        if confidence <= threshold:
            return None
        event = self._make_event(category, confidence, scores, [track_id], clip_frames,
                                 event_frames, fps, start_frame, end_frame, start_time, end_time)
        self.issued_events.append(event)
        return event

    # ------------------------------------------------------------------ 분류 + 사건 발급

    def _classify(self, clip_frames, score_provider):
        """FR-EVT-05. score_provider 가 있으면 테스트용 점수를, 없으면 학습된 모델을 쓴다."""
        if score_provider is not None:
            return self.i3d.decide(score_provider(clip_frames))
        return self.i3d.predict(clip_frames)

    def _issue(self, scope, category, confidence, scores, track_ids, clip_frames, event_frames,
               fps, start_frame, end_frame, start_time, end_time):
        """FR-EVT-06: 임계값 + 중복(cooldown) 검사를 통과하면 사건 후보를 발급한다."""
        if category in ("정상", "판정 보류"):
            return None
        if confidence <= CATEGORY_THRESHOLDS.get(category, 1.0):
            return None
        key = (scope, category)
        last = self._last_event_time.get(key)
        if last is not None and (end_time - last) < EVENT_COOLDOWN_SEC:
            if self.debug:
                print(f"[중복 억제] {category} {end_time:.1f}s (직전 발급 {last:.1f}s)")
            return None
        event = self._make_event(category, confidence, scores, track_ids, clip_frames,
                                 event_frames, fps, start_frame, end_frame, start_time, end_time)
        self._last_event_time[key] = end_time
        self.issued_events.append(event)
        return event

    def _infer_full(self, small_buf, window_tids, frames, fps, frame_no, score_provider):
        """화면 전체 모드: 최근 I3D_WINDOW_SEC 구간을 분류한다 (학습 때와 같은 입력 방식)."""
        track_ids = sorted(set().union(*[t for _, t in window_tids])) if window_tids else []
        if not track_ids:
            return None   # 사람이 한 명도 추적되지 않은 구간은 분류하지 않는다
        model_input = [img for _, img in small_buf]
        category, confidence, scores = self._classify(model_input, score_provider)
        t_now = frame_no / fps
        logger.info("[I3D] t=%.1fs %s %.3f | %s", t_now, category, confidence,
                    {k: round(v, 3) for k, v in scores.items() if v >= 0.01})

        # 빨간 표시용 기록: 임계값을 넘은 분류 창의 프레임을 표시한다 (cooldown과 무관).
        if (category not in ("정상", "판정 보류")
                and confidence > CATEGORY_THRESHOLDS.get(category, 1.0)):
            start_f = small_buf[0][0] if small_buf else frame_no
            for f in range(start_f, frame_no + 1):
                prev = self._alert_frames.get(f)
                if prev is None or confidence > prev[1]:
                    self._alert_frames[f] = (category, float(confidence))

        # 저장용 clip은 4초, 사건 영상은 판정 시점 기준 전 5초/후 5초로 확정한다.
        clip_n = max(1, int(CLIP_MAX_SEC * fps))
        clip_start = max(0, frame_no - clip_n + 1)
        ev_n = max(1, int(EVENT_VIDEO_PRE_SEC * fps))
        ev_start = max(0, frame_no - ev_n)
        return self._issue(
            "scene", category, confidence, scores, track_ids,
            frames[clip_start:frame_no + 1], frames[ev_start:frame_no + 1],
            fps, clip_start, frame_no, clip_start / fps, t_now)

    def analyze(self, video_path, score_provider: Optional[callable] = None):
        if I3D_INPUT_MODE != "full":
            raise ValueError("행동분류 운영 모드는 화면 전체(full)만 지원합니다.")
        crop_mode = False

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"영상 열기 실패: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames, det_log = [], []
        self._alert_frames = {}
        frame_no = 0

        # 실제 분류기(또는 테스트용 score_provider)가 없으면 사건은 만들 수 없다.
        can_classify = score_provider is not None or self.i3d.is_ready
        if not can_classify:
            logger.warning("I3D 미연결 - 탐지/추적/클립 생성까지만 수행하고 사건은 발급하지 않습니다.")

        win_frames = max(1, int(round(I3D_WINDOW_SEC * fps)))
        interval_frames = max(1, int(round(I3D_INFER_INTERVAL_SEC * fps)))
        sample_every = max(1, int(round(fps / I3D_BUFFER_FPS)))
        model_size = self.i3d.input_size
        small_buf = deque()      # 화면 전체 모드: (frame_no, 축소 프레임)
        window_tids = deque()    # 화면 전체 모드: (frame_no, 그 프레임의 track ID 집합)
        next_infer_full = win_frames - 1
        did_full_infer = False
        next_infer_track = {}    # crop 모드: track ID -> 다음 분류 프레임
        last_detections = []    # YOLO는 무겁기 때문에 분류 버퍼와 같은 주기로만 돌리고 재사용한다.
        # 웹캠은 가끔 한두 프레임을 못 읽고 넘어가므로, 약 1초 넘게 연속 실패할 때만 스트림 종료로 본다.
        consecutive_read_failures = 0
        max_consecutive_read_failures = max(1, int(round(fps)))

        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_consecutive_read_failures:
                    break
                continue
            consecutive_read_failures = 0
            frames.append(frame.copy())
            self._finalize_due_event_videos(frames, fps, frame_no)
            if frame_no % sample_every == 0:
                result = self.detector.track(frame)
                detections = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for i in range(len(result.boxes)):
                        d = self._box_from_result(result, i)
                        if d[0] >= 0:
                            detections.append(d)
                last_detections = detections
            else:
                detections = last_detections
            det_log.append(detections)

            # ---- 화면 전체 모드: 분류용 버퍼(3fps)와 최근 track ID 기록
            if can_classify and not crop_mode:
                if frame_no % sample_every == 0:
                    small_buf.append((frame_no, cv2.resize(
                        frame, (model_size, model_size), interpolation=cv2.INTER_LINEAR)))
                window_tids.append((frame_no, frozenset(d[0] for d in detections)))
                while small_buf and small_buf[0][0] <= frame_no - win_frames:
                    small_buf.popleft()
                while window_tids and window_tids[0][0] <= frame_no - win_frames:
                    window_tids.popleft()
                if frame_no >= next_infer_full:
                    next_infer_full = frame_no + interval_frames
                    did_full_infer = True
                    event = self._infer_full(small_buf, window_tids, frames, fps, frame_no, score_provider)
                    if event:
                        self._queue_event_video(event, frames, fps, frame_no)

            for tid, x1, y1, x2, y2, conf in detections:
                self.track_buffers[tid].append({
                    "frame_no": frame_no, "time": frame_no / fps,
                    "conf": conf, "box": (x1, y1, x2, y2), "frame": frame.copy(),
                })
                buf = self.track_buffers[tid]
                if len(buf) < int(CLIP_MIN_SEC * fps):
                    continue
                start = max(0, len(buf) - int(CLIP_MAX_SEC * fps))
                clip_items = list(buf)[start:]
                if len(clip_items) < int(CLIP_MIN_SEC * fps):
                    continue

                # FR-EVT-04: 2~4초 입력 clip을 생성할 수 있는 상태.
                # crop 모드에서만 이 clip 이 곧바로 분류 입력이 된다.
                if not crop_mode or not can_classify:
                    continue
                if frame_no < next_infer_track.get(tid, 0):
                    continue
                next_infer_track[tid] = frame_no + interval_frames

                clip_frames = [self._crop(item["frame"], item["box"], expand=False) for item in clip_items]
                category, confidence, scores = self._classify(clip_frames, score_provider)
                logger.info("[I3D] t=%.1fs track=%s %s %.3f", frame_no / fps, tid, category, confidence)
                event_clip_frames = clip_frames
                if category in EXPANDED_CROP_CATEGORIES:
                    event_clip_frames = [self._crop(item["frame"], item["box"], expand=True) for item in clip_items]
                event_frames = frames[max(0, frame_no - int(EVENT_VIDEO_PRE_SEC * fps)):frame_no + 1]
                event = self._issue(f"track{tid}", category, confidence, scores, [tid], event_clip_frames,
                                    event_frames, fps, clip_items[0]["frame_no"], clip_items[-1]["frame_no"],
                                    clip_items[0]["time"], clip_items[-1]["time"])
                if event:
                    self._queue_event_video(event, frames, fps, frame_no)

            frame_no += 1
        cap.release()

        # 영상이 I3D_WINDOW_SEC 보다 짧아 한 번도 분류하지 못했으면, 있는 구간 전체로 1회 분류한다.
        if can_classify and not crop_mode and not did_full_infer and frame_no > 0:
            if len(small_buf) >= 2 and (frame_no / fps) >= CLIP_MIN_SEC:
                logger.info("영상(%.1fs)이 분류 윈도우(%.1fs)보다 짧아 전체 구간을 1회 분류합니다.",
                            frame_no / fps, I3D_WINDOW_SEC)
                event = self._infer_full(small_buf, window_tids, frames, fps, frame_no - 1, score_provider)
                if event:
                    self._queue_event_video(event, frames, fps, frame_no - 1)

        # 입력 영상의 끝에서 후행 5초가 없으면 가능한 구간을 저장하고 incomplete를 명시한다.
        self._finalize_due_event_videos(frames, fps, frame_no, force=True)

        tracked_path = Path(video_path).with_name(Path(video_path).stem + "_tracked.mp4")
        # 분류 결과를 알게 된 뒤에 그린다: 사건 구간은 빨강, 나머지는 초록
        annotated = [self._annotate(f, d, self._alert_frames.get(i))
                     for i, (f, d) in enumerate(zip(frames, det_log))]
        save_frames(annotated, tracked_path, fps)
        if self.debug:
            print(f"[완료] Bounding Box/Track 영상: {tracked_path}")
            print(f"[완료] 이벤트 수: {len(self.issued_events)}")
        return self.issued_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera-id", default="CAM001")
    parser.add_argument("--i3d-weight")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    pipeline = Path1BehaviorPipeline(args.camera_id, args.i3d_weight, args.debug)
    events = pipeline.analyze(args.video)
    print(json.dumps([e.to_dict() for e in events], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()