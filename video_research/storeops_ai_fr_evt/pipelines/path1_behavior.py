from __future__ import annotations
import argparse
import cv2
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

from config.config import (
    YOLO_MODEL_PATH, TRACKER_CONFIG, CONF_THRESHOLD, PERSON_CLASS_ID,
    CLIP_MIN_SEC, CLIP_MAX_SEC, EVENT_VIDEO_SEC, CATEGORY_THRESHOLDS,
    EXPAND_X, EXPAND_Y, EVENT_DIR, CLIP_DIR, IMAGE_DIR,
)
from models.yolo_detector import YOLODetector
from models.i3d_classifier import I3DClassifier, I3DNotConfiguredError
from models.vlm_analyzer import VLMAnalyzer
from schemas.event_schema import EventCandidate
from utils.clip_manager import save_frames, save_representative_images
from utils.event_manager import EventManager
from utils.notification import send_first_alert


class Path1BehaviorPipeline:
    """FR-EVT-01~08 + FR-EVT-13~16 기능 연결.

    01~04: 기존 YOLO/BoT-SORT/crop/2~4초 clip 흐름 유지
    05: I3D 점수 인터페이스만 제공(실제 모델 없음)
    06~08: 점수 주입 시 사건 후보/번호/알림/10초 영상 저장
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
    def _annotate(frame, detections):
        out = frame.copy()
        for tid, x1, y1, x2, y2, conf in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"ID:{tid} Person {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return out

    @staticmethod
    def _event_video_frames(all_frames, start_index, fps):
        n = max(1, int(EVENT_VIDEO_SEC * fps))
        return all_frames[start_index:min(len(all_frames), start_index + n)]

    def _make_event(self, category, confidence, scores, track_id, clip_frames,
                    event_frames, fps, start_frame, end_frame, start_time, end_time):
        event_id = self.events.next_event_id()
        clip_path = CLIP_DIR / f"{event_id}_2to4sec.mp4"
        video_path = CLIP_DIR / f"{event_id}_10sec.mp4"
        save_frames(clip_frames, clip_path, fps)
        save_frames(event_frames, video_path, fps)
        images = save_representative_images(clip_frames, IMAGE_DIR, event_id)

        event = EventCandidate(
            event_id=event_id, camera_id=self.camera_id, event_type="행동",
            category=category, confidence=float(confidence), scores=scores,
            track_ids=[track_id], clip_start_frame=start_frame, clip_end_frame=end_frame,
            clip_start_time_sec=start_time, clip_end_time_sec=end_time,
            clip_path=str(clip_path), event_video_path=str(video_path),
            representative_images=images, status="사건후보", alert_sent=False,
        )
        # FR-EVT-06: 저장 후 번호가 붙은 사건 후보
        self.events.save(event)
        # FR-EVT-07: VLM보다 먼저 즉시 알림
        event.alert_sent = send_first_alert(event)
        self.events.save(event)
        # FR-EVT-15: 별도 작업. 실패/지연은 위 저장/알림에 영향 없음.
        try:
            future = self.vlm.analyze_async(images, event.to_dict())
            # future를 기다리지 않는다.
            if self.debug:
                print(f"[VLM 비동기 시작] {event_id}")
        except Exception as exc:
            if self.debug:
                print(f"[VLM 시작 실패] {exc}")
        return event

    def create_event_from_scores(self, clip_frames, event_frames, fps, track_id,
                                 scores: Dict[str, float], start_frame=0, end_frame=0,
                                 start_time=0.0, end_time=0.0):
        """FR-EVT-05/06/07/08 테스트 및 향후 실제 I3D 연결용.

        실제 I3D가 연결되면 predict() 결과를 그대로 이 함수에 넘기면 된다.
        """
        category, confidence, scores = self.i3d.result_from_scores(scores)
        if category == "정상":
            return None
        threshold = CATEGORY_THRESHOLDS[category]
        if confidence < threshold:
            return None
        event = self._make_event(category, confidence, scores, track_id, clip_frames,
                                 event_frames, fps, start_frame, end_frame, start_time, end_time)
        self.issued_events.append(event)
        return event

    def analyze(self, video_path, score_provider: Optional[callable] = None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"영상 열기 실패: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames, annotated = [], []
        frame_no = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame.copy())
            result = self.detector.track(frame, fps)
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    d = self._box_from_result(result, i)
                    if d[0] >= 0:
                        detections.append(d)
            annotated.append(self._annotate(frame, detections))

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
                # 사람의 동작뿐 아니라 물건과 주변 맥락도 필요한 7분류 공통 입력.
                clip_frames = [self._crop(item["frame"], item["box"], expand=True) for item in clip_items]

                # FR-EVT-04: 2~4초 입력 clip을 생성할 수 있는 상태
                if len(clip_frames) < int(CLIP_MIN_SEC * fps):
                    continue

                if score_provider is None:
                    # 실제 I3D 미연결 상태에서는 사건을 임의 생성하지 않는다.
                    try:
                        self.i3d.predict(clip_frames)
                    except I3DNotConfiguredError as exc:
                        if self.debug:
                            print(f"[I3D 대기] {exc}")
                    continue

                scores = score_provider(clip_frames)
                # 점수로 이벤트를 판정할 때만 FR-EVT-06 이후 진행
                category, confidence, checked_scores = self.i3d.result_from_scores(scores)
                if category == "정상" or confidence < CATEGORY_THRESHOLDS.get(category, 1.0):
                    continue
                event_frames = self._event_video_frames(frames, clip_items[0]["frame_no"], fps)
                self._make_event(category, confidence, checked_scores, tid, clip_frames,
                                 event_frames, fps, clip_items[0]["frame_no"], clip_items[-1]["frame_no"],
                                 clip_items[0]["time"], clip_items[-1]["time"])

            frame_no += 1
        cap.release()

        tracked_path = Path(video_path).with_name(Path(video_path).stem + "_tracked.mp4")
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
