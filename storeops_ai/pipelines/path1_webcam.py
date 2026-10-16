from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# 이 파일을 `python pipelines\\path1_webcam.py`로 실행해도
# C:\\storeops_ai의 config / models / utils를 찾을 수 있도록 프로젝트 루트를 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from config.config import (
    CATEGORY_THRESHOLDS,
    CROP_LARGEST_PERSON,
    VLM_ON_EVENT,
    CLIP_MAX_SEC,
    CLIP_MIN_SEC,
    CLIP_DIR,
    PERSON_TOP_K, PERSON_MERGE_IOU, PERSON_EDGE_MARGIN, PERSON_MIN_CONF,
    STATIC_OBJECT_ENABLED, STATIC_ABANDONED_CATEGORIES, STATIC_REMOVED_CATEGORIES, STATIC_SCALE, STATIC_PROC_FPS, STATIC_LONG_PERIOD_SEC,
    STATIC_RATIO, STATIC_MAX_EVIDENCE, STATIC_DECAY, STATIC_LONG_ALPHA, STATIC_MIN_AREA_FRAC, STATIC_HOLD_SEC,
    STATIC_PERSON_PAD, STATIC_GROW_TOL, STATIC_SIMILAR_RATIO, STATIC_PENDING_SEC, STATIC_OWNER_SEC, STATIC_OWNER_RADIUS, STATIC_WRIST_CONF, STATIC_DEPART_SEC,
    EVENT_DIR,
    EVENT_VIDEO_SEC,
    EVENT_VIDEO_PRE_SEC,
    EVENT_VIDEO_POST_SEC,
    EXPAND_X,
    EXPAND_Y,
    EXPANDED_CROP_CATEGORIES,
    IMAGE_DIR,
    I3D_BUFFER_FPS,
    I3D_INFER_INTERVAL_SEC,
    I3D_WINDOW_SEC,
    YOLO_MODEL_PATH,
    TRACKER_CONFIG,
    CONF_THRESHOLD,
    PERSON_CLASS_ID,
)
from models.i3d_classifier import I3DClassifier
from models.stack_head import largest_person_boxes, union_crop
from models.vlm_analyzer import VLMAnalyzer
from models.yolo_detector import YOLODetector
from utils.person_filter import select_people
from utils.static_object import StaticObjectDetector
from services.realtime import event_hub
from schemas.event_schema import EventCandidate
from utils.clip_manager import save_frames, save_representative_images
from utils.event_manager import EventManager
from utils.notification import send_event_media, send_first_alert, send_vlm_result
from webcam_i3d import draw_score_panel

logger = logging.getLogger("storeops_ai")


# ---------------------------------------------------------------------------
# 웹캠에서 S3D가 한 번 잘못 분류되는 것을 바로 이벤트로 만들지 않기 위한 설정
# ---------------------------------------------------------------------------
# 기본값 1 = 임계값을 넘으면 바로 이벤트 후보로 확정한다(즉시 알림, 사용자 결정 2026-10-07).
# 연속 확인이 필요하면 환경변수 CONFIRMATIONS_REQUIRED=3처럼 올린다(이전 기본값은 3).
CONFIRMATIONS_REQUIRED = int(os.getenv("CONFIRMATIONS_REQUIRED", "1"))

# 연속 확인 사이의 최대 허용 시간. 웹캠 프레임 드롭 등으로 한 번 늦어져도
# 바로 streak를 0으로 만들지 않도록 한다.
CONFIRMATION_MAX_GAP_SEC = max(8.0, I3D_INFER_INTERVAL_SEC * 2.5)


class Path1Webcam:
    """웹캠 실시간 FR-EVT Path 1.

    흐름:
        웹캠
          -> YOLO 사람 검출
          -> ByteTrack Track ID
          -> S3D 행동 분류
          -> 같은 이상행동이 여러 번 연속 확인되는지 검사
          -> 임계값 + 연속 확인 조건 통과
          -> E001 형식 사건 후보 저장

    이번 버전의 핵심 변경:
      1. 단 한 번의 S3D 오인식으로 E001을 만들지 않는다.
      2. 동일 category가 추적 번호와 무관하게 N회 연속 임계값 이상일 때만 이벤트 생성.
      3. 중간에 다른 category/정상이 나오면 확인 횟수를 리셋한다.
      4. 같은 사람(추적 번호) + 같은 행동은 그 사람이 사라지거나 행동이 바뀔 때까지 다시 발급하지 않는다.
      5. 프로젝트 루트를 sys.path에 넣어 직접 실행 시 `No module named config`를 방지한다.
    """

    def __init__(
        self,
        camera_id: str = "CAM001",
        i3d_weight_path: Optional[str] = None,
        mode: str = "full",
        debug: bool = False,
        confirmations_required: int = CONFIRMATIONS_REQUIRED,
    ):
        if mode != "full":
            raise ValueError("운영 행동분류 모드는 화면 전체(full)만 지원합니다.")
        self.camera_id = camera_id
        self.mode = mode
        self.debug = debug
        self.confirmations_required = max(1, int(confirmations_required))

        self.detector = YOLODetector(
            YOLO_MODEL_PATH,
            TRACKER_CONFIG,
            CONF_THRESHOLD,
            PERSON_CLASS_ID,
        )
        self.i3d = I3DClassifier(i3d_weight_path)
        self.vlm = VLMAnalyzer()
        self.events = EventManager(EVENT_DIR)

        # Track별 사람 영역 버퍼. crop 모드에서 사용.
        self.track_buffers = defaultdict(lambda: deque(maxlen=240))

        # S3D용 장면 버퍼는 학습 입력 FPS(기본 3fps)에 맞춰 저장한다.
        # 기존 30fps 전체 프레임을 60초 보관하던 방식보다 메모리를 크게 줄인다.
        self.scene_frames = deque(
            maxlen=max(24, int(I3D_WINDOW_SEC * I3D_BUFFER_FPS) + 10)
        )
        # C1(사람 크롭) 입력용: 표본 프레임마다 (frame_no, [(track_id, 박스)]). scene_frames와 같은 길이로 유지한다.
        self.window_boxes = deque(maxlen=self.scene_frames.maxlen)

        # 이벤트 영상 저장용 최근 실제 프레임. 최대 10초 정도만 보관한다.
        self.event_frames = deque(
            maxlen=max(30, int(EVENT_VIDEO_SEC * 30.0) + 30)
        )

        # 이미 발급한 (추적 번호, 행동). 사람이 사라지거나 행동이 바뀌면 해제한다.
        self.issued_active = set()
        self.latest_keypoints = {}   # 추적 번호 -> (관절 좌표, 신뢰도). 관절 헤드가 생기면 쓴다
        self.wrist_trail = deque()   # (시각, [(번호, 손목 x, y, 반경)]): 물건 영역의 주인을 찾으려고 최근 것만 둔다
        self._region_printed_at = -1e9
        self.last_seen = {}   # 추적 번호 -> 마지막으로 보인 시각(주인이 떠났는지 판단)
        self.pending = {}   # 물건 확인을 기다리는 후보: 행동 -> (후보 시각, _issue 인자)
        self.static_objects = StaticObjectDetector(
            STATIC_SCALE, STATIC_PROC_FPS, STATIC_LONG_PERIOD_SEC, STATIC_LONG_ALPHA, STATIC_RATIO,
            STATIC_MAX_EVIDENCE, STATIC_DECAY, STATIC_MIN_AREA_FRAC, STATIC_HOLD_SEC, STATIC_PERSON_PAD,
            STATIC_GROW_TOL, STATIC_SIMILAR_RATIO,
        ) if STATIC_OBJECT_ENABLED else None
        self.events_issued = []
        # 사건이 생긴 시점 후 5초가 지나면 전/후 5초 영상을 확정한다.
        self._pending_video_events = []
        self._media_executor = ThreadPoolExecutor(max_workers=1)

        # key=(track scope, category) -> 확인 상태
        self.confirmation_state = {}

    @staticmethod
    def _box(result, i):
        xyxy = result.boxes.xyxy[i].cpu().numpy().tolist()
        conf = float(result.boxes.conf[i].cpu().item())
        tid = (
            int(result.boxes.id[i].cpu().item())
            if result.boxes.id is not None
            else -1
        )
        x1, y1, x2, y2 = map(int, xyxy)
        return tid, x1, y1, x2, y2, conf

    @staticmethod
    def _expand(box, shape):
        x1, y1, x2, y2 = box
        h, w = shape[:2]
        bw, bh = x2 - x1, y2 - y1
        return (
            max(0, int(x1 - bw * EXPAND_X)),
            max(0, int(y1 - bh * EXPAND_Y)),
            min(w, int(x2 + bw * EXPAND_X)),
            min(h, int(y2 + bh * EXPAND_Y)),
        )

    def _crop(self, frame, box, expand=False):
        if expand:
            box = self._expand(box, frame.shape)
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size else frame.copy()

    def _reset_confirmation(self, scope=None):
        """확인 streak를 초기화한다. scope가 없으면 전체 초기화."""
        if scope is None:
            self.confirmation_state.clear()
        else:
            for key in list(self.confirmation_state):
                if key[0] == scope:
                    self.confirmation_state.pop(key, None)

    def _persist_vlm_result(self, event, future):
        """VLM 성공/실패는 사건과 알림을 막지 않고 사건 JSON에 나중에 반영한다."""
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

    def _frames_between(self, start_frame, end_frame):
        return [frame for frame_no, frame in self.event_frames if start_frame <= frame_no <= end_frame]

    def _queue_event_video(self, event, trigger_frame, fps):
        """판정 시점 전 5초 + 후 5초 정책으로 실시간 사건 영상을 확정한다."""
        pre = max(1, int(EVENT_VIDEO_PRE_SEC * fps))
        post = max(1, int(EVENT_VIDEO_POST_SEC * fps))
        start = max(0, trigger_frame - pre)
        target_end = trigger_frame + post - 1
        save_frames(self._frames_between(start, trigger_frame), event.event_video_path, fps)
        updated = self.events.update(event.event_id, event_video_complete=False)
        event_hub.publish("event.updated", updated)
        self._pending_video_events.append({"event": event, "start": start, "end": target_end})

    def _finalize_due_event_videos(self, frame_no, fps, force=False):
        remaining = []
        for pending in self._pending_video_events:
            if not force and frame_no < pending["end"]:
                remaining.append(pending)
                continue
            event = pending["event"]
            end = min(frame_no, pending["end"])
            save_frames(self._frames_between(pending["start"], end), event.event_video_path, fps)
            updated = self.events.update(event.event_id, event_video_complete=end == pending["end"])
            event_hub.publish("event.updated", updated)
            self._media_executor.submit(send_event_media, event)
        self._pending_video_events = remaining

    def _update_confirmation(self, category, confidence, track_ids, current_time):
        """S3D 결과를 이벤트 확정용 streak로 변환한다.

        반환값:
            confirmed: 이번 결과로 이벤트 생성 조건을 충족했는지
            count: 현재 연속 확인 횟수
            required: 필요한 횟수
        """
        # 정상/판정 보류는 이벤트 후보가 아니므로 이상행동 streak를 끊는다.
        if category in ("정상", "판정 보류"):
            self._reset_confirmation()
            return False, 0, self.confirmations_required

        threshold = CATEGORY_THRESHOLDS.get(category, 1.0)
        scope = ("scene",)   # 행동 인식이 목적이라 추적 번호가 바뀌어도 연속 확인은 이어 간다

        # 임계값 미만이면 단발성/약한 분류로 보고 streak를 끊는다.
        if confidence <= threshold:
            self._reset_confirmation(scope)
            return False, 0, self.confirmations_required

        # 동일 scope + 동일 category가 연속되는지 확인한다.
        key = (scope, category)
        previous = self.confirmation_state.get(key)

        if previous is None:
            count = 1
        else:
            gap = current_time - previous["time"]
            if gap <= CONFIRMATION_MAX_GAP_SEC:
                count = previous["count"] + 1
            else:
                count = 1

        self.confirmation_state[key] = {
            "count": count,
            "time": current_time,
            "confidence": float(confidence),
        }

        # 같은 scope에서 다른 category가 나왔다면 그 category streak는 제거한다.
        for other_key in list(self.confirmation_state):
            if other_key != key and other_key[0] == scope:
                self.confirmation_state.pop(other_key, None)

        return count >= self.confirmations_required, count, self.confirmations_required

    def _make_event(
        self,
        category,
        confidence,
        scores,
        track_ids,
        clip_frames,
        event_frames,
        fps,
        start_frame,
        end_frame,
        start_time,
        end_time,
    ):
        event_id = self.events.next_event_id()
        clip_path = CLIP_DIR / f"{event_id}_2to4sec.mp4"
        video_path = CLIP_DIR / f"{event_id}_10sec.mp4"

        save_frames(clip_frames, clip_path, fps)
        save_frames(event_frames, video_path, fps)
        images = save_representative_images(clip_frames, IMAGE_DIR, event_id)

        event = EventCandidate(
            event_id=event_id,
            camera_id=self.camera_id,
            event_type="행동",
            category=category,
            confidence=float(confidence),
            threshold=CATEGORY_THRESHOLDS[category],
            scores=scores,
            track_ids=list(track_ids),
            clip_start_frame=start_frame,
            clip_end_frame=end_frame,
            clip_start_time_sec=start_time,
            clip_end_time_sec=end_time,
            clip_path=str(clip_path),
            event_video_path=str(video_path),
            representative_images=images,
            event_video_complete=False,
            status="사건후보",
            alert_sent=False,
            vlm={"status": "PENDING"},
        )

        self.events.save(event)
        try:
            event.alert_sent = bool(send_first_alert(event))
        except Exception as exc:
            if self.debug:
                print(f"[알림 생략] {exc}")
        self.events.save(event)
        event_hub.publish("event.created", event.to_dict())
        if not VLM_ON_EVENT:   # 사건 시점에는 VLM을 부르지 않는다. 백엔드 화면에는 "해당 없음"으로 표시한다.
            updated = self.events.update(event_id, vlm={"status": "not_applicable"})
            send_vlm_result(event, "not_applicable")
            event_hub.publish("event.updated", updated)
        else:
            try:
                future = self.vlm.analyze_async(
                    images,
                    {"event_id": event.event_id, "camera_id": event.camera_id},
                )
                future.add_done_callback(
                    lambda done, event=event: self._persist_vlm_result(event, done)
                )
            except Exception as exc:
                logger.exception("[VLM START FAILED] event=%s", event_id)
                updated = self.events.update(event_id, vlm={"status": "FAILED", "error": str(exc)})
                event_hub.publish("event.updated", updated)
        self.events_issued.append(event)

        print(f"\n[EVENT] {event_id} | {category} | {confidence:.3f} | Track={track_ids}")
        print(f"        JSON: {EVENT_DIR / (event_id + '.json')}")
        print(f"        CLIP: {clip_path}")
        return event

    @staticmethod
    def _gate_kind(category):
        return "abandoned" if category in STATIC_ABANDONED_CATEGORIES else "removed" if category in STATIC_REMOVED_CATEGORIES else None

    def _object_ready(self, kind, t):
        """물건 확인이 있고, 그 물건 영역에 가장 가까웠던 손목의 주인이 떠났을 때만 참이다(주인이 아직 화면에 있으면 기다린다)."""
        static = self.static_objects
        if static is None or not static.recently_flagged(t, kind):
            return False
        owner = static.owner.get(kind)
        return owner is None or t - self.last_seen.get(owner, -1e9) >= STATIC_DEPART_SEC

    def _release_pending(self, t):
        """모델이 먼저 이상으로 본 후보에 물건 확인이 뒤늦게 오면 경보를 낸다. 시간이 지난 후보는 버린다."""
        events = []
        for category, (since, args) in list(self.pending.items()):
            if t - since > STATIC_PENDING_SEC:
                del self.pending[category]
            elif self._object_ready(self._gate_kind(category), t):
                del self.pending[category]
                event = self._issue(*args, _release=True)
                if event:
                    events.append(event)
        return events

    def _issue(
        self,
        category,
        confidence,
        scores,
        track_ids,
        clip_frames,
        event_frames,
        fps,
        start_frame,
        end_frame,
        _release=False,
    ):
        """임계값 + 연속 확인 + 같은 사람·행동 중복 금지를 모두 통과했을 때만 이벤트를 만든다."""
        end_time = end_frame / fps
        ids = list(track_ids) or ["scene"]   # 추적된 사람이 없어도 같은 장면의 같은 행동은 한 번만

        # 사람이 화면에서 사라졌거나 행동이 바뀌면 발급 기록을 푼다.
        if not _release:
            self.issued_active = {k for k in self.issued_active if k[0] in ids and k[1] == category}
        # 유기는 새로 멈춘 물건 영역이 최근에 확인됐을 때만 경보한다.
        static = getattr(self, "static_objects", None)
        kind = self._gate_kind(category)
        if kind and static is not None and not _release and not self._object_ready(kind, end_time):
            # 물건 확인이 나중에 오는 경우가 많아(증거를 쌓는 시간) 후보로 보류했다가 확인이 오면 내보낸다.
            self.pending[category] = (end_time, (category, confidence, scores, track_ids, clip_frames, event_frames,
                                                 fps, start_frame, end_frame))
            if self.debug:
                print(f"[물건 확인 대기] {category} {'새로 놓인' if kind == 'abandoned' else '없어진'} 물건 확인을 기다림 (최대 {STATIC_PENDING_SEC:.0f}s)")
            return None

        confirmed, count, required = self._update_confirmation(
            category,
            confidence,
            track_ids,
            end_time,
        )

        if self.debug and category != "정상":
            print(
                f"[확인] {category} {confidence:.3f} "
                f"{count}/{required} | tracks={track_ids}"
            )

        if not confirmed:
            return None

        new_ids = [t for t in ids if (t, category) not in self.issued_active]
        if not new_ids:
            if self.debug:
                print(f"[중복 방지] {category} tracks={track_ids} 이미 발급됨")
            return None
        self.issued_active.update((t, category) for t in new_ids)

        return self._make_event(
            category,
            confidence,
            scores,
            track_ids,
            clip_frames,
            event_frames,
            fps,
            start_frame,
            end_frame,
            start_frame / fps,
            end_time,
        )

    def _draw(self, frame, detections, latest):
        out = frame.copy()

        for tid, x1, y1, x2, y2, conf in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{'ID:' + str(tid) + ' ' if tid >= 0 else ''}Person {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        draw_score_panel(
            out,
            latest["scores"] if latest else {},
            latest["category"] if latest else None,
            latest["confidence"] if latest else 0.0,
            None,
            self.i3d.is_ready,
            self.i3d.meta,
        )

        cv2.putText(
            out,
            "Q/ESC: quit",
            (10, out.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return out

    def _crop_input(self, scene_pairs):
        """C1 입력: 창 안 사람 박스의 합집합(CROP_LARGEST_PERSON=1이면 가장 큰 사람)을 원본 프레임에서 잘라 모델 입력 크기로 줄인다.
        사람이 없으면 화면 전체가 된다. 학습 때와 같은 규칙이다."""
        frame_nos = {fn for fn, _ in scene_pairs}
        per_frame = [boxes for fn, boxes in self.window_boxes if fn in frame_nos]
        boxes = largest_person_boxes(per_frame) if CROP_LARGEST_PERSON else [b for boxes in per_frame for _, b in boxes]
        size = self.i3d.input_size
        return [cv2.resize(union_crop(f, boxes), (size, size), interpolation=cv2.INTER_LINEAR) for _, f in scene_pairs]

    def _run_s3d(self, frames, crop_frames=None):
        """S3D 예외가 전체 웹캠 프로그램을 죽이지 않도록 한 곳에서 처리. A+C1 결합이 있으면 크롭 입력도 함께 쓴다."""
        try:
            if crop_frames is not None and self.i3d.has_stack:
                return self.i3d.predict_pair(frames, crop_frames)
            return self.i3d.predict(frames)
        except Exception as exc:
            print(f"[S3D 오류] {type(exc).__name__}: {exc}")
            return None

    def run(self, camera_index=0):
        if not self.i3d.is_ready:
            raise RuntimeError(
                f"S3D 가중치를 로드하지 못했습니다: {self.i3d.weight_path}"
            )

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"웹캠을 열 수 없습니다. camera={camera_index}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps = 30.0

        sample_every = max(1, int(round(fps / I3D_BUFFER_FPS)))
        infer_every = max(1, int(round(I3D_INFER_INTERVAL_SEC * fps)))
        window_frames = max(1, int(round(I3D_WINDOW_SEC * I3D_BUFFER_FPS)))
        min_track_frames = max(1, int(round(CLIP_MIN_SEC * fps)))
        max_track_frames = max(1, int(round(CLIP_MAX_SEC * fps)))

        frame_no = 0
        next_scene_infer = max(1, int(round(I3D_WINDOW_SEC * fps)))
        next_track_infer = {}
        latest = None
        start_clock = time.time()

        print("[시작] YOLO -> Track ID -> S3D -> 연속확인 -> Event E###")
        print(f"[S3D] {self.i3d.weight_path}")
        print(
            f"[S3D] arch={self.i3d.meta.get('arch')} "
            f"val_acc={self.i3d.meta.get('val_acc')} "
            f"device={self.i3d.meta.get('device')}"
        )
        print(f"[모드] {self.mode} (full 권장)")
        print(
            f"[이벤트 방지] 임계값 + {self.confirmations_required}회 연속 확인 "
            f"+ 같은 사람·행동 재발급 금지"
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[경고] 웹캠 프레임을 읽지 못했습니다.")
                break

            # 이벤트 저장용 실제 프레임 버퍼
            self.event_frames.append((frame_no, frame.copy()))
            self._finalize_due_event_videos(frame_no, fps)

            result = self.detector.track(frame)
            self.latest_keypoints = self.detector.keypoints_by_track(result)
            display_detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    display_detections.append(self._box(result, i))
            detections = select_people(
                [d for d in display_detections if d[0] >= 0], frame.shape, PERSON_EDGE_MARGIN, PERSON_MERGE_IOU, PERSON_TOP_K, PERSON_MIN_CONF,
            )
            tracked = [d for d in display_detections if d[0] >= 0]   # C1 입력용: 학습과 같게 추적된 모든 사람 박스
            if self.static_objects is not None:
                now = frame_no / fps
                for d in display_detections:
                    if d[0] >= 0:
                        self.last_seen[d[0]] = now
                heights = {d[0]: d[4] - d[2] for d in display_detections if d[0] >= 0}
                points = [(tid, float(xy[w][0]), float(xy[w][1]), STATIC_OWNER_RADIUS * heights[tid])
                          for tid, (xy, cf) in self.latest_keypoints.items() if tid in heights and cf is not None
                          for w in (9, 10) if cf[w] >= STATIC_WRIST_CONF]   # COCO 9·10 = 왼·오른 손목
                self.wrist_trail.append((now, points))
                while self.wrist_trail and now - self.wrist_trail[0][0] > STATIC_OWNER_SEC:
                    self.wrist_trail.popleft()
                trail = [p for _, pts in self.wrist_trail for p in pts] if now >= self.static_objects.next_proc else None
                self.static_objects.update(frame, now, [d[1:5] for d in display_detections], trail)
                region = self.static_objects.last_region
                if self.debug and region and region["t"] == now and now - self._region_printed_at >= 1.0:
                    self._region_printed_at = now
                    dist = "손목 없음" if region["dist"] is None else f"손목 번호={region['tid']} 거리={region['dist']:.0f}px 반경={region['r']:.0f}px 비율={region['dist'] / region['r']:.2f}"
                    print(f"[물건 영역] t={now:.1f}s {region['kind']} 영역={region['box']} {dist} 주인 인정={'예' if region['ok'] else '아니오'}")
                for released in self._release_pending(frame_no / fps):
                    self._queue_event_video(released, frame_no, fps)

            # Track별 버퍼
            for tid, x1, y1, x2, y2, conf in detections:
                self.track_buffers[tid].append(
                    {
                        "frame_no": frame_no,
                        "time": frame_no / fps,
                        "box": (x1, y1, x2, y2),
                        "conf": conf,
                        "frame": frame.copy(),
                    }
                )

            # S3D용 장면 프레임은 3fps 등 I3D_BUFFER_FPS에 맞춰 샘플링한다.
            if frame_no % sample_every == 0:
                self.scene_frames.append((frame_no, frame.copy()))
                self.window_boxes.append((frame_no, [(d[0], d[1:5]) for d in tracked]))

            # ------------------------------------------------------------------
            # full 모드: 학습 때와 동일하게 화면 전체를 S3D에 넣는다.
            # 사람 유무로 건너뛰지 않는다: 쓰러져 추적이 끊긴 사람(전도)과 사람이 떠난 직후(유기)가 바로 판정 구간이다.
            # ------------------------------------------------------------------
            if self.mode == "full" and frame_no >= next_scene_infer:
                next_scene_infer = frame_no + infer_every

                if len(self.scene_frames) >= 24:
                    scene_pairs = list(self.scene_frames)[-window_frames:]
                    frames = [f for _, f in scene_pairs]
                    tids = sorted({d[0] for d in detections})

                    result_s3d = self._run_s3d(frames, self._crop_input(scene_pairs) if self.i3d.has_stack else None)
                    if result_s3d is not None:
                        category, confidence, scores = result_s3d
                        current_time = frame_no / fps

                        # 이벤트 확인 횟수를 화면에도 표시한다.
                        if category not in ("정상", "판정 보류"):
                            scope = ("scene",)
                            state = self.confirmation_state.get((scope, category))
                            confirm_count = state["count"] if state else 0
                        else:
                            confirm_count = 0

                        latest = {
                            "category": category,
                            "confidence": confidence,
                            "scores": scores,
                            "confirm_count": confirm_count,
                        }

                        if self.debug:
                            print(
                                f"[S3D] t={current_time:.1f}s "
                                f"{category} {confidence:.3f} tracks={tids} "
                                f"점수={ {k: round(v, 3) for k, v in scores.items()} }"
                            )

                        # 이벤트 후보용 2~4초 클립
                        clip_pairs = list(self.event_frames)[-max(1, int(CLIP_MAX_SEC * fps)):]
                        clip_frames = [f for _, f in clip_pairs]

                        # 이벤트 전후 확인용 영상
                        event_pairs = list(self.event_frames)[-max(1, int(EVENT_VIDEO_SEC * fps)):]
                        event_frames = [f for _, f in event_pairs]

                        event = self._issue(
                            category,
                            confidence,
                            scores,
                            tids,
                            clip_frames,
                            event_frames,
                            fps,
                            clip_pairs[0][0] if clip_pairs else frame_no,
                            clip_pairs[-1][0] if clip_pairs else frame_no,
                        )
                        if event:
                            self._queue_event_video(event, frame_no, fps)

            # ------------------------------------------------------------------
            # crop 모드: Track ID별 사람 영역을 S3D에 넣는다.
            # ------------------------------------------------------------------
            if self.mode == "crop":
                for tid, *_ in detections:
                    buf = self.track_buffers[tid]
                    if len(buf) < min_track_frames:
                        continue
                    if frame_no < next_track_infer.get(tid, 0):
                        continue

                    next_track_infer[tid] = frame_no + infer_every
                    items = list(buf)[-max_track_frames:]
                    crop_frames = [
                        self._crop(x["frame"], x["box"], False)
                        for x in items
                    ]

                    result_s3d = self._run_s3d(crop_frames)
                    if result_s3d is None:
                        continue

                    category, confidence, scores = result_s3d
                    if category not in ("정상", "판정 보류"):
                        state = self.confirmation_state.get(((tid,), category))
                        confirm_count = state["count"] if state else 0
                    else:
                        confirm_count = 0

                    latest = {
                        "category": category,
                        "confidence": confidence,
                        "scores": scores,
                        "confirm_count": confirm_count,
                    }

                    if self.debug:
                        print(
                            f"[S3D] t={frame_no/fps:.1f}s "
                            f"track={tid} {category} {confidence:.3f}"
                        )

                    event_clip = crop_frames
                    if category in EXPANDED_CROP_CATEGORIES:
                        event_clip = [
                            self._crop(x["frame"], x["box"], True)
                            for x in items
                        ]

                    event_pairs = list(self.event_frames)[-max(1, int(EVENT_VIDEO_SEC * fps)):]
                    event_frames = [f for _, f in event_pairs]

                    event = self._issue(
                        category,
                        confidence,
                        scores,
                        [tid],
                        event_clip,
                        event_frames,
                        fps,
                        items[0]["frame_no"],
                        items[-1]["frame_no"],
                    )
                    if event:
                        self._queue_event_video(event, frame_no, fps)

            annotated = self._draw(frame, display_detections, latest)
            cv2.imshow("StoreOps AI - WebCam", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

            frame_no += 1

        # 사용자가 종료했거나 카메라가 끊겨 후행 5초가 부족하면 incomplete로 확정한다.
        self._finalize_due_event_videos(frame_no, fps, force=True)
        self._media_executor.shutdown(wait=True)
        cap.release()
        cv2.destroyAllWindows()
        elapsed = max(time.time() - start_clock, 0.001)
        print(
            f"[종료] 처리 프레임={frame_no}, "
            f"실제시간={elapsed:.1f}s, 이벤트={len(self.events_issued)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="StoreOps AI FR-EVT Path1 webcam"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-id", default="CAM001")
    parser.add_argument("--i3d-weight", default=None)
    parser.add_argument(
        "--mode",
        choices=["full"],
        default="full",
        help="화면 전체 입력(학습 분포와 동일)",
    )
    parser.add_argument(
        "--confirmations",
        type=int,
        default=CONFIRMATIONS_REQUIRED,
        help="이상행동 이벤트 확정에 필요한 연속 S3D 확인 횟수",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = Path1Webcam(
        args.camera_id,
        args.i3d_weight,
        args.mode,
        args.debug,
        args.confirmations,
    )
    app.run(args.camera)


if __name__ == "__main__":
    main()
