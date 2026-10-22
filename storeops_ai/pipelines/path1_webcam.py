from __future__ import annotations

import argparse
import json
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
import numpy as np

from config.config import (
    CATEGORY_THRESHOLDS,
    CROP_LARGEST_PERSON,
    VLM_ON_EVENT, VLM_GATE_ENABLED, VLM_GATE_FAIL_OPEN, VLM_GATE_MAX_FRAMES, BEHAVIOR_DEFINITIONS,
    LIVE_STREAM_ENABLED, LIVE_STREAM_HOST, LIVE_STREAM_PORT, LIVE_STREAM_WAIT_SEC,
    CLIP_MAX_SEC,
    CLIP_MIN_SEC,
    CLIP_DIR,
    PERSON_TOP_K, PERSON_MERGE_IOU, PERSON_EDGE_MARGIN, PERSON_MIN_CONF,
    STATIC_OBJECT_ENABLED, STATIC_ABANDONED_CATEGORIES, STATIC_REMOVED_CATEGORIES, STATIC_SCALE, STATIC_PROC_FPS, STATIC_LONG_PERIOD_SEC,
    STATIC_RATIO, STATIC_MAX_EVIDENCE, STATIC_DECAY, STATIC_LONG_ALPHA, STATIC_MIN_AREA_FRAC, STATIC_HOLD_SEC,
    STATIC_GROW_TOL, STATIC_SIMILAR_RATIO, STATIC_PENDING_SEC, STATIC_WRIST_CONF, STATIC_DEPART_SEC, UNATTEND_HEIGHTS, PERSON_HEIGHT_M,
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
    POSE_IMGSZ,
    STATIC_DETECT_SHADOWS, STATIC_DARK_MIN, STATIC_BG_TAU_SEC, STATIC_HEAL_SEC, STATIC_TEXTURE_SIM, STATIC_TEXTURE_WIN, STATIC_EDGE_RATIO, FALL_LYING_HOLD_SEC, FALL_TORSO_DEG, FALL_KPT_CONF, FALL_GAP_SEC, FALL_SAME_SPOT,
    CAPTURE_WIDTH, CAPTURE_HEIGHT, FPS_MEASURE_FRAMES, MEDIA_DIR, JUDGMENT_LOGIC_ENABLED,
    CALIB_KEY, CALIB_EMPTY_SEC, CALIB_EMPTY_FRAMES, CALIB_LAB_MATCH, TRAIN_EMPTY_LAB_MEAN, TRAIN_EMPTY_LAB_STD, CALIB_DETECTION_RATE, CALIB_WALK_SEC, CALIB_MAX_SEC, TRAIN_EMPTY_LUMA, TRAIN_VP, TRAIN_VP_Y_RANGE, CALIB_DIR,
)
from models.i3d_classifier import I3DClassifier
from models.pose_head import WIN, pick_largest, window_feature
from models.object_detector import ObjectDetector
from utils.relation_features import RelationTracker
from config.config import CAMERA_LOCK, OBJECT_DETECTOR_PATH, STATIC_THEFT_OBJECT_FIRST, STATIC_ABANDON_OBJECT_FIRST, ABANDON_CATEGORY, ALERT_SHOW_SEC
from config.config import HAND_OBJECT_ENABLED, HAND_OBJECT_DIR, HAND_OBJECT_CONF
from utils.hand_object import HandObjectDetector, HandState, draw_hands
from models.stack_head import crop_region
from utils.auto_calibration import AutoCalibrator, load_calibration, load_color_model
from utils.lying_gate import LyingGate
from models.stack_head import largest_person_boxes, union_crop
from models.vlm_analyzer import VLMAnalyzer
from models.yolo_detector import YOLODetector
from utils.camera_open import lock_camera, open_camera
from utils.person_filter import select_people
from utils.hand_box import SIDES as HAND_SIDES, box_gap, hand_box, reach_boxes
from utils.static_object import StaticObjectDetector
from services.realtime import event_hub
from schemas.event_schema import EventCandidate
from utils.clip_manager import save_all_images, save_frames, save_representative_images
from utils.event_manager import EventManager
from utils.notification import send_event_media, send_first_alert, send_vlm_result
from utils.vlm_gate import VLMGate
from utils.live_stream import LiveStream
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
            POSE_IMGSZ,
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
        self.window_pose = deque(maxlen=self.scene_frames.maxlen)   # C1+P: 표본 프레임마다 가장 큰 사람의 (관절 17x3, 박스) 또는 None
        # P+R 헤드(입력 73)면 표본 프레임마다 물건 탐지 + 사람-물건 관계 20채널(연구 wrist_relation과 같은 계산)
        use_rel = self.i3d.has_pose and self.i3d.pose_head.uses_relation
        self.object_detector = ObjectDetector(Path(OBJECT_DETECTOR_PATH)) if use_rel else None
        self.relation = RelationTracker() if use_rel else None
        self.window_rel = deque(maxlen=self.scene_frames.maxlen)

        # 이벤트 영상 저장용 최근 실제 프레임. 최대 10초 정도만 보관한다.
        self.event_frames = deque(
            maxlen=max(30, int(EVENT_VIDEO_SEC * 30.0) + 30)
        )

        # 이미 발급한 (추적 번호, 행동). 경보는 번호마다 행동별 한 번만(사용자 결정 2026-10-21). 번호가 STATIC_DEPART_SEC 넘게 안 보이다 다시 보이면(나갔다 들어옴) 해제한다.
        self.issued_active = set()
        self.theft_at = -1e9   # 마지막 절도 경보 시각: 물건 경로와 모델 경로가 같은 절도를 두 번 내지 않게(사용자 결정 2026-10-21)
        self.issued_spots = []   # (행동, 물건 자리): 같은 자리 같은 행동은 사람 번호와 무관하게 한 번만(사용자 결정 2026-10-21). 다시 보정할 때 푼다
        self.latest_keypoints = {}   # 추적 번호 -> (관절 좌표, 신뢰도). 관절 헤드가 생기면 쓴다
        self._region_printed_at = -1e9
        self.last_seen = {}   # 추적 번호 -> 마지막으로 보인 시각(주인이 떠났는지 판단)
        self.current_boxes = {}   # 이번 프레임에 보이는 사람 박스(추적 번호 -> 박스)
        self.pending = {}   # 물건 확인을 기다리는 후보: 행동 -> (후보 시각, _issue 인자)
        self.static_objects = StaticObjectDetector(
            STATIC_SCALE, STATIC_PROC_FPS, STATIC_LONG_PERIOD_SEC, STATIC_LONG_ALPHA, STATIC_RATIO,
            STATIC_MAX_EVIDENCE, STATIC_DECAY, STATIC_MIN_AREA_FRAC, STATIC_HOLD_SEC,
            STATIC_GROW_TOL, STATIC_SIMILAR_RATIO, STATIC_DETECT_SHADOWS,
            STATIC_BG_TAU_SEC, STATIC_HEAL_SEC, STATIC_TEXTURE_SIM, STATIC_TEXTURE_WIN, edge_ratio=STATIC_EDGE_RATIO,
        ) if STATIC_OBJECT_ENABLED else None
        # 손-물건 접촉 탐지기(Shan 2020): 손 상태(쥠·빈손) 근거와 숨김 후보. 경보는 막지 않고 근거만 붙인다(보조, 2026-10-22 시연용)
        self.hand_obj = HandObjectDetector(HAND_OBJECT_DIR, HAND_OBJECT_CONF) if HAND_OBJECT_ENABLED else None
        self.hand_ts = STATIC_MAX_EVIDENCE / STATIC_PROC_FPS   # Lin 2015 Ts(초): 물건이 멈춘 시점 t0 = 확정 - Ts, 손 근거는 t0 - Ts부터 본다
        self.hand_state = HandState(horizon=4 * self.hand_ts)
        self.hand_banner = ("", -1e9)
        self.fall_streak = (None, None)   # 모델 쓰러짐 연속 판정 (첫 판정 시각, 마지막 판정 시각)
        self.lying_gate = LyingGate(FALL_LYING_HOLD_SEC, FALL_TORSO_DEG, FALL_KPT_CONF, FALL_GAP_SEC, FALL_SAME_SPOT)   # 전도: 누움이 같은 자리에서 3초 이상
        calib_path = Path(CALIB_DIR) / f"{camera_id}.json"
        train_lab = (np.asarray(TRAIN_EMPTY_LAB_MEAN, np.float32), np.asarray(TRAIN_EMPTY_LAB_STD, np.float32)) if CALIB_LAB_MATCH else None   # None이면 감마(빈 화면 Lab 통계는 기록만)
        self.calibrator = AutoCalibrator(CALIB_EMPTY_SEC, CALIB_WALK_SEC, CALIB_MAX_SEC, TRAIN_EMPTY_LUMA, TRAIN_VP, TRAIN_VP_Y_RANGE, calib_path,
                                         CALIB_EMPTY_FRAMES, CALIB_DETECTION_RATE, STATIC_DARK_MIN, train_lab)
        self.calibration = load_calibration(calib_path, TRAIN_VP, train_lab)   # 이전에 이 카메라로 보정했으면 그 값을 쓴다(없으면 보정 없이)
        self._color_model = load_color_model(calib_path)           # 보정 때 만든 빈 화면 색 모델: 물건 놓임·없어짐을 이것과 대조한다(첫 프레임에서 적용)
        self._saved_camera = json.loads(calib_path.read_text(encoding="utf-8")).get("camera") if calib_path.is_file() else None   # 보정 때 고정한 노출·화이트밸런스
        if CAMERA_LOCK and self._color_model is not None and not self._saved_camera:
            print("[보정] 저장된 색 모델은 카메라 자동 조정을 고정하기 전에 만든 것이라 쓰지 않습니다. 보정을 다시 하세요.")
            self._color_model = None
        self._calib_msg = ""
        if self.calibration is not None:
            print(f"[보정] 저장된 보정값 사용: {calib_path}")
        self.events_issued = []
        # 사건이 생긴 시점 후 5초가 지나면 전/후 5초 영상을 확정한다.
        self._pending_video_events = []
        self._media_executor = ThreadPoolExecutor(max_workers=1)
        self.vlm_gate = VLMGate(self.vlm, self.events, event_hub.publish, self._media_executor, VLM_GATE_FAIL_OPEN) if VLM_GATE_ENABLED else None

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

    def _queue_event_video(self, event, trigger_frame, fps, from_frame=None):
        """판정 시점 전 5초 + 후 5초 정책으로 실시간 사건 영상을 확정한다. from_frame을 주면 그 프레임 5초 전부터(물건 경보: 클립 시작 = 물건이 멈춘 무렵)."""
        pre = max(1, int(EVENT_VIDEO_PRE_SEC * fps))
        post = max(1, int(EVENT_VIDEO_POST_SEC * fps))
        start = max(0, min(trigger_frame, trigger_frame if from_frame is None else from_frame) - pre)
        target_end = trigger_frame + post - 1
        save_frames(self._frames_between(start, trigger_frame), event.event_video_path, fps)
        updated = self.events.update(event.event_id, event_video_complete=False)
        if self.vlm_gate is None or self.vlm_gate.is_sent(event.event_id):
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
            if self.vlm_gate is None or self.vlm_gate.is_sent(event.event_id):
                event_hub.publish("event.updated", updated)
            if self.vlm_gate is not None:
                self.vlm_gate.media_ready(event)
            else:
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
        if self.vlm_gate is not None:   # VLM이 맞다고 해야 백엔드·대시보드로 보낸다(결과는 비동기로 vlm_gate가 처리)
            vlm_images = save_all_images(clip_frames, IMAGE_DIR, event_id, VLM_GATE_MAX_FRAMES)   # 대표 3장이 아니라 클립 프레임 전부
            self.vlm_gate.submit(event, vlm_images, BEHAVIOR_DEFINITIONS.get(category, category), len(clip_frames) / max(fps, 1e-6))
            self.events_issued.append(event)
            print(f"\n[EVENT 후보] {event_id} | {category} | {confidence:.3f} | VLM 확인 대기")
            return event
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
        self.shown_alert = (category, confidence, time.monotonic())   # 모델 창 표시용(물건 우선 경보는 모델 판정이 정상이라 그대로는 안 뜸)
        print(f"        JSON: {EVENT_DIR / (event_id + '.json')}")
        print(f"        CLIP: {clip_path}")
        return event

    @staticmethod
    def _gate_kind(category):
        return "abandoned" if category in STATIC_ABANDONED_CATEGORIES else "removed" if category in STATIC_REMOVED_CATEGORIES else None

    def _spot_issued(self, category, box):
        """같은 행동 경보가 이미 이 물건 자리(상자 중심이 서로 안)에 나갔는가."""
        return any(c == category and StaticObjectDetector._same_spot(box, b) for c, b in self.issued_spots)

    def _object_ready(self, kind, t):
        """물건 확인이 있고 그 물건의 주인이 떠났을 때만 참이다. PETS 2006 방치 정의: 주인이 물건에서 b=3m 밖(다른 사람이 근처에 있는 것은 상관없음).
        주인이 화면에 있으면 거리로, 안 보이면 STATIC_DEPART_SEC 동안 안 보였는지로 본다. 사람 키 1.7m로 픽셀 환산(주인 박스 높이 기준).
        ponytail: 주인 번호가 바뀌면(2026-10-20 run4 앉은 사람 1->2) 떠난 것으로 본다. 예전엔 '아무 사람이나 3m 안'으로 막았지만
        여러 사람이 있으면 남이 근처에 있다는 이유로 경보가 막혀 PETS 정의대로 주인만 본다(사용자 결정 2026-10-21)."""
        static = self.static_objects
        if static is None or not static.recently_flagged(t, kind) or not static.still_there(kind):
            return False   # 경보 순간에도 물건(없어진 자리)이 그대로여야 한다(PETS 2006 방치 = 물건이 남아 있음, Porikli 2008 증거 최대)
        owner = static.owner.get(kind)
        if owner is None:
            self.depart_note = "주인 없음"
            return True
        b = self.current_boxes.get(owner)
        if b is not None:
            obj = static.kind_box[kind]   # 종류별 상자(boxes는 마지막 영역으로 덮여 다른 종류 자리일 수 있다, run27 E001·E002)
            gap, h = box_gap(b, obj), b[3] - b[1]
            self.depart_note = f"주인 화면에 있음 빈 거리={gap:.0f}px 키={h:.0f}px 환산={gap / max(h, 1) * PERSON_HEIGHT_M:.1f}m"   # 거리 확인용(run30 5.3m 문의)
            return gap > UNATTEND_HEIGHTS * h
        self.depart_note = f"주인 안 보임 {t - self.last_seen.get(owner, -1e9):.1f}s"
        return t - self.last_seen.get(owner, -1e9) >= STATIC_DEPART_SEC

    def _object_first(self, kind, category, now, frame_no, fps, latest):
        """물건 우선 경보: 절도(removed, 사용자 결정 2026-10-20)와 유기(abandoned, 2026-10-21). 물건 변화 자리에 최근 손 상자(주인)가 닿았고
        그 주인이 떠나면(PETS 2006 b=3m) 모델 점수와 무관하게 경보한다. 시연에서 유기 모델 점수가 임계 아래(최고 0.987 '판정 보류')라 유기가 안 나던 문제 대응.
        근거: 놓인·치워진 물건 Tian 외 IEEE TSMC-C 2011 + 손목 주인 역추적 Lin 외 IEEE TIFS 2015 + 방치 정의 PETS 2006. 같은 주인 번호는 들어온 동안 한 번만(issued_active)."""
        box = self.static_objects.kind_box[kind]
        owner = self.static_objects.owner[kind]
        if (owner, category) in self.issued_active or self._spot_issued(category, box):
            return None
        if category == "절도":
            if now - self.theft_at < STATIC_PENDING_SEC:   # 모델 절도 경보가 방금 나갔으면 같은 사건
                return None
            self.theft_at = now
        self.issued_spots.append((category, box))
        scores = dict(latest["scores"]) if latest else {}
        # 경보는 주인이 떠난 뒤(수 초~수십 초 뒤)라 직전 4초에는 놓거나 집는 동작이 없다(run31 E012 8.5초, run34 E003 22.2초 뒤).
        # 클립은 물건이 멈춘 시점 t0 = 확정 - Ts의 역추적 창 t0 ± Ts로 잡는다(Lin 외 TIFS 2015, 손 근거와 같은 창). 버퍼에 없는 앞부분은 빠진다
        f0, ts = int(round((self.static_objects.last_time[kind] - self.hand_ts) * fps)), max(1, int(round(self.hand_ts * fps)))
        clip_pairs = [(n, f) for n, f in self.event_frames if f0 - ts <= n <= f0 + ts] or list(self.event_frames)[-max(1, int(CLIP_MAX_SEC * fps)):]
        event_pairs = list(self.event_frames)[-max(1, int(EVENT_VIDEO_SEC * fps)):]
        start = clip_pairs[0][0] if clip_pairs else frame_no
        end = clip_pairs[-1][0] if clip_pairs else frame_no
        self.issued_active.add((owner, category))   # 모델 쪽 경로가 같은 사람으로 또 내지 않게
        self.pending.pop(category, None)   # 같은 행동의 모델 후보는 이 경보와 같은 사건이다(시연 run15: 남은 후보가 사람 없이 'scene'으로 또 나감)
        hand = self._hand_evidence(kind, box, owner, now)
        print(f"[물건 우선 {category}] t={now:.1f}s {kind} 자리={box} 손 주인={owner} 모델 점수={scores.get(category, 0.0):.3f} 떠남={getattr(self, 'depart_note', '-')}"
              + (f" 손 근거={hand}" if hand else ""))
        event = self._make_event(category, scores.get(category, 0.0), scores, [owner], [f for _, f in clip_pairs], [f for _, f in event_pairs],
                                 fps, start, end, start / fps, end / fps)
        if hand:
            self.events.update(event.event_id, hand_evidence=hand)
            if hand.get("concealed"):
                self.hand_banner = ("THEFT: CONCEALED (held -> empty hand, item nowhere)", time.monotonic())
        if kind == "abandoned":
            self.static_objects.absorb(box)   # MVP용: _issue와 같게 유기 경보 낸 물건은 배경으로 흡수
        return event

    def _hand_evidence(self, kind, box, owner, now):
        """손-물건 근거(보조, 경보는 막지 않음). t0 = 물건 확정 - Ts(Lin 2015), t0 - Ts 이후를 본다.
        숨김(절도만, 근거 약함): 주인의 같은 손이 쥠 → 빈손으로 보였고, 그 사이 새로 놓인 물건(유기 확정)이 없음."""
        if self.hand_obj is None:
            return None
        static = self.static_objects
        last = static.last_time.get(kind)
        since = (now if last is None else last) - 2 * self.hand_ts
        ev = {"touched": self.hand_state.touched(box, since)}
        if kind == "removed":
            placed = static.last_time.get("abandoned")   # 유기 경보 뒤 흡수하면 None(run33 E001 뒤 TypeError로 시연 중단)
            ev["concealed"] = self.hand_state.concealed(owner, since) and (placed is None or placed < since)
        return ev

    def _fall_held(self):
        """전도: 모델이 쓰러짐으로 연속 판정한 첫 판정부터 마지막 판정까지가 FALL_LYING_HOLD_SEC(사용자 지정 3초) 이상.
        누운 사람 관절 3초 확인(LyingGate)은 누운 사람이 사람으로 검출되지 않으면 영원히 대기해 뺐다(run31: 쓰러짐 1.000이 17초, tracks=[], 사용자 결정 2026-10-22).
        ponytail: 쪼그려 앉기·숙이기 거르기는 이제 모델에 맡긴다. 데모 run1~31 모델 판정 약 2만 5천 번 중 쓰러짐은 연출 전도 9번뿐(오경보 0)."""
        since, last = self.fall_streak
        return since is not None and last - since >= FALL_LYING_HOLD_SEC

    def _release_pending(self, t):
        """모델이 먼저 이상으로 본 후보에 물건 확인이 뒤늦게 오면 경보를 낸다. 시간이 지난 후보는 버린다."""
        events = []
        for category, (since, args) in list(self.pending.items()):
            if t - since > STATIC_PENDING_SEC:
                del self.pending[category]
            elif (self._fall_held() if category == "쓰러짐" else self._object_ready(self._gate_kind(category), t)):
                del self.pending[category]
                print(f"[물건 확인 도착] {category} 떠남={getattr(self, 'depart_note', '-')}")
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

        # 발급 기록은 나갔다 들어올 때(프레임 루프)와 다시 보정할 때만 푼다. 사람 없는 장면 경보("scene")도 행동별 한 번만(사용자 결정 2026-10-21).
        # 유기는 새로 멈춘 물건 영역이 최근에 확인됐을 때만 경보한다.
        static = getattr(self, "static_objects", None)
        kind = self._gate_kind(category)
        if JUDGMENT_LOGIC_ENABLED and category == "쓰러짐" and not _release and not self._fall_held():
            # 전도: 모델 쓰러짐 판정이 FALL_LYING_HOLD_SEC 이상 연속될 때까지 후보로 보류한다(사용자 지정 3초, 2026-10-22)
            self.pending[category] = (end_time, (category, confidence, scores, track_ids, clip_frames, event_frames,
                                                 fps, start_frame, end_frame))
            if self.debug:
                print(f"[쓰러짐 연속 대기] 모델 쓰러짐 판정 {FALL_LYING_HOLD_SEC:.0f}초 연속을 기다림")
            return None
        if JUDGMENT_LOGIC_ENABLED and kind and static is not None and not _release and not self._object_ready(kind, end_time):
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
        if category == "절도" and end_time - self.theft_at < STATIC_PENDING_SEC:   # 물건 우선 절도 경보가 방금 나갔으면 같은 사건
            new_ids = []
        spot = static.kind_box.get(kind) if kind and static is not None else None
        if spot is not None and self._spot_issued(category, spot):
            new_ids = []
        if not new_ids:
            if self.debug:
                print(f"[중복 방지] {category} tracks={track_ids} 이미 발급됨")
            return None
        self.issued_active.update((t, category) for t in new_ids)
        if spot is not None:
            self.issued_spots.append((category, spot))
        if category == "절도":
            self.theft_at = end_time

        event = self._make_event(
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
        if JUDGMENT_LOGIC_ENABLED and kind == "abandoned" and spot is not None:
            static.absorb(spot)   # MVP용(2026-10-21 사용자 결정): 유기 경보를 낸 물건은 바로 배경으로 처리해 다시 잡지 않는다
        return event

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

        if self.hand_obj is not None:
            draw_hands(out, self.hand_obj.latest)
            text, since = self.hand_banner
            if time.monotonic() - since < ALERT_SHOW_SEC:
                cv2.putText(out, text, (360, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        category = latest["category"] if latest else None
        confidence = latest["confidence"] if latest else 0.0
        alert, alert_conf, since = getattr(self, "shown_alert", (None, 0.0, -1e9))
        shown = time.monotonic() - since < ALERT_SHOW_SEC   # 발급 직후 ALERT_SHOW_SEC 동안은 모델 창에 그 경보를 띄운다
        if shown:
            category, confidence = alert, alert_conf
        draw_score_panel(
            out,
            latest["scores"] if latest else {},
            category,
            confidence,
            None,
            self.i3d.is_ready,
            self.i3d.meta,
            issued=shown or any(k[1] == category for k in self.issued_active),   # 모델 점수가 아니라 실제 발급된 경보만 EVENT로 표시
            pending=category in self.pending,                           # 물건 확인·누움 유지를 기다리는 후보
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
        cal = self.calibration
        if cal is None:
            return [cv2.resize(union_crop(f, boxes), (size, size), interpolation=cv2.INTER_LINEAR) for _, f in scene_pairs]
        # 자동 보정: 크롭 중심에서 학습 카메라와의 기울기 차이만큼 회전해 자르고, 밝기를 학습 평균에 맞춘다
        x0, y0, side = crop_region(scene_pairs[0][1].shape, boxes)
        delta = cal.delta_deg(x0 + side / 2, y0 + side / 2) if side > 0 else 0.0
        return [cal.photometric(cv2.resize(cal.rotated_crop(f, x0, y0, side, delta) if side > 0 else f, (size, size), interpolation=cv2.INTER_LINEAR))
                for _, f in scene_pairs]

    def _pose_input(self, scene_pairs):
        """P 입력: 창의 표본 프레임마다 가장 큰 사람의 관절 특징(12x53). 사람이 없는 프레임은 0(학습과 같다)."""
        by_frame = dict(self.window_pose)
        items = [by_frame.get(fn) for fn, _ in scene_pairs]
        if self.calibration is not None:   # 자동 보정: 관절도 크롭과 같은 각도만큼 박스 중심 기준으로 회전
            items = [None if it is None else (self.calibration.rotate_keypoints(it[0], it[1]), it[1]) for it in items]
        if self.relation is None:
            return window_feature(items)
        rel_by_frame = dict(self.window_rel)   # 표본마다 scene_frames와 함께 쌓이므로 창의 표본 번호가 모두 있다
        rel = np.stack([rel_by_frame[fn] for fn, _ in scene_pairs[-WIN:]])   # window_feature도 마지막 12표본을 쓴다
        return np.concatenate([window_feature(items), rel], axis=1)

    def _run_s3d(self, frames, crop_frames=None, pose_window=None):
        """S3D 예외가 전체 웹캠 프로그램을 죽이지 않도록 한 곳에서 처리. C1+P면 크롭과 관절, A+C1이면 크롭 입력도 함께 쓴다."""
        try:
            if pose_window is not None and crop_frames is not None and self.i3d.has_pose:
                return self.i3d.predict_c1p(crop_frames, pose_window)
            if crop_frames is not None and self.i3d.has_stack:
                return self.i3d.predict_pair(frames, crop_frames)
            return self.i3d.predict(frames)
        except Exception as exc:
            print(f"[S3D 오류] {type(exc).__name__}: {exc}")
            return None

    def _start_live(self):
        """점주 대시보드 실시간 화면 서버. 토큰이 없거나 포트를 못 열면 화면만 빼고 탐지는 계속한다."""
        if not LIVE_STREAM_ENABLED:
            return None
        try:
            live = LiveStream(os.getenv("EVENT_INGEST_TOKEN", ""), LIVE_STREAM_WAIT_SEC)
            live.serve(LIVE_STREAM_HOST, LIVE_STREAM_PORT)
        except (ValueError, OSError) as exc:
            print(f"[실시간 화면 꺼짐] {exc}")
            return None
        print(f"[실시간 화면] {LIVE_STREAM_HOST}:{LIVE_STREAM_PORT}/live/frame (토큰 필요)")
        return live

    def run(self, camera_index=0):
        if not self.i3d.is_ready:
            raise RuntimeError(
                f"S3D 가중치를 로드하지 못했습니다: {self.i3d.weight_path}"
            )

        cap = open_camera(camera_index)
        live = self._start_live()
        if CAMERA_LOCK:
            self.calibrator.camera_settings = lock_camera(cap, self._saved_camera if self._color_model is not None else None)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps = 30.0

        infer_every = max(1, int(round(I3D_INFER_INTERVAL_SEC * fps)))
        window_frames = max(1, int(round(I3D_WINDOW_SEC * I3D_BUFFER_FPS)))
        min_track_frames = max(1, int(round(CLIP_MIN_SEC * fps)))
        max_track_frames = max(1, int(round(CLIP_MAX_SEC * fps)))

        frame_no = 0
        next_scene_infer = max(1, int(round(I3D_WINDOW_SEC * fps)))
        next_track_infer = {}
        latest = None
        start_clock = time.time()
        if os.getenv("LAB_AUTO_CALIB") == "1":   # 실험용: 켜자마자 보정 시작('c'를 누른 것과 같다)
            self.calibrator.start(0.0)
        raw_path, raw_writer, raw_ts = os.getenv("LAB_RAW_PATH", ""), None, []
        next_sample_wall = 0.0
        win ="StoreOps AI - WebCam"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, CAPTURE_WIDTH, CAPTURE_HEIGHT)   # 화면도 학습 해상도(1920x1080)로 연다
        calib_btn = (12, 360, 332, 420)                  # 점수 패널 바로 아래 보정 버튼(x0, y0, x1, y1)
        calib_click = []

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and calib_btn[0] <= x <= calib_btn[2] and calib_btn[1] <= y <= calib_btn[3]:
                calib_click.append(True)

        cv2.setMouseCallback(win, _on_mouse)

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
            if raw_path:   # 실험용: 처리한 원본 프레임과 실제 시각을 저장해 replay_static으로 다시 돌린다
                if raw_writer is None:
                    raw_writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (frame.shape[1], frame.shape[0]))   # avi+MJPG: 강제 종료돼도 읽힌다
                raw_writer.write(frame); raw_ts.append(time.time() - start_clock)
                if len(raw_ts) % 100 == 0:
                    np.save(Path(raw_path).with_suffix(".ts.npy"), np.asarray(raw_ts, np.float64))
            if frame_no == FPS_MEASURE_FRAMES:
                # 카메라가 알려 주는 FPS(30)가 아니라 실제 처리 속도로 시간을 센다(1080p에서는 처리가 더 느리다).
                fps = frame_no / max(time.time() - start_clock, 1e-3)
                infer_every = max(1, int(round(I3D_INFER_INTERVAL_SEC * fps)))
                min_track_frames = max(1, int(round(CLIP_MIN_SEC * fps)))
                max_track_frames = max(1, int(round(CLIP_MAX_SEC * fps)))
                next_scene_infer = frame_no + max(1, int(round(I3D_WINDOW_SEC * fps)))
                print(f"[INFO] 실제 처리 속도 {fps:.1f}fps, S3D 표본은 실제 시간 {I3D_BUFFER_FPS}fps")
                if fps < I3D_BUFFER_FPS:
                    print(f"[경고] 처리 속도가 S3D 표본 {I3D_BUFFER_FPS}fps보다 느려 표본 간격이 학습과 다릅니다.")

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
            persons_kpt = [(np.concatenate([xy, np.asarray(cf if cf is not None else np.zeros(len(xy)))[:, None]], axis=1), d[1:5])
                           for d in tracked for tid, (xy, cf) in self.latest_keypoints.items() if tid == d[0]]
            self.lying_gate.update(frame_no / fps, persons_kpt)
            done = self.calibrator.update(frame_no / fps, frame, persons_kpt, len(display_detections))
            if done is not None:
                self.calibration = done
                self._color_model = self.calibrator.color_model
                self.issued_active.clear()
                self.issued_spots.clear()
            if self._color_model is not None and self.static_objects is not None:
                ok = self.static_objects.set_model(self._color_model, frame.shape)
                print("[보정] 물건 판정: 빈 화면 색 모델과 대조합니다" if ok else "[보정] 색 모델 해상도가 카메라와 달라 쓰지 않습니다. 다시 보정해 주세요")
                self._color_model = None
            if self.calibrator.message != self._calib_msg:
                self._calib_msg = self.calibrator.message
                print(f"[보정] {self._calib_msg}")
            if self.static_objects is not None:
                now = frame_no / fps
                for d in display_detections:
                    if d[0] >= 0:
                        if now - self.last_seen.get(d[0], now) > STATIC_DEPART_SEC:   # 나갔다 다시 들어옴: 이 번호의 발급 기록을 푼다
                            self.issued_active = {k for k in self.issued_active if k[0] != d[0]}
                        self.last_seen[d[0]] = now
                self.current_boxes = {d[0]: tuple(d[1:5]) for d in display_detections if d[0] >= 0}
                if self.hand_obj is not None:
                    self.hand_obj.submit(frame, now)   # 별도 스레드가 최신 장만 탐지한다
                    self.hand_state.update(self.hand_obj.latest, self.current_boxes)
                heights = {d[0]: d[4] - d[2] for d in display_detections if d[0] >= 0}
                points = [(tid, *hb) for tid, (xy, cf) in self.latest_keypoints.items() if tid in heights and cf is not None
                          for side in HAND_SIDES if (hb := hand_box(xy, cf, side, STATIC_WRIST_CONF)) is not None]   # OpenPose 손 상자(왼·오른)
                trail = reach_boxes(points, [d[:5] for d in display_detections if d[0] >= 0])   # 주인 후보: 손 상자 + 사람 박스(덮은 화소를 정지 물건 쪽에서 기록)
                self.static_objects.update(frame, now, [d[1:5] for d in display_detections], trail)
                for kind, last in self.static_objects.last_time.items():   # 경보 근거가 되는 '주인 인정' 확인은 빠짐없이 남긴다
                    if self.debug and last == now:
                        print(f"[물건 확인] t={now:.1f}s {kind} 영역={self.static_objects.boxes} 주인 번호={self.static_objects.owner.get(kind)}")
                if self.debug and getattr(self.static_objects, "debug_view", None) is not None and now - getattr(self, "_dbg_saved", -1e9) >= 5.0:
                    self._dbg_saved = now   # 5초마다 정지 물건 단계별 마스크를 덮어써 저장(진단용)
                    cv2.imwrite(str(Path(MEDIA_DIR) / "static_debug.jpg"), self.static_objects.debug_view)
                    print(f"[물건 진단] t={now:.1f}s 기준과다름={self.static_objects.debug_counts[0]} 움직임={self.static_objects.debug_counts[1]} "
                          f"멈춘후보={self.static_objects.debug_counts[2]} 증거가득={self.static_objects.debug_counts[3]} 영역={self.static_objects.debug_counts[4]}")
                for kind, category, on in (("removed", "절도", STATIC_THEFT_OBJECT_FIRST), ("abandoned", ABANDON_CATEGORY, STATIC_ABANDON_OBJECT_FIRST)):
                    if (JUDGMENT_LOGIC_ENABLED and on and self.static_objects.owner.get(kind) is not None
                            and self.calibrator.state not in ("empty", "walk")   # 보정 중에는 기준 화면을 새로 만드는 중이라 판정하지 않는다(run28 E001·E002, run32 E001~E003)
                            and self._object_ready(kind, now)):   # 물건 확인(손 주인 있음) 뒤 그 주인이 떠나면 경보
                        event = self._object_first(kind, category, now, frame_no, fps, latest)
                        if event:
                            self._queue_event_video(event, frame_no, fps, from_frame=event.clip_start_frame)   # 놓기·집기부터 떠남 뒤 5초까지
                region = self.static_objects.last_region
                if self.debug and region and region["t"] == now and now - self._region_printed_at >= 1.0:
                    self._region_printed_at = now
                    dist = "주인 후보 없음" if region["dist"] is None else f"주인 후보={region['tid']} 손·몸 상자와 빈 거리={region['dist']:.0f}px 상자 가로={2 * region['r']:.0f}px"
                    print(f"[물건 영역] t={now:.1f}s {region['kind']} 영역={region['box']} 키운넓이(현재,배경)={region.get('heal')} {dist} 주인 인정={'예' if region['ok'] else '아니오'}")
            for released in self._release_pending(frame_no / fps):   # 물건 확인·누움 유지를 기다리던 후보(정지 물건 검출이 꺼져 있어도 전도는 돌아야 한다)
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
            wall = time.time() - start_clock
            if wall >= next_sample_wall:   # 프레임 수가 아니라 실제 시간으로 I3D_BUFFER_FPS(3fps) 간격을 맞춘다(처리 속도가 변해도 학습과 같은 간격)
                next_sample_wall += 1.0 / I3D_BUFFER_FPS
                if next_sample_wall <= wall:   # 한 간격 넘게 밀렸으면(멈춤 등) 몰아서 뽑지 않고 지금부터 다시 센다
                    next_sample_wall = wall + 1.0 / I3D_BUFFER_FPS
                self.scene_frames.append((frame_no, frame.copy()))
                self.window_boxes.append((frame_no, [(d[0], d[1:5]) for d in tracked]))
                self.window_pose.append((frame_no, pick_largest(persons_kpt)))
                if self.relation is not None:   # 학습과 같게 사람 점수 0.6 이상만(PERSON_MIN_CONF = 연구 WEBCAM_PERSON_CONF)
                    kept = [d for d in tracked if d[5] >= PERSON_MIN_CONF]
                    kept_boxes = {tuple(d[1:5]) for d in kept}
                    kept_kpt = [p for p in persons_kpt if tuple(p[1]) in kept_boxes]
                    obj_boxes, obj_confs = self.object_detector.detect(frame)
                    self.window_rel.append((frame_no, self.relation.step(np.array([d[1:5] for d in kept], np.float32), pick_largest(kept_kpt), obj_boxes, obj_confs)))

            # ------------------------------------------------------------------
            # full 모드: 학습 때와 동일하게 화면 전체를 S3D에 넣는다.
            # 사람 유무로 건너뛰지 않는다: 쓰러져 추적이 끊긴 사람(전도)과 사람이 떠난 직후(유기)가 바로 판정 구간이다.
            # ------------------------------------------------------------------
            if self.mode == "full" and frame_no >= next_scene_infer and self.calibrator.state not in ("empty", "walk"):   # 보정 중에는 판정하지 않는다
                next_scene_infer = frame_no + infer_every

                if len(self.scene_frames) >= 24:
                    scene_pairs = list(self.scene_frames)[-window_frames:]
                    frames = [f for _, f in scene_pairs]
                    tids = sorted({d[0] for d in detections})

                    result_s3d = self._run_s3d(frames, self._crop_input(scene_pairs) if self.i3d.has_stack else None,
                                               self._pose_input(scene_pairs) if self.i3d.has_pose else None)
                    if result_s3d is not None:
                        category, confidence, scores = result_s3d
                        current_time = frame_no / fps
                        win_nos = {fn for fn, _ in scene_pairs}
                        if JUDGMENT_LOGIC_ENABLED and scores.get("쓰레기 투기", 0.0) > 0 and not any(b for fn, b in self.window_boxes if fn in win_nos):
                            # 유기 정의(사람이 들고 있던 물건을 내려놓고 떠남): 창 안에 사람이 한 번도 없으면 투기로 보지 않는다.
                            # 빈·정지 화면에서 C1이 투기로 기우는 문제(학습에 빈 화면 정상이 없음) 대응. 정의 기반 규칙, 근거 약함.
                            cut = {**scores, "쓰레기 투기": 0.0}
                            cut["정상"] = max(0.0, 1.0 - max(v for k, v in cut.items() if k != "정상"))   # predict_c1p와 같은 정상 점수 규칙
                            category, confidence, scores = self.i3d.decide(cut)
                        since = self.fall_streak[0] if self.fall_streak[0] is not None else current_time
                        self.fall_streak = (since, current_time) if category == "쓰러짐" else (None, None)   # 다른 판정이 끼면 처음부터

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
            cal, st = self.calibration, self.calibrator.state
            cal_txt = (f"CALIB 1/2 EMPTY {self.calibrator.empty_time:.0f}/{CALIB_EMPTY_SEC:.0f}s" if st == "empty"
                       else f"CALIB 2/2 WALK lines={len(self.calibrator.lines)}" if st == "walk"
                       else "CALIB FAILED - press C" if st == "failed"
                       else f"CAL gamma={cal.gamma:.2f} tilt={'on' if cal.vp is not None else 'off'} (C: recalibrate)" if cal is not None
                       else "NO CALIB - press C")
            cv2.putText(annotated, cal_txt, (12, annotated.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            busy = st in ("empty", "walk")
            cv2.rectangle(annotated, calib_btn[:2], calib_btn[2:], (80, 80, 80) if busy else (0, 160, 0), -1)
            cv2.putText(annotated, "CALIBRATING..." if busy else "AUTO CALIBRATE", (calib_btn[0] + 14, calib_btn[3] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(win, annotated)
            if live is not None:
                live.publish(frame)   # 대시보드에는 분석 표시 없는 원본 카메라 화면(사용자 결정 2026-10-22). _draw는 복사본에 그린다

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            pressed = calib_click or key in (ord(CALIB_KEY.lower()), ord(CALIB_KEY.upper()))
            calib_click.clear()
            if pressed and not busy:
                self.calibrator.start(frame_no / fps)

            frame_no += 1

        # 사용자가 종료했거나 카메라가 끊겨 후행 5초가 부족하면 incomplete로 확정한다.
        self._finalize_due_event_videos(frame_no, fps, force=True)
        self._media_executor.shutdown(wait=True)
        cap.release()
        if raw_writer is not None:
            raw_writer.release()
            np.save(Path(raw_path).with_suffix(".ts.npy"), np.asarray(raw_ts, np.float64))
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
