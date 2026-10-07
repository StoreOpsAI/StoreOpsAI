import time
import threading
from pathlib import Path
from config.config import CAMERA_DISCONNECT_SEC, EVENT_DIR
from schemas.event_schema import EventCandidate
from services.realtime import event_hub
from utils.event_manager import EventManager
from utils.notification import send_first_alert


class CameraConnectionMonitor:
    """FR-EVT-09~12. 프레임 내용/행동분류는 전혀 확인하지 않는다."""
    def __init__(self, camera_id, disconnect_threshold=CAMERA_DISCONNECT_SEC):
        self.camera_id = camera_id
        self.threshold = float(disconnect_threshold)
        self.last_frame_time = None
        self.disconnected = False
        self.disconnect_started = None
        self.event_created_for_disconnect = False
        self.events = EventManager(EVENT_DIR)
        self.lock = threading.Lock()

    def frame_received(self, timestamp=None):
        now = time.time() if timestamp is None else float(timestamp)
        with self.lock:
            gap = max(0.0, now - self.last_frame_time) if self.last_frame_time is not None else 0.0
            if self.last_frame_time is not None and gap > self.threshold and not self.event_created_for_disconnect:
                self.disconnected = True
                self.disconnect_started = self.last_frame_time
                self._create_disconnect_event(gap)
            was_disconnected = self.disconnected and self.event_created_for_disconnect
            outage_started = self.disconnect_started
            self.last_frame_time = now
            self.disconnected = False
            self.disconnect_started = None
            self.event_created_for_disconnect = False
            if was_disconnected:
                event = EventCandidate(
                    event_id=self.events.next_event_id(), camera_id=self.camera_id,
                    event_type="카메라재연결", category="카메라재연결",
                    confidence=None, scores={}, track_ids=[],
                    clip_path=None, event_video_path=None, representative_images=[],
                    disconnect_seconds=max(0.0, now - outage_started),
                    disconnect_threshold_seconds=self.threshold,
                    status="재연결", alert_sent=False,
                )
                self.events.save(event)
                event.alert_sent = send_first_alert(event)
                self.events.save(event)
                event_hub.publish("event.created", event.to_dict())

    def check(self, timestamp=None):
        now = time.time() if timestamp is None else float(timestamp)
        with self.lock:
            if self.last_frame_time is None:
                return None
            gap = max(0.0, now - self.last_frame_time)
            if gap <= self.threshold or self.event_created_for_disconnect:
                return None
            self.disconnected = True
            self.disconnect_started = self.last_frame_time
            return self._create_disconnect_event(gap)

    def _create_disconnect_event(self, gap):
        event = EventCandidate(
            event_id=self.events.next_event_id(), camera_id=self.camera_id,
            event_type="카메라끊김", category="카메라끊김",
            confidence=None, scores={}, track_ids=[],
            clip_path=None, event_video_path=None, representative_images=[],
            disconnect_seconds=gap, disconnect_threshold_seconds=self.threshold,
            status="끊김", alert_sent=False,
        )
        self.events.save(event)
        event.alert_sent = send_first_alert(event)
        self.events.save(event)
        event_hub.publish("event.created", event.to_dict())
        self.event_created_for_disconnect = True
        return event

    def state(self):
        with self.lock:
            return {
                "camera_id": self.camera_id,
                "connected": not self.disconnected,
                "last_frame_time": self.last_frame_time,
                "disconnect_threshold_seconds": self.threshold,
            }
