"""RTSP/HTTP 카메라 수신과 Path 2 끊김 감시를 연결하는 런타임."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from config.config import CONNECTION_CHECK_INTERVAL_SEC, STREAM_RECONNECT_DELAY_SEC
from pipelines.path2_connection import CameraConnectionMonitor

logger = logging.getLogger("storeops_ai")


@dataclass
class StreamWorker:
    camera_id: str
    stream_url: str
    thread: threading.Thread
    stop_event: threading.Event


class CameraRuntime:
    """카메라별 monitor는 공유하고, check는 별도 스케줄러가 실행한다."""
    def __init__(self):
        self.monitors: Dict[str, CameraConnectionMonitor] = {}
        self.workers: Dict[str, StreamWorker] = {}
        self.lock = threading.RLock()
        self._scheduler_stop = threading.Event()
        self._scheduler: Optional[threading.Thread] = None

    def monitor(self, camera_id: str) -> CameraConnectionMonitor:
        with self.lock:
            if camera_id not in self.monitors:
                self.monitors[camera_id] = CameraConnectionMonitor(camera_id)
            return self.monitors[camera_id]

    def start(self):
        with self.lock:
            if self._scheduler and self._scheduler.is_alive():
                return
            self._scheduler_stop.clear()
            self._scheduler = threading.Thread(target=self._check_loop, name="camera-connection-checker", daemon=True)
            self._scheduler.start()

    def shutdown(self):
        self._scheduler_stop.set()
        with self.lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            worker.stop_event.set()

    def _check_loop(self):
        while not self._scheduler_stop.wait(CONNECTION_CHECK_INTERVAL_SEC):
            with self.lock:
                monitors = list(self.monitors.values())
            for monitor in monitors:
                try:
                    monitor.check()
                except Exception:
                    logger.exception("카메라 끊김 검사 실패: %s", monitor.camera_id)

    def start_stream(self, camera_id: str, stream_url: str) -> dict:
        """OpenCV가 읽을 수 있는 RTSP/HTTP URL 수신을 시작한다. 같은 ID는 URL 변경 시 재시작한다."""
        self.monitor(camera_id)
        self.stop_stream(camera_id)
        stop_event = threading.Event()
        thread = threading.Thread(target=self._read_loop, args=(camera_id, stream_url, stop_event),
                                  name=f"camera-stream-{camera_id}", daemon=True)
        worker = StreamWorker(camera_id, stream_url, thread, stop_event)
        with self.lock:
            self.workers[camera_id] = worker
        thread.start()
        return self.stream_state(camera_id)

    def stop_stream(self, camera_id: str) -> bool:
        with self.lock:
            worker = self.workers.pop(camera_id, None)
        if not worker:
            return False
        worker.stop_event.set()
        return True

    def _read_loop(self, camera_id: str, stream_url: str, stop_event: threading.Event):
        import cv2
        monitor = self.monitor(camera_id)
        while not stop_event.is_set():
            capture = cv2.VideoCapture(stream_url)
            if not capture.isOpened():
                logger.warning("카메라 연결 실패, 재시도 예정: %s", camera_id)
                capture.release()
                stop_event.wait(STREAM_RECONNECT_DELAY_SEC)
                continue
            logger.info("카메라 수신 시작: %s", camera_id)
            try:
                while not stop_event.is_set():
                    ok, _frame = capture.read()
                    if not ok:
                        logger.warning("카메라 프레임 수신 중단: %s", camera_id)
                        break
                    # Path 2는 프레임 내용에 접근하지 않고 수신 시각만 전달한다.
                    monitor.frame_received()
            finally:
                capture.release()
            stop_event.wait(STREAM_RECONNECT_DELAY_SEC)

    def stream_state(self, camera_id: str) -> dict:
        with self.lock:
            worker = self.workers.get(camera_id)
        return {**self.monitor(camera_id).state(), "stream_registered": worker is not None,
                "stream_running": bool(worker and worker.thread.is_alive())}


camera_runtime = CameraRuntime()
