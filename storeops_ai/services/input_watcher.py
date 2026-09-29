"""input 폴더에 새 영상이 들어오면 사람이 API를 호출하지 않아도 자동으로 분석한다."""
from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

from config.config import (
    AUTO_ANALYSIS_DEFAULT_CAMERA_ID,
    AUTO_ANALYSIS_POLL_INTERVAL_SEC,
    INPUT_DIR,
)
from pipelines.path1_behavior import Path1BehaviorPipeline

logger = logging.getLogger("storeops_ai")

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class InputVideoWatcher:
    """카메라별 파이프라인을 재사용하면서 input 폴더의 새 영상을 순서대로 분석한다."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipelines: dict[str, Path1BehaviorPipeline] = {}
        self._processed_dir = INPUT_DIR / "processed"
        self._failed_dir = INPUT_DIR / "failed"
        # 파일 복사가 끝나기 전에 분석하지 않도록, 두 번의 스캔에서 크기가 같을 때만 처리한다.
        self._pending_signatures: dict[str, tuple[int, float]] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._failed_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="input-video-watcher", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()

    def _pipeline_for(self, camera_id: str) -> Path1BehaviorPipeline:
        if camera_id not in self._pipelines:
            self._pipelines[camera_id] = Path1BehaviorPipeline(camera_id=camera_id)
        return self._pipelines[camera_id]

    @staticmethod
    def _camera_id_for(path: Path) -> str:
        """파일명이 카메라 ID로 시작하면 그 값을 쓰고, 아니면 기본 카메라로 처리한다."""

        prefix = path.stem.split("_")[0]
        return prefix if prefix.upper().startswith("CAM") else AUTO_ANALYSIS_DEFAULT_CAMERA_ID

    def _loop(self) -> None:
        logger.info("입력 영상 자동 분석 시작: %s (주기 %.1fs)", INPUT_DIR, AUTO_ANALYSIS_POLL_INTERVAL_SEC)
        while not self._stop_event.wait(AUTO_ANALYSIS_POLL_INTERVAL_SEC):
            self._scan_once()

    def _scan_once(self) -> None:
        try:
            candidates = sorted(
                p for p in INPUT_DIR.iterdir()
                if p.is_file()
                and p.suffix.lower() in _VIDEO_EXTENSIONS
                # analyze()가 같은 폴더에 만드는 추적 영상은 입력으로 다시 잡지 않는다.
                and not p.stem.endswith("_tracked")
            )
        except FileNotFoundError:
            return

        seen_names = {p.name for p in candidates}
        for stale_name in set(self._pending_signatures) - seen_names:
            del self._pending_signatures[stale_name]

        for path in candidates:
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime)
            if self._pending_signatures.get(path.name) == signature:
                del self._pending_signatures[path.name]
                self._process(path)
            else:
                self._pending_signatures[path.name] = signature

    def _process(self, path: Path) -> None:
        camera_id = self._camera_id_for(path)
        logger.info("[자동 분석 시작] %s (camera=%s)", path.name, camera_id)
        try:
            events = self._pipeline_for(camera_id).analyze(str(path))
            logger.info("[자동 분석 완료] %s 사건 %d건", path.name, len(events))
            shutil.move(str(path), str(self._processed_dir / path.name))
        except Exception:
            logger.exception("[자동 분석 실패] %s", path.name)
            shutil.move(str(path), str(self._failed_dir / path.name))
        finally:
            # analyze()가 같은 폴더에 남긴 추적 영상도 input 폴더에 쌓이지 않도록 같이 옮긴다.
            tracked_path = path.with_name(f"{path.stem}_tracked{path.suffix}")
            if tracked_path.is_file():
                shutil.move(str(tracked_path), str(self._processed_dir / tracked_path.name))


input_video_watcher = InputVideoWatcher()
