"""VLM 확인 뒤 경보: 모델이 낸 사건 후보를 VLM이 '그 행동이 맞다'고 할 때만 백엔드(프론트)로 보낸다 (사용자 결정 2026-10-19).

- 맞음(confirmed=true): 1차 경보 → 대시보드(event.created) → VLM 결과 → (영상이 준비됐으면) 미디어 순서로 보낸다.
- 아님(confirmed=false): 백엔드·대시보드 어디에도 보내지 않고 사건 JSON에만 "VLM 기각"으로 남긴다(오경보 분석용).
- VLM 실패·미설정: VLM_GATE_FAIL_OPEN=1(기본)이면 보낸다. 실제 사건을 VLM 장애로 놓치지 않기 위해서다.
근거: Zanella 외, LAVAD, CVPR 2024(학습 없이 VLM·LLM으로 이상 판단), arXiv 2601.06204(값싼 1차 필터 뒤 애매한 사건만 추론 모델로).
판정 직전 클립(약 4초)의 처리 프레임 전부를 사진 여러 장으로 보는 것은 논문과 다르고, VLM이 실제 사건을 기각하는 비율은 아직 재지 않았다(근거 약함, 측정 필요).
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Executor, Future
from typing import Callable

from utils.notification import send_event_media, send_first_alert, send_vlm_result

logger = logging.getLogger("storeops_ai")


class VLMGate:
    def __init__(self, vlm, events, publish: Callable[[str, dict], None], media_executor: Executor, fail_open: bool = True):
        self.vlm, self.events, self.publish, self.media_executor, self.fail_open = vlm, events, publish, media_executor, fail_open
        self._lock = threading.Lock()
        self._decision: dict[str, bool] = {}     # 사건 ID -> 보냄 여부(결정 전이면 없음)
        self._media_ready: set[str] = set()      # 결정 전에 영상이 먼저 준비된 사건

    def submit(self, event, images: list[str], definition: str, clip_seconds: float) -> None:
        context = {"event_id": event.event_id, "camera_id": event.camera_id, "category": event.category, "definition": definition,
                   "clip_seconds": round(clip_seconds, 1), "frame_count": len(images)}
        try:
            future = self.vlm.verify_async(images, context)
        except Exception as exc:                 # 이미지 수가 틀리는 등 제출 자체가 실패
            self._decide(event, None, str(exc))
            return
        future.add_done_callback(lambda done, event=event: self._on_done(event, done))

    def mark_sent(self, event_id: str) -> None:
        """VLM 확인 없이 보낸 사건(물건 우선 경보)도 영상·진행 갱신이 나가게 보냄으로 기록한다."""
        with self._lock:
            self._decision[event_id] = True

    def is_sent(self, event_id: str) -> bool:
        """백엔드·대시보드로 보낸 사건인가(결정 전·기각이면 False). 영상 진행 상태 갱신을 이걸로 거른다."""
        with self._lock:
            return bool(self._decision.get(event_id))

    def media_ready(self, event) -> None:
        """사건 영상이 완성됐다. 보내기로 결정된 사건만 미디어를 올린다."""
        with self._lock:
            decided = self._decision.get(event.event_id)
            if decided is None:
                self._media_ready.add(event.event_id)
        if decided:
            self.media_executor.submit(send_event_media, event)

    def _on_done(self, event, future: Future) -> None:
        try:
            self._decide(event, future.result(), None)
        except Exception as exc:
            logger.exception("[VLM GATE FAILED] event=%s", event.event_id)
            self._decide(event, None, str(exc))

    def _decide(self, event, result: dict | None, error: str | None) -> None:
        status = (result or {}).get("status", "").upper()
        if result is not None and status != "VLM_NOT_CONFIGURED":
            send, vlm = bool(result["confirmed"]), {"status": "completed", "result": result}
        else:
            error = error or "QWEN_VLM_BASE_URL이 설정되지 않았습니다."
            send, vlm = self.fail_open, {"status": "failed", "error": error}
        event.status = "사건후보" if send else "VLM 기각"
        if send:
            event.alert_sent = bool(send_first_alert(event))
        self.events.save(event)
        updated = self.events.update(event.event_id, vlm=vlm)
        if send:
            send_vlm_result(event, vlm["status"], result=vlm.get("result"), error=vlm.get("error"))
            self.publish("event.created", updated)
        with self._lock:
            self._decision[event.event_id] = send
            media = event.event_id in self._media_ready
            self._media_ready.discard(event.event_id)
        if send and media:
            self.media_executor.submit(send_event_media, event)
        print(f"[VLM 확인] {event.event_id} {event.category}: {'맞음 → 전송' if send and vlm['status'] == 'completed' else '실패 → 전송(fail-open)' if send else '아님 → 보내지 않음'}")
