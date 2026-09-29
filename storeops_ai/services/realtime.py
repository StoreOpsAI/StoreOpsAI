"""React 대시보드에 사건 변경을 전달하는 WebSocket 이벤트 허브."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger("storeops_ai")


class EventHub:
    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        """FastAPI 시작 시 실행 루프를 기록해 동기 파이프라인에서도 publish할 수 있게 한다."""
        self._loop = asyncio.get_running_loop()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for websocket in tuple(self._connections):
            try:
                await websocket.send_json(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)

    def publish(self, message_type: str, event: dict[str, Any]) -> None:
        """동기·백그라운드 스레드에서 안전하게 React 연결 전체에 이벤트를 보낸다."""
        if self._loop is None or self._loop.is_closed():
            return
        message = {
            "type": message_type,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)
        except RuntimeError:
            logger.warning("WebSocket 이벤트 전송을 예약하지 못했습니다.")


event_hub = EventHub()
