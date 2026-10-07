"""탐지 서비스와 StoreOpsAI 백엔드 사이의 내부 사건 수신 라우터입니다."""

import os
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, status

from app.schemas.event import Event, EventClip, EventIngestRequest, EventMediaReference, EventVlm, EventVlmUpdateRequest
from app.services.event_service import EventNotFoundError, EventService


# NFR-09: 모든 사건 시각은 한국 시간(+09:00) 기준으로 저장한다.
KOREA_TIMEZONE = timezone(timedelta(hours=9))


CATEGORY_EVENT_TYPES = {
    "전도": "fall",
    "파손": "vandalism",
    "방화": "fire",
    "유기": "littering",
    "절도": "theft",
    "폭행": "fight",
    "fall": "fall",
    "broken": "vandalism",
    "fire": "fire",
    "abandon": "littering",
    "theft": "theft",
    "fight": "fight",
    "쓰러짐": "fall",
    "싸움": "fight",
    "파손": "vandalism",
    "쓰레기 투기": "littering",
    "카메라끊김": "camera_disconnect",
    "camera_disconnect": "camera_disconnect",
    "카메라재연결": "camera_reconnected",
}


def create_internal_events_router(event_service: EventService) -> APIRouter:
    """탐지기 웹훅을 내부 인증 토큰으로 보호합니다."""

    router = APIRouter(prefix="/api/internal", tags=["internal"])

    @router.post("/events", response_model=Event, status_code=status.HTTP_201_CREATED)
    def ingest_event(
        payload: EventIngestRequest,
        authorization: str | None = Header(default=None),
    ) -> Event:
        received_token = authorization.removeprefix("Bearer ").strip() if authorization else None
        token_store_map = _load_token_store_map()
        store_id = token_store_map.get(received_token or "")
        if not store_id:
            raise HTTPException(status_code=401, detail="탐지 서비스 인증이 필요합니다.")

        event_type = CATEGORY_EVENT_TYPES.get(
            payload.category or "",
            CATEGORY_EVENT_TYPES.get(payload.event_type, payload.event_type),
        )
        is_connection_event = event_type in {"camera_disconnect", "camera_reconnected"}
        occurred_at = _to_korea_time(payload.occurred_at or payload.created_at)
        source = "time_rule" if is_connection_event else "behavior_model"
        clip_path = payload.event_video_path or payload.clip_path
        clip_length = 0
        if payload.clip_start_time_sec is not None and payload.clip_end_time_sec is not None:
            clip_length = max(0, round(payload.clip_end_time_sec - payload.clip_start_time_sec))
        elif clip_path:
            clip_length = 10
        event_id = event_service.next_event_id()
        representative_images = _store_representative_images(
            payload.representative_images,
            payload.event_id,
        )

        event = Event(
            event_id=event_id,
            store_id=store_id,
            camera_id=payload.camera_id,
            source=source,
            event_type=event_type,
            occurred_at=occurred_at,
            clip=EventClip(uri=clip_path, length_sec=clip_length) if clip_path else None,
            representative_images=representative_images,
            scores=payload.scores,
            threshold=payload.threshold if not is_connection_event else None,
            gap_sec=round(payload.disconnect_seconds) if is_connection_event and payload.disconnect_seconds is not None else None,
            threshold_sec=round(payload.disconnect_threshold_seconds) if is_connection_event and payload.disconnect_threshold_seconds is not None else None,
            vlm=EventVlm(status="not_applicable" if is_connection_event else "pending"),
            status="unconfirmed",
            data_label=payload.data_label,
        )
        return event_service.ingest(event)

    @router.post("/events/{event_id}/vlm", response_model=Event)
    def update_vlm_result(
        event_id: str,
        payload: EventVlmUpdateRequest,
        authorization: str | None = Header(default=None),
    ) -> Event:
        """인증된 탐지 서비스의 VLM 분석 결과를 해당 매장 사건에 반영합니다."""

        received_token = authorization.removeprefix("Bearer ").strip() if authorization else None
        store_id = _load_token_store_map().get(received_token or "")
        if not store_id:
            raise HTTPException(status_code=401, detail="탐지 서비스 인증이 필요합니다.")

        try:
            event = event_service.get_event(event_id)
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.") from error
        if event.store_id != store_id:
            raise HTTPException(status_code=403, detail="해당 매장 사건에 접근할 수 없습니다.")
        if payload.status == "completed" and (
            not payload.observation
            or not payload.uncertain_points
            or len(payload.owner_actions) != 3
        ):
            raise HTTPException(status_code=422, detail="완료된 VLM 결과에는 관찰, 불확실성, 점주 확인 3개 항목이 필요합니다.")

        vlm = EventVlm(
            status=payload.status,
            summary=payload.observation,
            uncertain_points=payload.uncertain_points,
            owner_actions=payload.owner_actions,
            error=payload.error,
            model_name=payload.model_name,
        )
        try:
            return event_service.update_vlm(event_id, vlm)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return router


def _to_korea_time(value: datetime | None) -> datetime:
    """storeops_ai 컨테이너는 UTC로 동작하므로 시간대 정보가 없는 값은 UTC로 간주한 뒤 KST로 바꾼다."""

    if value is None:
        return datetime.now(KOREA_TIMEZONE)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(KOREA_TIMEZONE)


def _load_token_store_map() -> dict[str, str]:
    """환경변수의 토큰-매장 매핑만 읽고 클라이언트가 보낸 매장 ID는 신뢰하지 않습니다."""

    raw_mapping = os.getenv("STOREOPS_AI_INGEST_TOKENS_JSON", "")
    try:
        mapping = json.loads(raw_mapping)
    except json.JSONDecodeError:
        return {}
    if not isinstance(mapping, dict):
        return {}
    return {
        str(token): str(store_id)
        for token, store_id in mapping.items()
        if isinstance(token, str) and isinstance(store_id, str) and token and store_id
    }


def _store_representative_images(
    image_paths: list[str],
    source_event_id: str | None,
) -> list[EventMediaReference]:
    """DB 사건에 연결할 대표 이미지 프레임의 원본 URI를 검증합니다."""

    if not image_paths:
        return []
    output_root = Path(os.getenv('STOREOPS_OUTPUT_DIR', '/app/output')).resolve()
    image_root = output_root / 'representative_images'
    image_root.mkdir(parents=True, exist_ok=True)
    media_types = ('frame_start', 'frame_middle', 'frame_end')
    media_rows = []
    for index, (image_path, media_type) in enumerate(zip(image_paths[:3], media_types), start=1):
        normalized_path = image_path.replace('\\', '/')
        source_name = Path(normalized_path).name
        if not re.fullmatch(rf'{re.escape(source_event_id or "")}_{index}\.jpe?g', source_name, re.IGNORECASE):
            continue
        relative_image_path = normalized_path.removeprefix('output/')
        source_path = (output_root / relative_image_path).resolve()
        try:
            source_path.relative_to(image_root.resolve())
        except ValueError:
            continue
        if not source_path.is_file():
            continue
        media_rows.append(
            EventMediaReference(
                media_type=media_type,
                uri=source_path.relative_to(output_root).as_posix(),
            )
        )
    return media_rows
