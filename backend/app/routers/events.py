"""사건 조회와 점주 상태 변경을 위한 HTTP 라우터입니다."""

import os
from pathlib import Path, PurePosixPath
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.schemas.auth import AuthUser
from app.schemas.event import Event, EventListResponse, EventStatus, EventStatusChangeRequest, EventStatusHistory
from app.services.event_service import EventNotFoundError, EventService, InvalidEventStatusTransitionError


def create_events_router(event_service: EventService, current_user) -> APIRouter:
    """세션 점주의 매장으로 범위가 제한된 사건 라우터를 생성합니다."""

    router = APIRouter(prefix='/api/events', tags=['events'])

    def get_owned_event(event_id: str, user: AuthUser) -> Event:
        event = event_service.get_event(event_id)
        if event.store_id != user.store_id:
            raise EventNotFoundError(event_id)
        return event

    @router.get('', response_model=EventListResponse)
    def list_events(
        store_id: str | None = Query(default=None),
        status_filter: EventStatus | None = Query(default=None, alias='status'),
        user: AuthUser = Depends(current_user),
    ) -> EventListResponse:
        events = event_service.list_events(store_id=user.store_id, status=status_filter)
        return EventListResponse(events=events, count=len(events))

    @router.get('/{event_id}', response_model=Event)
    def get_event(event_id: str, user: AuthUser = Depends(current_user)) -> Event:
        try:
            return get_owned_event(event_id, user)
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail='사건을 찾을 수 없습니다.') from error

    @router.post('/{event_id}/status', response_model=Event)
    def change_event_status(
        event_id: str,
        payload: EventStatusChangeRequest,
        user: AuthUser = Depends(current_user),
    ) -> Event:
        try:
            get_owned_event(event_id, user)
            return event_service.change_status(event_id=event_id, status=payload.status, changed_by=user.user_id)
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail='사건을 찾을 수 없습니다.') from error
        except InvalidEventStatusTransitionError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    @router.get('/{event_id}/history', response_model=list[EventStatusHistory])
    def get_event_history(event_id: str, user: AuthUser = Depends(current_user)) -> list[EventStatusHistory]:
        try:
            get_owned_event(event_id, user)
            return event_service.history(event_id)
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail='사건을 찾을 수 없습니다.') from error

    @router.get('/{event_id}/clip')
    def get_event_clip(event_id: str, user: AuthUser = Depends(current_user)) -> FileResponse:
        """공유 출력 폴더의 사건 영상을 점주 세션으로만 제공합니다."""

        try:
            event = get_owned_event(event_id, user)
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail='사건을 찾을 수 없습니다.') from error
        if event.clip is None:
            raise HTTPException(status_code=404, detail='사건 영상이 없습니다.')

        try:
            safe_path = _resolve_output_uri(event.clip.uri)
        except ValueError as error:
            raise HTTPException(status_code=404, detail='사건 영상 경로가 유효하지 않습니다.') from error
        if not safe_path.is_file():
            raise HTTPException(status_code=404, detail='사건 영상 파일을 찾을 수 없습니다.')
        return FileResponse(safe_path, media_type='video/mp4', filename=safe_path.name)

    @router.get('/{event_id}/representative-images/{media_type}')
    def get_representative_image(
        event_id: str,
        media_type: Literal['frame_start', 'frame_middle', 'frame_end'],
        user: AuthUser = Depends(current_user),
    ) -> FileResponse:
        try:
            event = get_owned_event(event_id, user)
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail='사건을 찾을 수 없습니다.') from error
        media = next(
            (item for item in event.representative_images if item.media_type == media_type),
            None,
        )
        if media is None:
            raise HTTPException(status_code=404, detail='대표 이미지를 찾을 수 없습니다.')
        try:
            image_path = _resolve_output_uri(media.uri, 'representative_images')
        except ValueError as error:
            raise HTTPException(status_code=404, detail='대표 이미지 경로가 유효하지 않습니다.') from error
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail='대표 이미지 파일을 찾을 수 없습니다.')
        return FileResponse(image_path, media_type='image/jpeg', filename=image_path.name)

    return router


def _resolve_output_uri(uri: str, required_directory: str | None = None) -> Path:
    output_root = Path(os.getenv('STOREOPS_OUTPUT_DIR', '/app/output')).resolve()
    normalized_uri = uri.replace('\\', '/')
    uri_path = PurePosixPath(normalized_uri)
    if uri_path.is_absolute():
        candidate = Path(str(uri_path))
    else:
        relative_parts = uri_path.parts
        if relative_parts and relative_parts[0] == 'output':
            relative_parts = relative_parts[1:]
        candidate = output_root.joinpath(*relative_parts)

    resolved_path = candidate.resolve()
    try:
        relative_path = resolved_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError('URI is outside output directory') from error
    if required_directory and (not relative_path.parts or relative_path.parts[0] != required_directory):
        raise ValueError('URI is outside required directory')
    return resolved_path