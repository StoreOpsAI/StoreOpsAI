"""질문 Agent가 지난 사건의 대표 이미지를 VLM으로 해석해 달라고 요청하는 내부 API입니다.

질문 Agent는 컴포즈 밖 PC에서 돌고, 사건 이미지와 VLM 서버는 컴포즈 안에 있으므로 이 백엔드가 중계합니다.
인증은 백엔드가 질문 서비스를 부를 때 쓰는 것과 같은 공유 비밀(STOREOPS_QNA_TOKEN)을 X-QNA-Token 헤더로 받습니다.
요청한 매장의 사건만 해석하며, 다른 매장 사건은 "찾을 수 없음"으로 답해 존재 여부를 알려 주지 않습니다.
"""

import os
import secrets
from datetime import timedelta, timezone

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.routers.events import _resolve_output_uri
from app.services.event_service import EventNotFoundError, EventService
from app.services.vlm_describe import VlmUnavailableError, describe_images

KST = timezone(timedelta(hours=9))
MAX_EVENTS_PER_REQUEST = 5
EVENT_TYPE_KO = {
    'fall': '쓰러짐', 'fight': '싸움', 'fire': '방화', 'theft': '절도', 'vandalism': '파손', 'littering': '쓰레기 투기',
    'camera_disconnect': '카메라 연결 끊김', 'camera_reconnected': '카메라 재연결',
}


class DescribeRequest(BaseModel):
    """해석을 요청할 매장과 사건 번호 목록입니다."""

    store_id: str = Field(pattern=r'^S\d{2,}$')
    event_ids: list[str] = Field(min_length=1, max_length=MAX_EVENTS_PER_REQUEST)


def create_internal_qna_router(event_service: EventService) -> APIRouter:
    """질문 Agent 전용 내부 라우터를 만듭니다."""

    router = APIRouter(prefix='/api/internal/qna', tags=['internal'])

    @router.post('/describe')
    def describe_events(payload: DescribeRequest, x_qna_token: str | None = Header(default=None)) -> dict:
        expected = os.getenv('STOREOPS_QNA_TOKEN', '')
        if not expected or not x_qna_token or not secrets.compare_digest(expected, x_qna_token):
            raise HTTPException(status_code=403, detail='질문 서비스 인증에 실패했습니다.')
        return {'descriptions': [_describe_one(event_service, payload.store_id, event_id) for event_id in payload.event_ids]}

    return router


def _describe_one(event_service: EventService, store_id: str, event_id: str) -> dict:
    """사건 하나의 기본 정보와 VLM 해석(가능하면)을 돌려줍니다. 실패는 사유와 함께 상태로 알립니다."""

    try:
        event = event_service.get_event(event_id)
    except EventNotFoundError:
        return {'event_id': event_id, 'status': 'not_found'}
    if event.store_id != store_id:
        return {'event_id': event_id, 'status': 'not_found'}
    info = {
        'event_id': event.event_id,
        'event_type': event.event_type,
        'category_ko': EVENT_TYPE_KO.get(event.event_type, event.event_type),
        'camera_id': event.camera_id,
        'occurred_at': event.occurred_at.astimezone(KST).isoformat(timespec='seconds'),
    }
    if event.source != 'behavior_model':
        return {**info, 'status': 'no_images'}
    paths = []
    for media in event.representative_images:
        try:
            path = _resolve_output_uri(media.uri, 'representative_images')
        except ValueError:
            continue
        if path.is_file():
            paths.append(path)
    if not paths:
        return {**info, 'status': 'no_images'}
    try:
        result = describe_images(paths, info['category_ko'])
    except VlmUnavailableError as error:
        return {**info, 'status': 'vlm_unavailable', 'error': str(error)}
    return {**info, 'status': 'ok', 'description': result['text'], 'model_name': result['model_name'], 'image_count': len(paths)}
