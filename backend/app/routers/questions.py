"""인증된 매장의 운영 기록을 질문 서비스의 조회 전용 도구에 전달합니다."""

import json
import os
from datetime import timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.routers.demand import _demand_token, _forward
from app.schemas.auth import AuthUser
from app.services.event_service import EventService


KST = timezone(timedelta(hours=9))


class QuestionRequest(BaseModel):
    """한 번의 점주 질문으로 실행할 조회 요청입니다."""

    question: str = Field(min_length=1, max_length=500)


def _call_qna(path: str, payload: dict | None = None) -> dict:
    """브라우저 세션 대신 서버 간 비밀 토큰으로 내부 질문 서비스에 접근합니다."""

    token = os.getenv('STOREOPS_QNA_TOKEN')
    if not token:
        raise HTTPException(status_code=503, detail='질문 서비스 인증이 설정되지 않았습니다.')
    base = os.getenv('STOREOPS_QNA_URL', 'http://127.0.0.1:8002').rstrip('/')
    request = Request(
        base + path,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'X-QNA-Token': token, 'Content-Type': 'application/json; charset=utf-8'},
        method='POST' if payload is not None else 'GET',
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        if error.code == 404:
            raise HTTPException(status_code=404, detail='질문 기록을 찾을 수 없습니다.') from error
        raise HTTPException(status_code=502, detail='질문 서비스 응답을 확인하지 못했습니다.') from error
    except (URLError, TimeoutError, ValueError) as error:
        raise HTTPException(status_code=502, detail='질문 서비스에 연결할 수 없습니다.') from error


def create_questions_router(event_service: EventService, current_user) -> APIRouter:
    """현재 로그인 점주의 매장에 한정된 질문 및 실행 로그 API를 만듭니다."""

    router = APIRouter(prefix='/api/ask', tags=['questions'])

    @router.post('')
    def ask(payload: QuestionRequest, user: AuthUser = Depends(current_user)) -> dict:
        errors = {}
        try:
            events = [
                {
                    'event_id': event.event_id,
                    'store_id': event.store_id,
                    'camera_id': event.camera_id,
                    'source': event.source,
                    'event_type': event.event_type,
                    'occurred_at': event.occurred_at.astimezone(KST).isoformat(timespec='seconds'),
                    'clip_uri': event.clip.uri if event.clip else None,
                    'clip_length_sec': event.clip.length_sec if event.clip else None,
                }
                for event in event_service.list_events(store_id=user.store_id)
                if event.store_id == user.store_id
            ]
        except Exception:
            events = []
            errors['event'] = '사건 기록 조회에 실패했습니다.'

        try:
            drafts = [
                {
                    'draft_id': draft['draft_id'], 'store_id': user.store_id,
                    'product_id': draft['product_id'], 'target_date': draft['target_date'],
                    'status': 'approved' if draft.get('approved') else 'draft_saved',
                }
                for draft in _forward('/api/orders/drafts', 'GET', None, _demand_token(user.store_id))
                if draft.get('store_id') == user.store_id
            ]
        except Exception:
            drafts = []
            errors['order_draft'] = '발주 기록 조회에 실패했습니다.'

        return _call_qna('/internal/ask', {
            'question': payload.question,
            'owner_id': user.user_id,
            'store_id': user.store_id,
            'events': events,
            'order_drafts': drafts,
            'errors': errors,
        })

    @router.get('/{question_session_id}/log')
    def get_log(question_session_id: int, user: AuthUser = Depends(current_user)) -> dict:
        query = urlencode({'owner_id': user.user_id, 'store_id': user.store_id})
        return _call_qna(f'/internal/ask/{question_session_id}/log?{query}')

    return router