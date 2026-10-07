"""수요 예측·발주 서비스로 요청을 전달하는 인증 프록시 라우터입니다."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal

from app.schemas.auth import AuthUser


class DraftRequest(BaseModel):
    """예측값을 사용해 발주 초안을 만들기 위한 입력입니다."""

    product_id: str = Field(min_length=1, max_length=100)
    target_date: str
    d1: float = Field(ge=0)
    d2: float = Field(ge=0)
    on_hand: int = Field(ge=0)
    incoming: int = Field(default=0, ge=0)
    incoming_day: int = Field(default=1, ge=1, le=2)
    arrival_after_days: int = Field(default=2, ge=0, le=2)


class DailySaleRequest(BaseModel):
    """상품별 일 판매량과 기록 상태입니다."""

    date: str
    sales: float | None = Field(default=None, ge=0)
    status: Literal["observed", "closed", "stockout", "missing"] = "observed"


class CalendarDayRequest(BaseModel):
    """예측 대상일의 휴일·행사 정보입니다."""

    date: str
    is_holiday: int = Field(ge=0, le=1)
    has_event: int = Field(default=0, ge=0, le=1)


class DemandForecastRequest(BaseModel):
    """XGBoost 수요 예측에 필요한 상품 이력과 달력입니다."""

    product_id: str = Field(min_length=1, max_length=100)
    as_of: str
    history: list[DailySaleRequest] = Field(min_length=7, max_length=2000)
    calendar: list[CalendarDayRequest] = Field(min_length=2, max_length=2000)
    country: Literal["US", "KR"] = "US"


class ForecastDraftRequest(BaseModel):
    """XGBoost 이틀 예측과 점주 재고를 발주 추천으로 연결합니다."""

    demand: DemandForecastRequest
    on_hand: int = Field(ge=0)
    incoming: int = Field(default=0, ge=0)
    incoming_day: int = Field(default=1, ge=1, le=2)
    arrival_after_days: int = Field(default=2, ge=0, le=2)


class ApprovalRequest(BaseModel):
    """발주 초안 승인 수량입니다."""

    qty: int = Field(ge=0)


def _demand_token(store_id: str) -> str:
    """매장 ID에 대응하는 하위 서비스 토큰을 환경변수에서 선택합니다."""

    tokens = json.loads(os.getenv("STOREOPS_DEMAND_TOKENS_JSON", "{}"))
    token = tokens.get(store_id) or os.getenv("STOREOPS_DEMAND_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="수요 서비스 인증이 설정되지 않았습니다.")
    return token


def _forward(path: str, method: str, payload: dict | None, token: str, idempotency_key: str | None = None):
    """하위 수요 서비스의 응답과 오류를 현재 API 규칙으로 전달합니다."""

    base_url = os.getenv("STOREOPS_DEMAND_URL", "http://localhost:8765").rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    request = Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("detail", "수요 서비스 요청이 실패했습니다.")
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = "수요 서비스 요청이 실패했습니다."
        raise HTTPException(status_code=error.code, detail=detail) from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="수요 서비스에 연결할 수 없습니다.") from error


def create_demand_router(current_user) -> APIRouter:
    """StoreOpsAI 세션 인증을 수요 서비스 토큰 인증으로 변환합니다."""

    router = APIRouter(prefix="/api/demand", tags=["demand"])

    @router.get("/orders/drafts")
    def list_drafts(user: AuthUser = Depends(current_user)):
        return _forward("/api/orders/drafts", "GET", None, _demand_token(user.store_id))

    @router.post("/orders/drafts", status_code=status.HTTP_201_CREATED)
    def create_draft(
        payload: DraftRequest,
        idempotency_key: str = Header(min_length=1, max_length=120),
        user: AuthUser = Depends(current_user),
    ):
        return _forward(
            "/api/orders/drafts",
            "POST",
            payload.model_dump(),
            _demand_token(user.store_id),
            idempotency_key,
        )

    @router.post("/orders/forecast-draft", status_code=status.HTTP_201_CREATED)
    def create_forecast_draft(
        payload: ForecastDraftRequest,
        idempotency_key: str = Header(min_length=1, max_length=120),
        user: AuthUser = Depends(current_user),
    ):
        return _forward(
            "/api/orders/forecast-draft",
            "POST",
            payload.model_dump(),
            _demand_token(user.store_id),
            idempotency_key,
        )

    @router.post("/orders/drafts/{draft_id}/approve")
    def approve_draft(
        draft_id: str,
        payload: ApprovalRequest,
        user: AuthUser = Depends(current_user),
    ):
        return _forward(
            f"/api/orders/drafts/{draft_id}/approve",
            "POST",
            payload.model_dump(),
            _demand_token(user.store_id),
        )

    return router