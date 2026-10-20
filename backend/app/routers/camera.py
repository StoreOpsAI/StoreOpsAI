"""인증된 점주 웹캠 프레임을 내부 탐지 서비스로 전달합니다."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field


class CameraFrameBatch(BaseModel):
    """10초 동안 수집한 웹캠 이미지와 촬영 시각입니다."""

    camera_id: str = Field(default="CAM-01", min_length=1, max_length=32)
    captured_at: float = Field(gt=0)
    frames: list[str] = Field(min_length=12, max_length=48)


def _ingest_token(store_id: str) -> str:
    """현재 점주의 매장에 연결된 탐지 서비스 토큰을 선택합니다."""

    try:
        mapping = json.loads(os.getenv("STOREOPS_AI_INGEST_TOKENS_JSON", "{}"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=503, detail="탐지 서비스 인증 설정을 확인할 수 없습니다.") from error
    if not isinstance(mapping, dict):
        raise HTTPException(status_code=503, detail="탐지 서비스 인증 설정을 확인할 수 없습니다.")
    token = next(
        (key for key, mapped_store in mapping.items() if mapped_store == store_id),
        None,
    )
    if not token:
        raise HTTPException(status_code=503, detail="매장 탐지 서비스 인증이 설정되지 않았습니다.")
    return token


def _forward_frame_batch(payload: CameraFrameBatch, token: str) -> dict:
    """웹캠 프레임을 사설 탐지 서비스에 전달하고 결과 오류를 API 응답으로 바꿉니다."""

    base_url = os.getenv("STOREOPS_AI_URL", "http://localhost:8100").rstrip("/")
    request = Request(
        f"{base_url}/path1/analyze-frames",
        data=payload.model_dump_json().encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("detail", "탐지 요청이 실패했습니다.")
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = "탐지 요청이 실패했습니다."
        raise HTTPException(status_code=error.code, detail=detail) from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="탐지 서비스에 연결할 수 없습니다.") from error


def create_camera_router(current_user) -> APIRouter:
    """점주 세션을 확인한 뒤 해당 매장의 탐지 토큰으로 요청을 중계합니다."""

    router = APIRouter(prefix="/api/camera", tags=["camera"])

    @router.post("/analyze")
    def analyze_camera_batch(
        payload: CameraFrameBatch,
        user=Depends(current_user),
    ):
        return _forward_frame_batch(payload, _ingest_token(user.store_id))

    return router