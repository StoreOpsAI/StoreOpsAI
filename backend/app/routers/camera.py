"""탐지 PC의 실시간 카메라 화면을 점주 세션으로 중계합니다."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException, Query, Response

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


def _fetch_live_frame(after: int, token: str) -> Response:
    """탐지 서비스의 최신 화면(JPEG)을 받아 그대로 돌려줍니다. 새 화면이 없으면 204입니다."""

    base_url = os.getenv("STOREOPS_AI_LIVE_URL", "http://localhost:8101").rstrip("/")
    request = Request(
        f"{base_url}/live/frame?after={after}",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=float(os.getenv("STOREOPS_AI_LIVE_TIMEOUT_SEC", "15"))) as response:
            if response.status == 204:
                return Response(status_code=204)
            return Response(
                content=response.read(),
                media_type="image/jpeg",
                headers={"X-Frame-Seq": response.headers.get("X-Frame-Seq", "0"), "Cache-Control": "no-store"},
            )
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="실시간 화면 인증 또는 요청이 실패했습니다.") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="실시간 화면 서비스에 연결할 수 없습니다.") from error


def create_camera_router(current_user) -> APIRouter:
    """점주 세션을 확인한 뒤 해당 매장의 탐지 토큰으로 요청을 중계합니다."""

    router = APIRouter(prefix="/api/camera", tags=["camera"])

    @router.get("/live-frame")
    def get_live_frame(
        after: int = Query(default=0, ge=0),
        user=Depends(current_user),
    ):
        return _fetch_live_frame(after, _ingest_token(user.store_id))

    return router