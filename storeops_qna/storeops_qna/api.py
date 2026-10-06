"""서버 API (명세서 7.1절 제안): POST /api/ask, GET /api/ask/{session_id}/log

로그인 대신 시연용으로 X-Owner-Id 헤더를 씁니다 (config.yaml 의 owners 에 등록된 점주만).
실행: python -m storeops_qna.cli serve
"""
from __future__ import annotations

import os
import secrets
import threading

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .bootstrap import build_agent
from .config import Config


class AskRequest(BaseModel):
    question: str


class InternalAskRequest(AskRequest):
    owner_id: str = Field(min_length=1)
    store_id: str = Field(pattern=r"^S\d{2,}$")
    events: list[dict]
    order_drafts: list[dict]
    errors: dict[str, str] = Field(default_factory=dict)


def create_app(cfg: Config | None = None, agent=None) -> FastAPI:
    app = FastAPI(title="StoreOps AI · 질문")
    app.state.agent = agent or build_agent(cfg)
    lock = threading.Lock()  # SQLite 연결 하나를 같이 쓰므로 요청을 한 줄로 세운다 (시연 규모)

    def require_internal_token(token: str | None) -> None:
        expected = os.getenv("STOREOPS_QNA_TOKEN", "")
        if not expected or not token or not secrets.compare_digest(expected, token):
            raise HTTPException(status_code=403, detail="질문 서비스 인증에 실패했습니다.")

    @app.post("/internal/ask")
    def internal_ask(req: InternalAskRequest, x_qna_token: str | None = Header(default=None)):
        require_internal_token(x_qna_token)
        with lock:
            records = {"events": req.events, "order_drafts": req.order_drafts, "errors": req.errors}
            return app.state.agent.ask(req.question, req.owner_id, store_id=req.store_id, records=records).to_dict()

    @app.get("/internal/ask/{session_id}/log")
    def internal_log(session_id: int, owner_id: str, store_id: str,
                     x_qna_token: str | None = Header(default=None)):
        require_internal_token(x_qna_token)
        try:
            with lock:
                return app.state.agent.get_log(session_id, owner_id, store_id=store_id)
        except PermissionError as error:
            raise HTTPException(status_code=404, detail="질문 기록을 찾을 수 없습니다.") from error

    @app.post("/api/ask")
    def ask(req: AskRequest, x_owner_id: str = Header(...)):
        if os.getenv("STOREOPS_QNA_TOKEN"):
            raise HTTPException(status_code=403, detail="로그인한 점주는 기존 백엔드로 질문해 주세요.")
        try:
            with lock:
                return app.state.agent.ask(req.question, x_owner_id).to_dict()
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

    @app.get("/api/ask/{session_id}/log")
    def ask_log(session_id: int, x_owner_id: str = Header(...)):
        if os.getenv("STOREOPS_QNA_TOKEN"):
            raise HTTPException(status_code=403, detail="로그인한 점주는 기존 백엔드로 질문해 주세요.")
        try:
            with lock:
                return app.state.agent.get_log(session_id, x_owner_id)
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

    return app
