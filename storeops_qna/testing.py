"""테스트와 모의 연결용 도우미: 가짜 LLM, 시연 환경 만들기."""
from __future__ import annotations

import json
from typing import Any

from .agent import Agent
from .config import ROOT, Config
from .db import connect, init_schema
from .embedding import HashEmbedder
from .llm import LLMError
from .manual_index import ingest_manual
from .seed import seed_demo


def tc(name: str, args: dict | str, call_id: str = "c1", content: str = "") -> dict:
    """도구 호출 assistant 메시지를 만든다."""
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {"content": content, "tool_calls": [{"id": call_id, "type": "function",
                                                "function": {"name": name, "arguments": arguments}}]}


def text(content: str) -> dict:
    return {"content": content}


class FakeLLM:
    """미리 정한 응답을 순서대로 돌려준다. Exception 이 들어 있으면 던진다."""

    def __init__(self, script: list[Any]):
        self.script, self.calls = list(script), []

    def chat(self, messages, tools=None, json_mode=False, thinking=None):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": bool(tools), "json_mode": json_mode})
        if not self.script:
            raise AssertionError("가짜 LLM 응답이 모자랍니다")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_cfg() -> Config:
    cfg = Config()
    cfg.agent.demo_now = "2026-03-15T09:00:00+09:00"
    cfg.manual.similarity_threshold = 0.2  # 해시 임베딩용 기준값
    cfg.manual.doc_path = "M001_StoreOps_AI_매장점검규정_v0.1.md"
    cfg.owners = {"owner-01": "S01", "owner-02": "S02"}
    return cfg


def make_env(script: list[Any] | None = None):
    cfg = make_cfg()
    conn = connect(":memory:")
    init_schema(conn)
    seed_demo(conn)
    embedder = HashEmbedder()
    ingest_manual(conn, embedder, cfg.resolve(cfg.manual.doc_path), cfg.manual.max_chunk_chars)
    llm = FakeLLM(script or [])
    return cfg, conn, embedder, llm, Agent(cfg, conn, llm, embedder)


__all__ = ["tc", "text", "FakeLLM", "make_cfg", "make_env", "LLMError"]
