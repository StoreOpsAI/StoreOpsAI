from __future__ import annotations

from .agent import Agent
from .config import Config, load_config
from .db import connect, init_schema
from .embedding import get_embedder
from .llm import LLMClient


def build_agent(cfg: Config | None = None) -> Agent:
    cfg = cfg or load_config()
    conn = connect(cfg.resolve(cfg.db.path))
    init_schema(conn)
    embedder = get_embedder(cfg.embedding.backend, cfg.embedding.model_name)
    return Agent(cfg, conn, LLMClient(cfg.llm), embedder)
