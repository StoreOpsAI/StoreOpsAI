from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8003/v1"
    model: str = "qwen3.6-35b-a3b"
    api_key: str = "sk-no-key-required"
    enable_thinking: bool = False
    temperature: float = 0.3
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 0.0
    max_tokens: int = 1024
    timeout_sec: int = 180


@dataclass
class AgentConfig:
    max_tool_calls: int = 5
    max_consecutive_failures: int = 2
    timezone_offset_hours: int = 9
    demo_now: str | None = None


@dataclass
class ManualConfig:
    doc_path: str = "data/manual_M001_v0.1.md"
    max_chunk_chars: int = 700
    top_k: int = 3
    similarity_threshold: float = 0.45


@dataclass
class EmbeddingConfig:
    backend: str = "bge-m3"
    model_name: str = "BAAI/bge-m3"


@dataclass
class DBConfig:
    path: str = "storeops.db"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    manual: ManualConfig = field(default_factory=ManualConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    db: DBConfig = field(default_factory=DBConfig)
    owners: dict[str, str] = field(default_factory=lambda: {"owner-01": "S01"})

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path


def _fill(dc_cls, data: dict[str, Any] | None):
    data = data or {}
    names = {f.name for f in fields(dc_cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{dc_cls.__name__}: 알 수 없는 설정 키 {sorted(unknown)}")
    return dc_cls(**data)


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    cfg = Config(
        llm=_fill(LLMConfig, raw.get("llm")),
        agent=_fill(AgentConfig, raw.get("agent")),
        manual=_fill(ManualConfig, raw.get("manual")),
        embedding=_fill(EmbeddingConfig, raw.get("embedding")),
        db=_fill(DBConfig, raw.get("db")),
    )
    if llm_url := os.getenv("STOREOPS_QNA_LLM_URL"):
        cfg.llm.base_url = llm_url
    if db_path := os.getenv("STOREOPS_QNA_DB_PATH"):
        cfg.db.path = db_path
    if raw.get("owners"):
        cfg.owners = {str(k): str(v) for k, v in raw["owners"].items()}
    return cfg
