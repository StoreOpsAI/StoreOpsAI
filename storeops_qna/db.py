"""업무 기록 DB (명세서 5.4절 제안 구조). 개발용 SQLite. 6절 표의 'PostgreSQL' 로 옮기기 쉽게 단순한 SQL만 씁니다."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, camera_id TEXT,
  source TEXT NOT NULL CHECK (source IN ('behavior_model','time_rule')),
  event_type TEXT NOT NULL, occurred_at TEXT NOT NULL,
  clip_uri TEXT, clip_length_sec INTEGER, scores TEXT, threshold REAL,
  gap_sec INTEGER, threshold_sec INTEGER,
  vlm_status TEXT, vlm_result TEXT, status TEXT NOT NULL DEFAULT 'unconfirmed'
);
CREATE TABLE IF NOT EXISTS order_drafts (
  draft_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, product_id TEXT NOT NULL, target_date TEXT NOT NULL,
  forecast_d1 INTEGER, forecast_d2 INTEGER, forecast_total INTEGER, safety_stock INTEGER,
  on_hand INTEGER, incoming INTEGER, need INTEGER, pack_size INTEGER,
  recommended_qty INTEGER, recommended_boxes INTEGER, shortage_before_arrival INTEGER,
  data_label TEXT NOT NULL DEFAULT 'real', status TEXT NOT NULL DEFAULT 'draft_saved',
  sent_to_supplier INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS order_approvals (
  draft_id TEXT NOT NULL, approved_qty INTEGER, approved_boxes INTEGER, approved_by TEXT, approved_at TEXT
);
CREATE TABLE IF NOT EXISTS manual_chunks (
  chunk_id TEXT PRIMARY KEY, doc_name TEXT NOT NULL, doc_version TEXT NOT NULL,
  section TEXT NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL,
  doc_status TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS qa_sessions (
  session_id INTEGER PRIMARY KEY AUTOINCREMENT, store_id TEXT NOT NULL, owner_id TEXT,
  question TEXT NOT NULL, answer TEXT, stop_reason TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, step INTEGER NOT NULL,
  tool_name TEXT, input TEXT, result_status TEXT, output TEXT, reason TEXT,
  rejected_reason TEXT, called_at TEXT NOT NULL
);
-- 추가 제안: '자료를 아직 받지 못함'(not_received)을 판단하려면 어느 기간까지 받았는지 알아야 합니다.
CREATE TABLE IF NOT EXISTS data_coverage (
  store_id TEXT NOT NULL, record_type TEXT NOT NULL, covered_from TEXT NOT NULL, covered_to TEXT NOT NULL,
  PRIMARY KEY (store_id, record_type)
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    if not any(row[1] == "doc_status" for row in conn.execute("PRAGMA table_info(manual_chunks)")):
        conn.execute("ALTER TABLE manual_chunks ADD COLUMN doc_status TEXT NOT NULL DEFAULT ''")
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        conn.execute("BEGIN")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
