"""매뉴얼 색인과 검색 (FR-QNA-07, 09). 조각 카드 = ID, 원문, 문서명·버전·절, 임베딩."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .db import transaction
from .embedding import Embedder


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    doc_version: str
    section: str  # 예: "3절"
    text: str  # 원문 그대로
    embed_text: str  # 임베딩에만 쓰는 '제목 + 원문' (저장하지 않음)
    doc_status: str = ""


@dataclass
class Hit:
    chunk_id: str
    doc_name: str
    doc_version: str
    section: str
    text: str
    score: float
    doc_status: str = ""
    retrieval_method: str = "semantic"
    keyword_score: float = 0.0


def _query_terms(question: str) -> set[str]:
    """질문의 조사와 질문형 어미를 제거한 표면 핵심어. 의미 검색을 대체하지 않는다."""
    stop = {"몇", "무엇", "어떻게", "언제", "얼마", "무슨", "어떤", "하나요", "하나", "되나요",
            "있나요", "인가요", "알려줘", "주세요", "해줘", "해도", "되나", "해야", "되면"}
    terms = set()
    for token in re.findall(r"[가-힣]+|[A-Za-z]+", question):
        token = re.sub(r"(?:해야|하면|마다|인가요|인가|나요)$", "", token)
        if len(token) >= 3:
            token = re.sub(r"(?:에서|으로|에게|은|는|이|가|을|를|의)$", "", token)
        if len(token) >= 2 and token not in stop:
            terms.add(token.lower())
    return terms


def _split_long(body_lines: list[str], max_chars: int) -> list[str]:
    parts, cur = [], ""
    for line in body_lines:
        if cur and len(cur) + len(line) + 1 > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def parse_manual(markdown: str, max_chunk_chars: int = 700) -> list[Chunk]:
    meta: dict[str, str] = {}
    lines = markdown.splitlines()
    i = next((index for index, line in enumerate(lines) if line.startswith("## ")), len(lines))
    for line in lines[:i]:
        heading = re.match(r"^#\s+(.+)$", line)
        if heading:
            meta.setdefault("doc_name", heading.group(1).strip())
        field = re.match(r"^([^:]+):\s*(.+)$", line)
        if field:
            key = {"문서 ID": "doc_id", "버전": "doc_version", "상태": "doc_status"}.get(
                field.group(1).strip(), field.group(1).strip())
            meta[key] = field.group(2).strip()
    for key in ("doc_id", "doc_name", "doc_version"):
        if key not in meta:
            raise ValueError(f"문서 머리말에 {key} 가 없습니다")

    chunks: list[Chunk] = []
    cur_no, cur_title, cur_body = None, "", []

    def flush():
        if cur_no is None:
            return
        body = [b.strip() for b in cur_body if b.strip()]
        for k, text in enumerate(_split_long(body, max_chunk_chars), start=1):
            cid = f"{meta['doc_id']}-C{int(cur_no):02d}" + ("" if k == 1 else f"-{k}")
            chunks.append(
                Chunk(cid, meta["doc_name"], meta["doc_version"], f"{cur_no}절", text,
                      f"{cur_title}\n{text}", meta.get("doc_status", ""))
            )

    for line in lines[i:]:
        m = re.match(r"^##\s*(\d+)절\.?\s*(.*)$", line)
        if m:
            flush()
            cur_no, cur_title, cur_body = m.group(1), m.group(2).strip(), []
        elif cur_no is not None:
            cur_body.append(line)
    flush()
    if not chunks:
        raise ValueError("'## N절. 제목' 형식의 절을 찾지 못했습니다")
    return chunks


def ingest_manual(conn: sqlite3.Connection, embedder: Embedder, path: str | Path, max_chunk_chars: int = 700) -> int:
    """문서를 조각 카드로 저장한다. 같은 문서의 이전 버전 조각은 원문·조각·임베딩을 함께 지우고 다시 넣는다."""
    chunks = parse_manual(Path(path).read_text(encoding="utf-8"), max_chunk_chars)
    vecs = embedder.embed([c.embed_text for c in chunks]).astype(np.float32)
    doc_id = chunks[0].chunk_id.split("-")[0]
    with transaction(conn):
        conn.execute("DELETE FROM manual_chunks WHERE chunk_id LIKE ?", (f"{doc_id}-%",))
        for c, v in zip(chunks, vecs):
            conn.execute(
                "INSERT INTO manual_chunks (chunk_id, doc_name, doc_version, section, text, embedding, doc_status)"
                " VALUES (?,?,?,?,?,?,?)",
                (c.chunk_id, c.doc_name, c.doc_version, c.section, c.text, v.tobytes(), c.doc_status),
            )
    return len(chunks)


def search_chunks(
    conn: sqlite3.Connection,
    embedder: Embedder,
    question: str,
    top_k: int = 3,
    threshold: float = 0.45,
    doc_scope: str | None = None,
    keyword_threshold: float = 0.75,
) -> list[Hit]:
    """의미 유사도 또는 강한 핵심어 일치를 통과한 후보를 반환한다. 답변 가능 여부는 원문으로 검사한다."""
    sql, params = "SELECT * FROM manual_chunks", ()
    if doc_scope:
        sql, params = sql + " WHERE chunk_id LIKE ?", (f"{doc_scope}-%",)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []
    mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    q = embedder.embed([question])[0]
    scores = mat @ q
    terms = _query_terms(question)
    matches = [sum(term in r["text"].lower() for term in terms) for r in rows]
    lexical = [count / len(terms) if terms else 0.0 for count in matches]
    keyword_ok = [count >= 2 and score >= keyword_threshold for count, score in zip(matches, lexical)]
    eligible = [i for i in range(len(rows)) if float(scores[i]) >= threshold or keyword_ok[i]]
    # 코사인 점수는 그대로 반환하며, 강한 핵심어 일치만 순위에 가산한다.
    order = sorted(eligible, key=lambda i: float(scores[i]) + (0.2 * lexical[i] if keyword_ok[i] else 0),
                   reverse=True)[:top_k]
    hits = []
    for idx in order:
        r = rows[int(idx)]
        semantic_ok = float(scores[idx]) >= threshold
        method = "both" if semantic_ok and keyword_ok[idx] else "semantic" if semantic_ok else "keyword"
        hits.append(
            Hit(r["chunk_id"], r["doc_name"], r["doc_version"], r["section"], r["text"],
                round(float(scores[idx]), 4), r["doc_status"], method, round(lexical[idx], 4))
        )
    return hits
