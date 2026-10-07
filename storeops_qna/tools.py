"""Agent 도구 3개 (FR-QNA-02): query_records, search_manual, get_event_video. 모두 조회만 합니다.

- validate_input : 호출 전 입력 검사 (FR-QNA-10). 틀리면 고치지 않고 ToolInputError.
- execute_tool   : 검사 → 실행 → 결과 dict. 절대 예외를 밖으로 던지지 않고, 조회 오류를 0건으로 바꾸지 않는다 (NFR-04).
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from .config import Config
from .llm import LLMClient, LLMError
from .manual_index import Hit, search_chunks
from .timeutil import KST, parse_kst_iso
from datetime import datetime

ALLOWED_RECORD_TYPES = ("event", "order_draft")
ALLOWED_EVENT_TYPES = ("fall", "fight", "fire", "theft", "vandalism", "littering", "camera_disconnect", "camera_reconnected")
RECORD_LABEL = {"event": "사건", "order_draft": "발주 초안"}
EVENT_ID_RE = re.compile(r"^E\d{3}$")
STORE_ID_RE = re.compile(r"^S\d{2}$")
DOC_ID_RE = re.compile(r"^M\d{3}$")


class ToolInputError(Exception):
    code = "invalid_input"

    def __init__(self, field: str, value: Any, reason: str, suggestion: str | None = None):
        super().__init__(f"{field}: {reason}")
        self.field, self.value, self.reason, self.suggestion = field, value, reason, suggestion

    def owner_message(self) -> str:
        msg = f"{self.field} 값 '{self.value}'을(를) 사용할 수 없습니다: {self.reason}."
        if self.suggestion:
            particle = "이" if self.suggestion[-1] in "013678" else "가"
            msg += f" {self.suggestion}{particle} 맞나요?"
        else:
            msg += " 올바른 값을 알려 주세요."
        return msg


class PermissionDenied(ToolInputError):
    code = "permission_denied"


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    store_id: str  # 로그인한 점주의 매장. Agent(LLM)가 바꿀 수 없다 (FR-QNA-12)
    embedder: Any
    llm: LLMClient | None
    cfg: Config
    records: dict | None = None  # 인증된 백엔드가 제공한 매장별 운영 기록


# ---------------------------------------------------------------- 도구 명세 (LLM에 보이는 형식, 교안 28·31p)
TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_records",
            "description": (
                "저장된 업무 기록의 건수와 목록을 조회한다. 사건이 몇 건인지, 어떤 사건이 있었는지, 발주 초안이 있는지 같은 "
                "'건수·기록' 질문에 쓴다. 사건 번호 없는 영상 요청에서는 먼저 이 도구로 사건을 찾는다. "
                "규정·절차 질문에는 쓰지 않는다. 조회만 하며 기록을 바꾸지 않는다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "매장 번호. 시스템 프롬프트에 적힌 값만 쓴다. 예: S01"},
                    "record_type": {"type": "string", "enum": list(ALLOWED_RECORD_TYPES), "description": "event=사건, order_draft=발주 초안"},
                    "event_type": {"type": "string", "enum": list(ALLOWED_EVENT_TYPES), "description": "사건 종류(선택). 쓰러짐=fall. 발주 조회에는 사용하지 않는다"},
                    "date_from": {"type": "string", "description": "시작 시각(포함). 한국 시간 ISO 8601, 예: 2026-03-14T00:00:00+09:00"},
                    "date_to": {"type": "string", "description": "끝 시각(미포함). 한국 시간 ISO 8601, 예: 2026-03-15T00:00:00+09:00"},
                },
                "required": ["store_id", "record_type", "date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": (
                "매장 운영 규정 문서에서 근거를 찾아 답한다. '~할 때 어떻게 하나요', '무엇을 기록해야 하나요' 같은 규정·절차 질문에 쓴다. "
                "지금 재고나 어제 건수처럼 기록으로 확인할 질문에는 쓰지 않는다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "점주의 질문 문장"},
                    "doc_scope": {"type": "string", "description": "찾을 문서 번호(선택). 예: M001. 모르면 생략"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_video",
            "description": (
                "특정 사건 번호의 영상 위치와 사건 정보를 조회한다. 점주가 특정 사건의 영상을 보여 달라고 할 때 쓴다. "
                "사건 번호를 모르면 먼저 query_records로 찾는다. 점주가 쓴 사건 번호는 형식을 고치지 말고 그대로 넣는다."
            ),
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string", "description": "사건 번호. E와 숫자 세 자리, 예: E015"}},
                "required": ["event_id"],
            },
        },
    },
]
TOOL_NAMES = tuple(d["function"]["name"] for d in TOOL_DEFS)  # 허용 목록 (NFR-05)

_REQUIRED = {d["function"]["name"]: d["function"]["parameters"]["required"] for d in TOOL_DEFS}
_ALLOWED_KEYS = {d["function"]["name"]: set(d["function"]["parameters"]["properties"]) for d in TOOL_DEFS}


# ---------------------------------------------------------------- 입력 검사
def validate_input(name: str, args: Any, ctx: ToolContext) -> dict[str, Any]:
    if name not in TOOL_NAMES:
        raise ToolInputError("tool", name, "허용 목록에 없는 도구입니다")
    if not isinstance(args, dict):
        raise ToolInputError("arguments", args, "입력값이 JSON 객체가 아닙니다")
    for key in args:
        if key not in _ALLOWED_KEYS[name]:
            raise ToolInputError(key, args[key], "이 도구에 없는 입력 항목입니다")
    for key in _REQUIRED[name]:
        if args.get(key) in (None, ""):
            raise ToolInputError(key, args.get(key), "필수 입력값이 없습니다")

    out: dict[str, Any] = {}
    if name == "query_records":
        store = args["store_id"]
        if not isinstance(store, str) or not STORE_ID_RE.match(store):
            raise ToolInputError("store_id", store, "매장 번호는 S와 숫자 두 자리여야 합니다")
        if store != ctx.store_id:
            raise PermissionDenied("store_id", store, f"자기 매장({ctx.store_id}) 기록만 조회할 수 있습니다")
        rt = args["record_type"]
        if rt not in ALLOWED_RECORD_TYPES:
            raise ToolInputError("record_type", rt, f"허용된 기록 종류는 {', '.join(ALLOWED_RECORD_TYPES)} 입니다")
        event_type = args.get("event_type")
        if event_type is not None and (rt != "event" or event_type not in ALLOWED_EVENT_TYPES):
            raise ToolInputError("event_type", event_type, "사건 조회에 사용할 수 없는 종류입니다")
        try:
            d_from, d_to = parse_kst_iso(args["date_from"]), parse_kst_iso(args["date_to"])
        except ValueError as e:
            bad = "date_from" if not _is_ok_date(args["date_from"]) else "date_to"
            raise ToolInputError(bad, args[bad], str(e)) from e
        if d_from > d_to:
            raise ToolInputError("date_from", args["date_from"], "시작이 끝보다 늦습니다")
        out = {"store_id": store, "record_type": rt, "date_from": args["date_from"], "date_to": args["date_to"]}
        if event_type is not None:
            out["event_type"] = event_type
    elif name == "search_manual":
        q = args["question"]
        if not isinstance(q, str) or not q.strip():
            raise ToolInputError("question", q, "질문이 비어 있습니다")
        if len(q) > 500:
            raise ToolInputError("question", q[:20] + "…", "질문이 너무 깁니다(500자 이하)")
        scope = args.get("doc_scope")
        if scope is not None and (not isinstance(scope, str) or not DOC_ID_RE.match(scope)):
            raise ToolInputError("doc_scope", scope, "문서 번호는 M과 숫자 세 자리여야 합니다")
        out = {"question": q.strip(), "doc_scope": scope}
    elif name == "get_event_video":
        eid = args["event_id"]
        if not isinstance(eid, str) or not EVENT_ID_RE.match(eid):
            digits = re.findall(r"\d+", str(eid))
            suggestion = f"E{int(digits[0]):03d}" if digits and int(digits[0]) < 1000 else None
            raise ToolInputError("event_id", eid, "사건 번호는 E와 숫자 세 자리여야 합니다", suggestion)
        out = {"event_id": eid}
    return out


def _is_ok_date(v: Any) -> bool:
    try:
        parse_kst_iso(v)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- 실행
def execute_tool(name: str, raw_args: Any, ctx: ToolContext) -> dict[str, Any]:
    """검사 후 실행한다. 어떤 실패도 status='error' 결과로 돌려주며, 오류를 0건으로 바꾸지 않는다."""
    try:
        args = validate_input(name, raw_args, ctx)
    except ToolInputError as e:
        return _error(name, e.code, e.owner_message(), raw_args)
    try:
        if name == "query_records":
            return _query_records(args, ctx)
        if name == "search_manual":
            return _search_manual(args, ctx)
        return _get_event_video(args, ctx)
    except Exception as e:  # 저장소 오류 등
        return _error(name, "storage_error" if isinstance(e, sqlite3.Error) else "tool_error",
                      _FAIL_MESSAGE.get(name, "조회에 실패했습니다."), args, detail=f"{type(e).__name__}: {e}")


_FAIL_MESSAGE = {
    "query_records": "기록 조회에 실패했습니다. 기록이 없다는 뜻이 아니니 잠시 뒤 다시 시도해 주세요.",
    "search_manual": "규정 검색에 실패했습니다. 근거가 없다는 뜻이 아니니 잠시 뒤 다시 시도해 주세요.",
    "get_event_video": "사건 영상 조회에 실패했습니다.",
}


def _error(name: str, code: str, message: str, args: Any, detail: str | None = None) -> dict[str, Any]:
    res: dict[str, Any] = {"status": "error", "error_code": code, "message": message, "tool": name}
    if name == "query_records":
        res.update({"count": None, "record_ids": [], "conditions": args if isinstance(args, dict) else {}})
    if detail:
        res["detail"] = detail
    return res


def _query_records(args: dict, ctx: ToolContext) -> dict[str, Any]:
    cond = dict(args)
    label = RECORD_LABEL[args["record_type"]]
    d_from, d_to = parse_kst_iso(args["date_from"]), parse_kst_iso(args["date_to"])

    if ctx.records is not None:
        if args["record_type"] in ctx.records.get("errors", {}):
            return _error("query_records", "source_error", _FAIL_MESSAGE["query_records"], cond)
        source = ctx.records["events" if args["record_type"] == "event" else "order_drafts"]
        records = []
        for record in source:
            if record["store_id"] != ctx.store_id:
                continue
            if args["record_type"] == "event":
                if args.get("event_type") and record["event_type"] != args["event_type"]:
                    continue
                date = parse_kst_iso(record["occurred_at"])
                item = {"record_id": record["event_id"], "event_type": record["event_type"],
                        "source": record["source"], "camera_id": record["camera_id"],
                        "occurred_at": record["occurred_at"]}
            else:
                date = datetime.fromisoformat(record["target_date"]).replace(tzinfo=KST)
                item = {"record_id": record["draft_id"], "product_id": record["product_id"],
                        "target_date": record["target_date"], "status": record.get("status", "draft_saved")}
            if d_from <= date < d_to:
                records.append(item)
        ids = [record["record_id"] for record in records]
        msg = f"{label} {len(ids)}건: {', '.join(ids)}" if ids else f"해당 기간에 기록된 {label}이(가) 없습니다."
        return {"status": "ok", "count": len(ids), "record_ids": ids, "records": records,
                "conditions": cond, "message": msg}

    cov = ctx.conn.execute(
        "SELECT covered_from, covered_to FROM data_coverage WHERE store_id=? AND record_type=?",
        (args["store_id"], args["record_type"]),
    ).fetchone()
    if cov is None or d_from < parse_kst_iso(cov["covered_from"]) or d_to > parse_kst_iso(cov["covered_to"]):
        return {
            "status": "not_received", "count": None, "record_ids": [], "conditions": cond,
            "received_range": None if cov is None else [cov["covered_from"], cov["covered_to"]],
            "message": f"해당 기간의 {label} 기록을 아직 받지 못했습니다.",
        }

    records: list[dict] = []
    if args["record_type"] == "event":
        rows = ctx.conn.execute(
            "SELECT event_id, event_type, source, camera_id, occurred_at FROM events "
            "WHERE store_id=? AND (? IS NULL OR event_type=?) ORDER BY occurred_at",
            (args["store_id"], args.get("event_type"), args.get("event_type")),
        ).fetchall()
        for r in rows:
            if d_from <= parse_kst_iso(r["occurred_at"]) < d_to:
                records.append({"record_id": r["event_id"], "event_type": r["event_type"], "source": r["source"],
                                "camera_id": r["camera_id"], "occurred_at": r["occurred_at"]})
    else:
        rows = ctx.conn.execute(
            "SELECT draft_id, product_id, target_date, status FROM order_drafts WHERE store_id=? ORDER BY target_date",
            (args["store_id"],),
        ).fetchall()
        for r in rows:
            day = datetime.fromisoformat(r["target_date"]).replace(tzinfo=KST)
            if d_from <= day < d_to:
                records.append({"record_id": r["draft_id"], "product_id": r["product_id"],
                                "target_date": r["target_date"], "status": r["status"]})

    ids = [r["record_id"] for r in records]
    msg = f"{label} {len(ids)}건: {', '.join(ids)}" if ids else f"해당 기간에 기록된 {label}이(가) 없습니다."
    return {"status": "ok", "count": len(ids), "record_ids": ids, "records": records, "conditions": cond, "message": msg}


def _get_event_video(args: dict, ctx: ToolContext) -> dict[str, Any]:
    eid = args["event_id"]
    # 다른 매장 사건은 '없는 번호'와 똑같이 취급해서 존재 여부도 알려 주지 않는다 (NFR-06)
    if ctx.records is not None:
        if "event" in ctx.records.get("errors", {}):
            return _error("get_event_video", "source_error", _FAIL_MESSAGE["get_event_video"], args)
        r = next((record for record in ctx.records["events"]
                  if record["event_id"] == eid and record["store_id"] == ctx.store_id), None)
    else:
        r = ctx.conn.execute("SELECT * FROM events WHERE event_id=? AND store_id=?", (eid, ctx.store_id)).fetchone()
    if r is None:
        return {"status": "error", "error_code": "not_found", "tool": "get_event_video",
                "message": f"{eid} 번호의 사건 기록이 없습니다."}
    info = {"event_id": eid, "camera_id": r["camera_id"], "event_type": r["event_type"], "occurred_at": r["occurred_at"]}
    clip_uri = r.get("clip_uri") if isinstance(r, dict) else r["clip_uri"]
    if not clip_uri:
        why = ("카메라 연결 끊김 사건은 영상이 들어오지 않는 상황이라 저장된 영상이 없습니다."
               if r["source"] == "time_rule" else "이 사건에는 저장된 영상이 없습니다.")
        return {"status": "no_video", **info, "video_uri": None, "message": f"{eid}: {why}"}
    length = r["clip_length_sec"]
    return {"status": "ok", **info, "video_uri": clip_uri,
            "clip": {"start_sec": 0, "end_sec": length, "length_sec": length},
            "message": f"{eid} 영상 위치: {clip_uri} · 카메라: {r['camera_id']} · 영상 구간: 0~{length}초"}


# ---------------------------------------------------------------- 매뉴얼 검색 + 문장별 근거 (FR-QNA-08, 09)
_COMPOSE_SYSTEM = (
    "너는 매장 규정 문서에서 근거를 찾아 답하는 도우미다. 아래 [조각]에 적힌 내용만 근거로 답한다.\n"
    "조각에 질문의 답이 없으면 지어내지 말고 answerable을 false로 한다.\n"
    "질문이 숫자나 기간을 요구해도 원문이 '점주가 정한다', '별도 승인 정책을 따른다', "
    "'임의로 정하지 않는다'처럼 결정 주체나 미정 상태를 설명하면 answerable을 true로 하고 "
    "그 설명을 답한다. 원문에 없는 숫자를 만들지 않는다.\n"
    "반드시 아래 JSON 한 개만 출력한다. 설명, 마크다운, 코드블록은 쓰지 않는다.\n"
    '{"answerable": true, "sentences": [{"text": "답변 한 문장", "chunk_ids": ["M001-C03"]}]}\n'
    "- 모든 문장에 근거 조각 ID를 하나 이상 적는다. 조각에 없는 ID는 쓰지 않는다.\n"
    "- 조각의 내용을 바탕으로 짧고 정확하게 쓰고, 조각에 없는 내용은 덧붙이지 않는다."
)


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(text[i : j + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _check_composed(obj: dict | None, hit_ids: set[str]) -> str | None:
    """문제가 있으면 이유를 돌려주고, 괜찮으면 None."""
    if obj is None:
        return "JSON 형식이 아닙니다"
    if not isinstance(obj.get("answerable"), bool):
        return "answerable 이 true/false 가 아닙니다"
    if obj["answerable"] is False:
        return None
    sents = obj.get("sentences")
    if not isinstance(sents, list) or not sents:
        return "sentences 가 비어 있습니다"
    for s in sents:
        if not isinstance(s, dict) or not str(s.get("text", "")).strip():
            return "빈 문장이 있습니다"
        ids = s.get("chunk_ids")
        if not isinstance(ids, list) or not ids:
            return f"근거 조각이 없는 문장이 있습니다: {s.get('text')!r}"
        if not set(ids) <= hit_ids:
            return f"검색되지 않은 조각 ID를 썼습니다: {ids}"
    return None


def _search_manual(args: dict, ctx: ToolContext) -> dict[str, Any]:
    mcfg = ctx.cfg.manual
    hits: list[Hit] = search_chunks(ctx.conn, ctx.embedder, args["question"], mcfg.top_k,
                                    mcfg.similarity_threshold, args.get("doc_scope"), mcfg.keyword_threshold)
    insufficient = {
        "status": "insufficient_evidence", "answer": "규정 문서에서 근거를 찾지 못해 답할 수 없습니다. (근거 부족)",
        "sentences": [], "evidence": [], "message": "규정 문서에서 근거를 찾지 못했습니다. (근거 부족)",
    }
    if not hits:
        return {**insufficient, "evidence_reason": "no_matching_chunks"}
    if ctx.llm is None:
        raise RuntimeError("search_manual 에는 LLM 이 필요합니다")

    by_id = {h.chunk_id: h for h in hits}
    pieces = "\n\n".join(f"[{h.chunk_id}] ({h.doc_name} {h.doc_version}, {h.section})\n{h.text}" for h in hits)
    messages = [
        {"role": "system", "content": _COMPOSE_SYSTEM},
        {"role": "user", "content": f"질문: {args['question']}\n\n[조각]\n{pieces}"},
    ]
    obj, problem = None, "응답 없음"
    for _ in range(2):  # 형식이 틀리면 한 번만 다시 요청
        try:
            msg = ctx.llm.chat(messages, json_mode=True, thinking=False)
        except LLMError as e:
            return _error("search_manual", "llm_error", _FAIL_MESSAGE["search_manual"], args, detail=str(e))
        obj = _extract_json(msg.get("content") or "")
        problem = _check_composed(obj, set(by_id))
        if problem is None:
            break
        messages += [{"role": "assistant", "content": msg.get("content") or ""},
                     {"role": "user", "content": f"형식 오류: {problem}. 규칙에 맞춰 JSON만 다시 출력하세요."}]
    if problem is not None:
        return _error("search_manual", "answer_validation_failed", _FAIL_MESSAGE["search_manual"], args, detail=problem)
    if obj["answerable"] is False:
        return {**insufficient, "evidence_reason": "answer_not_in_retrieved_chunks",
                "retrieved_chunks": [{"chunk_id": h.chunk_id, "score": h.score,
                                      "retrieval_method": h.retrieval_method,
                                      "keyword_score": h.keyword_score} for h in hits]}

    sentences = [{"text": s["text"].strip(), "chunk_ids": list(dict.fromkeys(s["chunk_ids"]))} for s in obj["sentences"]]
    cited = list(dict.fromkeys(cid for s in sentences for cid in s["chunk_ids"]))
    evidence = [{"chunk_id": cid, "text": by_id[cid].text, "doc_name": by_id[cid].doc_name,
                 "doc_version": by_id[cid].doc_version, "doc_status": by_id[cid].doc_status,
                 "section": by_id[cid].section, "score": by_id[cid].score,
                 "retrieval_method": by_id[cid].retrieval_method, "keyword_score": by_id[cid].keyword_score}
                for cid in cited]
    answer = " ".join(s["text"] for s in sentences)
    if any("초안" in item["doc_status"] or "미시행" in item["doc_status"] for item in evidence):
        answer = "검토용 초안·미시행 문서의 내용입니다. 운영 기준으로 사용하기 전 승인 여부를 확인하세요. " + answer
    return {"status": "ok", "answer": answer, "sentences": sentences, "evidence": evidence,
            "conditions": {"doc_scope": args.get("doc_scope")}, "message": answer}
