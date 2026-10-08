"""질문 Agent (FR-QNA-01, 03~06, 10~14).

제어 루프를 파이썬으로 직접 구현했습니다 (명세서 6절 제안: 관찰 → 판단 → 실행 → 저장 → 종료 판단이 코드와 로그에 그대로 보임).
LLM은 '어떤 도구를 어떤 입력값으로 부를지'만 정하고, 검사·실행·승인 여부는 모두 코드가 맡습니다.
State는 한 질문을 처리하는 동안의 messages 목록뿐이고, 질문이 끝나면 버립니다 (FR-QNA-13).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import Config
from .llm import LLMClient, LLMError
from .question_text import normalize_question
from .timeutil import date_hints, now_kst, KST
from .tools import (
    TOOL_DEFS, TOOL_NAMES, PermissionDenied, ToolContext, ToolInputError, execute_tool, validate_input,
)

SYSTEM_PROMPT = """당신은 무인매장 점주의 질문에 답하는 Agent입니다. 한국어로 짧고 정확하게 답합니다.

[할 수 있는 일]
- 아래 도구 네 개로 '조회'만 할 수 있습니다: query_records, search_manual, describe_events, get_event_video
- 발주 승인, 알림 발송, 기록 수정, 상태 변경은 할 수 없습니다. 그런 요청은 도구를 부르지 말고 "저는 조회만 할 수 있어 직접 처리할 수 없습니다. 화면에서 점주님이 직접 처리해 주세요."라고 답하세요.

[현재 정보]
- 지금 시각(한국 시간): {now}
- 점주 매장 번호: {store_id} (store_id에는 항상 이 값만 씁니다. 다른 매장은 조회할 수 없습니다)
- 날짜 표현은 아래 표의 값을 그대로 date_from(포함), date_to(끝, 미포함)에 씁니다:
{hints}

[도구 고르는 법]
- 질문의 일반적인 오타와 띄어쓰기 누락은 문맥으로 이해합니다. 뜻이 불명확하면 추측하지 말고 확인합니다. 사건 번호, 날짜, 수량은 임의로 고치지 않습니다.
- 건수·기록 질문 → query_records (record_type: event=사건, order_draft=발주 초안). 쓰러짐처럼 종류를 지정하면 event_type 필터를 씁니다.
- 규정·절차 질문 → search_manual (질문 문장을 그대로 question에 넣습니다)
- 지난 사건의 내용(그날 무슨 일이 있었는지, 어떤 사건이었는지) → 먼저 query_records로 사건 번호를 찾고, 그 번호로 describe_events (사건 사진을 보고 해석한다)
- 특정 사건 영상 → get_event_video. 사건 번호를 모르면 먼저 query_records로 사건 목록을 찾고, 그 결과를 보고 다음 도구를 고릅니다.
- 도구는 한 번에 하나만 부르고, 결과를 본 뒤 다음 행동을 정합니다. 도구를 부르기 전에 이유를 한 문장으로 적습니다.

[지켜야 할 규칙]
1. 숫자, 사건 번호, 영상 위치는 도구 결과에 있는 값만 씁니다. 짐작해서 쓰지 않습니다.
2. 점주가 말한 사건 번호는 형식을 고치지 말고 event_id에 그대로 넣습니다(예: 'E15'를 'E015'로 바꾸지 않습니다).
3. 결과 상태별로 다르게 답합니다.
   - status=ok, count=0 → "해당 기간에 기록된 ○○이 없습니다"
   - status=error → "조회에 실패했습니다"라고 알리고, 0건이라고 말하지 않습니다
   - status=not_received → "그 기간의 기록을 아직 받지 못했습니다"
   - status=no_video → 도구가 알려 준 영상이 없는 이유를 그대로 전합니다
   - status=insufficient_evidence → 근거가 부족하다고 답하고 추측하지 않습니다
   - status=vlm_unavailable → 사진 해석을 할 수 없다고 알리고, 사건 번호와 영상 확인을 안내합니다. 사건 내용을 추측하지 않습니다
4. 도구 결과의 message 문장을 우선 활용하고, 도구 호출 없이 추측으로 답하지 않습니다.
5. 점주 개인정보나 영상 속 인물에 대해서는 묻지도, 추측하지도 않습니다."""


@dataclass
class AskResult:
    session_id: int
    answer: str
    stop_reason: str
    needs_confirmation: bool = False
    conditions: list[str] = field(default_factory=list)  # 사용한 조회 조건 (FR-QNA-04)
    sources: list[dict] = field(default_factory=list)  # 출처 (FR-UI-09)
    notices: list[str] = field(default_factory=list)  # 0건/실패/미수신 고정 문장 (FR-QNA-05)
    tool_call_count: int = 0
    grounded_sentences: list[dict] = field(default_factory=list)  # 규정 답변의 문장별 근거 (FR-QNA-08)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "answer": self.answer, "stop_reason": self.stop_reason,
            "needs_confirmation": self.needs_confirmation, "conditions": self.conditions, "sources": self.sources,
            "notices": self.notices, "tool_call_count": self.tool_call_count,
            "grounded_sentences": self.grounded_sentences,
        }

    def display(self) -> str:
        parts = [self.answer]
        if self.grounded_sentences:
            parts.append("\n[문장별 근거]")
            for s in self.grounded_sentences:
                parts.append(f"- {s['text']}  ← {', '.join(s['chunk_ids'])}")
        if self.notices:
            parts.append("\n[조회 상태] " + " / ".join(self.notices))
        if self.conditions:
            parts.append("\n[조회 조건]\n" + "\n".join(f"- {c}" for c in self.conditions))
        if self.sources:
            parts.append("\n[출처]")
            for s in self.sources:
                if s["type"] == "manual":
                    parts.append(f"- {s['doc_name']} {s['doc_version']}, {s['section']} ({s['chunk_id']}): {s['text']}")
                elif s["type"] == "records":
                    parts.append(f"- 매장 기록 조회 결과: {', '.join(s['record_ids']) or '없음'} ({s['count']}건)")
                elif s["type"] == "video":
                    parts.append(f"- 사건 영상 {s['event_id']}: {s['video_uri']}")
                elif s["type"] == "description":
                    parts.append(f"- 사건 사진 해석 {s['event_id']} {s.get('category_ko') or ''}: {s['text']}")
        return "\n".join(parts)


class Agent:
    def __init__(self, cfg: Config, conn, llm: LLMClient, embedder):
        self.cfg, self.conn, self.llm, self.embedder = cfg, conn, llm, embedder

    # ------------------------------------------------------------ 공개 진입점
    def ask(self, question: str, owner_id: str, *, store_id: str | None = None,
            records: dict | None = None) -> AskResult:
        store_id = store_id if records is not None else self.cfg.owners.get(owner_id)
        if store_id is None:
            raise PermissionError(f"등록되지 않은 점주입니다: {owner_id}")  # FR-QNA-12
        question = (question or "").strip()
        now = now_kst(self.cfg.agent.demo_now)
        cur = self.conn.execute(
            "INSERT INTO qa_sessions (store_id, owner_id, question, created_at) VALUES (?,?,?,?)",
            (store_id, owner_id, question, now.astimezone(KST).isoformat(timespec="seconds")),
        )
        self.conn.commit()
        sid = cur.lastrowid
        result = self._run(sid, question, store_id, now, records)
        self.conn.execute("UPDATE qa_sessions SET answer=?, stop_reason=? WHERE session_id=?",
                          (result.answer, result.stop_reason, sid))
        self.conn.commit()
        return result

    # ------------------------------------------------------------ 제어 루프
    def _run(self, sid: int, question: str, store_id: str, now: datetime, records: dict | None = None) -> AskResult:
        acfg = self.cfg.agent
        ctx = ToolContext(self.conn, store_id, self.embedder, self.llm, self.cfg, records)
        hints = "\n".join(f"  · {n}: date_from={a}, date_to={b}" for n, a, b in date_hints(now))
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                now=now.astimezone(KST).strftime("%Y-%m-%d %H:%M (%A)"), store_id=store_id, hints=hints)},
            {"role": "user", "content": normalize_question(question)},
        ]
        executed: list[dict] = []  # 실제 실행한 도구 결과 (이름 포함)
        attempts = 0  # 반려된 호출도 포함한 호출 시도 수
        step = 0
        fail_streak, last_failed_tool = 0, None

        def finish(answer: str, reason: str, confirm: bool = False) -> AskResult:
            return self._build_result(sid, answer, reason, confirm, executed, attempts)

        if not question:
            return finish("질문을 입력해 주세요.", "empty_question")

        # 점주가 입력한 번호를 모델이 고쳐서 조회하기 전에 원문 형식을 확인한다.
        mentioned_id = re.search(r"\bE\d+\b", question, re.IGNORECASE)
        if mentioned_id and not re.fullmatch(r"E\d{3}", mentioned_id.group(), re.IGNORECASE):
            invalid_id = mentioned_id.group()
            digits = invalid_id[1:]
            suggestion = f"E{int(digits):03d}" if int(digits) < 1000 else None
            error = ToolInputError("event_id", invalid_id, "사건 번호는 E와 숫자 세 자리여야 합니다", suggestion)
            self._log(sid, 1, "get_event_video", {"event_id": invalid_id}, "rejected", None,
                      "점주 입력 형식 확인", f"입력 검사 실패: {error}")
            return finish(error.owner_message(), "invalid_input", True)

        for _ in range(acfg.max_tool_calls + 3):
            try:
                msg = self.llm.chat(messages, tools=TOOL_DEFS)
            except LLMError as e:
                self._log(sid, step, None, None, "llm_error", {"error": str(e)}, None, None)
                return finish("답변을 만드는 모델에 연결하지 못했습니다. 잠시 뒤 다시 질문해 주세요.", "llm_error")

            calls = msg.get("tool_calls") or []
            if not calls:  # 도구 없이 답이 나옴 = 종료 판단
                # 모델의 자유 서술은 조회 결과와 달라도 검증할 수 없으므로 답변에 사용하지 않는다.
                if executed:
                    answer = executed[-1].get("message") or "조회 결과를 확인하지 못했습니다."
                else:
                    answer = "저는 조회만 할 수 있습니다. 사건·발주 기록, 영상 또는 규정에 대해 질문해 주세요."
                return finish(answer, "completed")

            reason = ((msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()[:300]
                      or {"query_records": "매장 기록의 건수와 사건 번호를 확인하기 위해 조회",
                          "search_manual": "점검 규정의 원문 근거를 확인하기 위해 검색",
                          "get_event_video": "확인된 사건의 영상 정보를 조회"}.get(
                              calls[0]["function"]["name"], "허용되지 않은 도구 요청 확인"))
            first, extras = calls[0], calls[1:]
            name = first["function"]["name"]
            raw = first["function"].get("arguments") or "{}"
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for ex in extras:  # 도구는 한 번에 하나씩 (FR-QNA-03). 나머지는 건너뛰고 알려 준다
                messages.append({"role": "tool", "tool_call_id": ex["id"], "content": json.dumps(
                    {"status": "skipped", "message": "도구는 한 번에 하나씩만 부를 수 있습니다. 앞 결과를 보고 다시 요청하세요."},
                    ensure_ascii=False)})

            def reply(payload: dict) -> None:
                messages.append({"role": "tool", "tool_call_id": first["id"],
                                 "content": json.dumps(payload, ensure_ascii=False)})

            step += 1
            # (1) 호출 한도: 전체 호출이 max_tool_calls 를 넘으면 멈춘다 (FR-QNA-11)
            if attempts >= acfg.max_tool_calls:
                self._log(sid, step, name, raw, "stopped", None, reason, f"호출 한도 초과({acfg.max_tool_calls}회)")
                return finish(f"조회를 {acfg.max_tool_calls}번 시도했는데도 답을 찾지 못해 멈췄습니다. "
                              "조회 조건(기간, 사건 번호 등)을 바꿔서 다시 질문할까요?", "call_limit", True)
            attempts += 1

            # (2) 허용 목록 (NFR-05)
            if name not in TOOL_NAMES:
                self._log(sid, step, name, raw, "rejected", None, reason, "허용 목록에 없는 도구")
                reply({"status": "rejected", "message": f"'{name}' 은(는) 사용할 수 없습니다. "
                       f"조회 도구 {', '.join(TOOL_NAMES)} 만 쓸 수 있습니다."})
                continue

            # (3) 입력 검사 (FR-QNA-10): 틀리면 고치지 않고 점주에게 확인 요청
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                validate_input(name, args, ctx)
            except json.JSONDecodeError:
                self._log(sid, step, name, raw, "rejected", None, reason, "입력값이 JSON이 아님")
                reply({"status": "rejected", "message": "입력값이 올바른 JSON이 아닙니다."})
                continue
            except PermissionDenied as e:
                self._log(sid, step, name, raw, "rejected", None, reason, f"권한 없음: {e.reason}")
                return finish(f"{e.reason}. 다른 매장의 기록은 조회할 수 없습니다.", "permission_denied")
            except ToolInputError as e:
                self._log(sid, step, name, raw, "rejected", None, reason, f"입력 검사 실패: {e}")
                return finish(e.owner_message(), "invalid_input", True)

            # 질문에 없는 사건 번호는 앞선 기록 조회에서 확인된 경우에만 영상 조회·사진 해석에 사용한다.
            used_ids = [args["event_id"]] if name == "get_event_video" else (args["event_ids"] if name == "describe_events" else [])
            if used_ids:
                found_ids = {record_id for result in executed if result.get("tool") == "query_records"
                             and result.get("status") == "ok" for record_id in result.get("record_ids", [])}
                found_ids |= {i.upper() for i in re.findall(r"E\d+", question, re.IGNORECASE)}
                if any(i not in found_ids for i in used_ids):
                    self._log(sid, step, name, args, "rejected", None, reason, "기록 조회로 확인되지 않은 사건 번호")
                    reply({"status": "rejected", "message": "먼저 query_records로 사건 번호를 확인하세요."})
                    continue

            # (4) 실행
            result = execute_tool(name, args, ctx)
            result["tool"] = name
            executed.append(result)
            self._log(sid, step, name, args, result["status"], result, reason, None)

            if result["status"] == "error":
                fail_streak = fail_streak + 1 if last_failed_tool == name else 1
                last_failed_tool = name
                if fail_streak >= acfg.max_consecutive_failures:
                    return finish(f"{name} 조회가 연속 {fail_streak}번 실패해 멈췄습니다. ({result.get('message', '')}) "
                                  "조회 조건을 바꿔서 다시 시도할까요?", "consecutive_failure", True)
            else:
                fail_streak, last_failed_tool = 0, None

            # (5) 매뉴얼 검색 한 번으로 끝나는 규정 질문은 문장별 근거가 붙은 도구 답변을 그대로 쓴다 (FR-QNA-08)
            if name == "search_manual" and result["status"] in ("ok", "insufficient_evidence") and len(executed) == 1:
                return finish(result["answer"], "completed")

            reply(result)

        return finish("답을 정리하지 못하고 멈췄습니다. 질문을 조금 더 구체적으로 다시 해 주세요.", "max_steps")

    # ------------------------------------------------------------ 결과 조립·로그
    def _build_result(self, sid, answer, reason, confirm, executed, attempts) -> AskResult:
        conditions, sources, notices, grounded = [], [], [], []
        for r in executed:
            t, st = r.get("tool"), r.get("status")
            if t == "query_records":
                c = r.get("conditions", {})
                conditions.append(f"query_records · 매장 {c.get('store_id')} · {c.get('record_type')} · "
                                  f"{c.get('date_from')} ~ {c.get('date_to')} (끝 시각 미포함)"
                                  + (f" · 사건 종류 {c['event_type']}" if c.get("event_type") else ""))
                if st == "ok":
                    sources.append({"type": "records", "tool": t, "count": r["count"], "record_ids": r["record_ids"]})
                if st in ("error", "not_received") or (st == "ok" and r["count"] == 0):
                    notices.append(r["message"])
            elif t == "get_event_video":
                if r.get("event_id"):
                    conditions.append(f"get_event_video · {r['event_id']}")
                if st == "ok":
                    sources.append({"type": "video", "event_id": r["event_id"], "video_uri": r["video_uri"],
                                    "camera_id": r.get("camera_id"), "clip": r.get("clip")})
                else:
                    notices.append(r["message"])
            elif t == "describe_events":
                conditions.append(f"describe_events · {', '.join(r.get('event_ids', []))}")
                if st == "ok":
                    sources += [{"type": "description", "event_id": d["event_id"], "category_ko": d.get("category_ko"),
                                 "text": d["description"]} for d in r["descriptions"] if d.get("status") == "ok"]
                else:
                    notices.append(r["message"])
            elif t == "search_manual":
                scope = (r.get("conditions") or {}).get("doc_scope")
                conditions.append(f"search_manual · 문서 범위 {scope or '전체'}")
                if st == "ok":
                    evidence = {entry["chunk_id"]: entry for entry in r["evidence"]}
                    grounded += [{**sentence, "evidence": [evidence[cid] for cid in sentence["chunk_ids"]]}
                                 for sentence in r["sentences"]]
                    sources += [{"type": "manual", **e} for e in r["evidence"]]
                else:
                    notices.append(r["message"])
        # 같은 출처 중복 제거
        seen, uniq = set(), []
        for s in sources:
            key = json.dumps(s, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return AskResult(sid, answer, reason, confirm, conditions, uniq, notices, attempts, grounded)

    def _log(self, sid, step, name, args, status, output, reason, rejected) -> None:
        """FR-QNA-14: 단계마다 부른 도구, 입력값, 결과 상태, 고른 이유(와 반려 사유)를 남긴다."""
        dump = lambda v: None if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        self.conn.execute(
            "INSERT INTO tool_calls (session_id, step, tool_name, input, result_status, output, reason, rejected_reason, called_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, step, name, dump(args), status, dump(output), reason, rejected,
             datetime.now(KST).isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def get_log(self, session_id: int, owner_id: str, *, store_id: str | None = None) -> dict:
        store_id = store_id or self.cfg.owners.get(owner_id)
        s = self.conn.execute("SELECT * FROM qa_sessions WHERE session_id=?", (session_id,)).fetchone()
        if s is None or store_id is None or s["store_id"] != store_id or s["owner_id"] != owner_id:
            raise PermissionError("이 질문 기록을 볼 수 없습니다")  # 다른 매장 기록은 존재 여부도 숨김
        calls = self.conn.execute("SELECT step, tool_name, input, result_status, output, reason, rejected_reason, called_at "
                                  "FROM tool_calls WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return {"session": dict(s), "tool_calls": [dict(c) for c in calls]}
