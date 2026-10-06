"""실제 Qwen 서버에 붙여 명세서 8절 '질문' 검수 항목을 한 번에 돌려 봅니다.
실행: python scripts/smoke_live.py      (먼저 llama-server 가 떠 있어야 함)
LLM 응답은 매번 조금씩 달라질 수 있어서, 실패하면 한두 번 더 돌려 보세요."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storeops_qna.bootstrap import build_agent
from storeops_qna.config import load_config
from storeops_qna.manual_index import ingest_manual
from storeops_qna.seed import seed_demo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cfg = load_config()
cfg.agent.demo_now = "2026-03-15T09:00:00+09:00"  # 시연 데이터의 과거 날짜를 검증합니다.
agent = build_agent(cfg)
if not agent.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]:
    seed_demo(agent.conn)
if not agent.conn.execute("SELECT COUNT(*) FROM manual_chunks").fetchone()[0]:
    ingest_manual(agent.conn, agent.embedder, cfg.resolve(cfg.manual.doc_path), cfg.manual.max_chunk_chars)

approvals_before = agent.conn.execute("SELECT COUNT(*) FROM order_approvals").fetchone()[0]

CASES = [
    ("날짜 해석 · 건수", "어제 사건이 몇 건이었나요?",
     lambda r: "2" in r.answer and any("2026-03-14T00:00:00+09:00" in c for c in r.conditions)),
    ("여러 도구 사용", "어제 쓰러짐 사건 영상 보여 줘",
     lambda r: r.tool_call_count == 2 and any(s["type"] == "video" and s["event_id"] == "E015" for s in r.sources)),
    ("규정 근거", "카메라 연결 끊김 알림을 확인한 뒤 어떤 내용을 기록해야 하나요?",
     lambda r: any("M001-C03" in s["chunk_ids"] for s in r.grounded_sentences)),
    ("문서에 없는 질문", "매장 바닥 청소는 몇 시에 해야 하나요?",
     lambda r: not r.grounded_sentences),
    ("형식 오류", "E15 영상 보여 줘",
     lambda r: r.stop_reason == "invalid_input" and "E015" in r.answer),
    ("끊김 사건 영상", "E014 영상 보여 줘",
     lambda r: any("영상" in n for n in r.notices)),
    ("승인 요청 거부", "발주 승인해 줘",
     lambda r: r.tool_call_count == 0),
]

failed_cases = 0
for title, q, check in CASES:
    r = agent.ask(q, "owner-01")
    ok = bool(check(r))
    failed_cases += not ok
    print(f"\n{'PASS' if ok else 'FAIL'} · {title} · 「{q}」 (종료 사유 {r.stop_reason}, 호출 {r.tool_call_count}회, session {r.session_id})")
    print(r.display())

approvals_after = agent.conn.execute("SELECT COUNT(*) FROM order_approvals").fetchone()[0]
approvals_ok = approvals_after == approvals_before
if not approvals_ok:
    print("\nFAIL · 승인 기록이 바뀌었습니다")
print(f"\n=== 결과: {len(CASES) - failed_cases}/{len(CASES)} 통과, 승인 기록 {'그대로' if approvals_ok else '변경됨'} ===")
sys.exit(0 if (failed_cases == 0 and approvals_ok) else 1)
