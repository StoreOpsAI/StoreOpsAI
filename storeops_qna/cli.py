"""명령줄 도구.
  python -m storeops_qna.cli init-db      DB 만들고 기획서 예시 값 넣기
  python -m storeops_qna.cli ingest       M001 규정 문서를 조각 카드로 저장 (임베딩 생성)
  python -m storeops_qna.cli ask "질문"   질문하기
  python -m storeops_qna.cli log 3        3번 질문의 실행 로그 보기
  python -m storeops_qna.cli serve        API 서버 실행
"""
from __future__ import annotations

import argparse
import json
import sys

from .bootstrap import build_agent
from .config import load_config
from .manual_index import ingest_manual
from .seed import seed_demo


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 한글 깨짐 방지
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="storeops_qna")
    p.add_argument("--config", default=None, help="config.yaml 경로")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("ingest")
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--owner", default="owner-01"); a.add_argument("--json", action="store_true")
    l = sub.add_parser("log"); l.add_argument("session_id", type=int); l.add_argument("--owner", default="owner-01")
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    agent = build_agent(cfg)

    if args.cmd == "init-db":
        seed_demo(agent.conn)
        print("DB 준비 완료: 사건 E014·E015, 발주 초안 D001(합성 P001) 등 기획서 예시 값을 넣었습니다.")
    elif args.cmd == "ingest":
        n = ingest_manual(agent.conn, agent.embedder, cfg.resolve(cfg.manual.doc_path), cfg.manual.max_chunk_chars)
        print(f"M001 조각 카드 {n}개를 저장했습니다 (임베딩: {agent.embedder.name}).")
    elif args.cmd == "ask":
        r = agent.ask(args.question, args.owner)
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2) if args.json else r.display())
        print(f"\n(session_id={r.session_id}, 종료 사유={r.stop_reason}, 도구 호출 {r.tool_call_count}회)")
    elif args.cmd == "log":
        print(json.dumps(agent.get_log(args.session_id, args.owner), ensure_ascii=False, indent=2))
    elif args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(cfg, agent), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
