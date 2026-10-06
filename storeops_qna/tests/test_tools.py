import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from storeops_qna.manual_index import ingest_manual, parse_manual
from storeops_qna.testing import FakeLLM, make_env, text
from storeops_qna.timeutil import date_hints, parse_kst_iso
from storeops_qna.tools import (
    PermissionDenied, ToolContext, ToolInputError, execute_tool, validate_input,
)

YDAY = ("2026-03-14T00:00:00+09:00", "2026-03-15T00:00:00+09:00")


def qr(frm=YDAY[0], to=YDAY[1], store="S01", rt="event"):
    return {"store_id": store, "record_type": rt, "date_from": frm, "date_to": to}


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.conn, self.emb, self.llm, _ = make_env()
        self.ctx = ToolContext(self.conn, "S01", self.emb, self.llm, self.cfg)

    # ---- 날짜 해석 (FR-QNA-04)
    def test_date_hints(self):
        h = {n: (a, b) for n, a, b in date_hints(parse_kst_iso("2026-03-15T09:00:00+09:00"))}
        self.assertEqual(h["어제"], YDAY)
        self.assertEqual(h["이번 주(월~일)"][0], "2026-03-09T00:00:00+09:00")  # 3/15 는 일요일
        self.assertEqual(h["이번 주(월~일)"][1], "2026-03-16T00:00:00+09:00")

    # ---- query_records
    def test_yesterday_two_events(self):
        r = execute_tool("query_records", qr(), self.ctx)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["count"], 2)
        self.assertEqual(r["record_ids"], ["E014", "E015"])
        self.assertEqual(r["count"], len(r["record_ids"]))  # 성공 확인
        self.assertEqual({x["record_id"]: x["event_type"] for x in r["records"]},
                         {"E014": "camera_disconnect", "E015": "fall"})
        self.assertEqual(r["conditions"]["date_from"], YDAY[0])

    def test_event_type_filter_restricts_results(self):
        r = execute_tool("query_records", {**qr(), "event_type": "fall"}, self.ctx)
        self.assertEqual((r["count"], r["record_ids"]), (1, ["E015"]))
        self.assertEqual(r["conditions"]["event_type"], "fall")
        with self.assertRaises(ToolInputError):
            validate_input("query_records", {**qr(rt="order_draft"), "event_type": "fall"}, self.ctx)

    def test_zero_records_is_ok_with_count_zero(self):
        r = execute_tool("query_records", qr("2026-03-10T00:00:00+09:00", "2026-03-11T00:00:00+09:00"), self.ctx)
        self.assertEqual((r["status"], r["count"]), ("ok", 0))
        self.assertIn("없습니다", r["message"])

    def test_not_received(self):
        r = execute_tool("query_records", qr("2026-02-01T00:00:00+09:00", "2026-02-02T00:00:00+09:00"), self.ctx)
        self.assertEqual(r["status"], "not_received")
        self.assertIsNone(r["count"])
        self.assertIn("받지 못했습니다", r["message"])

    def test_storage_error_is_never_zero(self):  # NFR-04
        class Broken:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("disk I/O error")
        ctx = ToolContext(Broken(), "S01", self.emb, self.llm, self.cfg)
        r = execute_tool("query_records", qr(), ctx)
        self.assertEqual(r["status"], "error")
        self.assertIsNone(r["count"])
        self.assertEqual(r["record_ids"], [])
        self.assertIn("실패", r["message"])

    def test_three_outcomes_have_different_messages(self):  # 검수: 세 가지 결과 구분
        zero = execute_tool("query_records", qr("2026-03-10T00:00:00+09:00", "2026-03-11T00:00:00+09:00"), self.ctx)
        miss = execute_tool("query_records", qr("2026-02-01T00:00:00+09:00", "2026-02-02T00:00:00+09:00"), self.ctx)

        class Broken:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("x")
        err = execute_tool("query_records", qr(), ToolContext(Broken(), "S01", self.emb, self.llm, self.cfg))
        self.assertEqual(len({zero["message"], miss["message"], err["message"]}), 3)

    def test_other_store_is_denied(self):  # NFR-06
        r = execute_tool("query_records", qr(store="S02"), self.ctx)
        self.assertEqual((r["status"], r["error_code"]), ("error", "permission_denied"))
        with self.assertRaises(PermissionDenied):
            validate_input("query_records", qr(store="S02"), self.ctx)

    def test_bad_inputs_are_rejected_not_fixed(self):
        bad = [
            qr("2026-03-14", "2026-03-15T00:00:00+09:00"),                       # 시간대 없음
            qr("2026-03-14T00:00:00Z", "2026-03-15T00:00:00+09:00"),             # 한국 시간 아님
            qr("어제", "오늘"),                                                    # 형식 아님
            qr(YDAY[1], YDAY[0]),                                                # 시작이 끝보다 늦음
            qr(rt="payment"),                                                    # 허용 밖 기록 종류
            qr(store="1"),                                                       # 매장 번호 형식
            {"store_id": "S01", "record_type": "event", "date_from": YDAY[0]},   # 필수값 없음
            {**qr(), "limit": 5},                                                # 없는 입력 항목
        ]
        for args in bad:
            with self.subTest(args=args):
                with self.assertRaises(ToolInputError):
                    validate_input("query_records", args, self.ctx)
                self.assertEqual(execute_tool("query_records", args, self.ctx)["status"], "error")

    # ---- get_event_video
    def test_video_ok(self):
        r = execute_tool("get_event_video", {"event_id": "E015"}, self.ctx)
        self.assertEqual((r["status"], r["event_type"], r["camera_id"]), ("ok", "fall", "CAM-01"))
        self.assertTrue(r["video_uri"])
        self.assertEqual(r["clip"]["length_sec"], 10)
        self.assertEqual((r["clip"]["start_sec"], r["clip"]["end_sec"]), (0, 10))
        self.assertIn("CAM-01", r["message"])

    def test_video_for_disconnect_event_explains_why(self):  # FR-QNA-06
        r = execute_tool("get_event_video", {"event_id": "E014"}, self.ctx)
        self.assertEqual(r["status"], "no_video")
        self.assertIsNone(r["video_uri"])
        self.assertIn("영상", r["message"])
        self.assertIn("끊김", r["message"])

    def test_video_not_found_and_format(self):
        self.assertEqual(execute_tool("get_event_video", {"event_id": "E999"}, self.ctx)["error_code"], "not_found")
        with self.assertRaises(ToolInputError) as cm:
            validate_input("get_event_video", {"event_id": "E15"}, self.ctx)
        self.assertEqual(cm.exception.suggestion, "E015")
        self.assertIn("E015가 맞나요?", cm.exception.owner_message())

    def test_video_of_other_store_hidden(self):
        self.conn.execute("INSERT INTO events (event_id, store_id, source, event_type, occurred_at, clip_uri)"
                          " VALUES ('E500','S02','behavior_model','fall','2026-03-14T10:00:00+09:00','clips/x.mp4')")
        r = execute_tool("get_event_video", {"event_id": "E500"}, self.ctx)
        self.assertEqual(r["error_code"], "not_found")  # 존재 여부도 알려 주지 않음

    def test_unknown_tool_rejected(self):  # NFR-05
        with self.assertRaises(ToolInputError):
            validate_input("approve_order", {}, self.ctx)

    # ---- search_manual
    def _compose(self, obj):
        return text(json.dumps(obj, ensure_ascii=False))

    def test_manual_answer_has_sentence_level_evidence(self):
        self.llm.script = [self._compose({"answerable": True, "sentences": [
            {"text": "알림 시각, 마지막 영상 확인 시각, 현재 연결 상태를 기록합니다.", "chunk_ids": ["M001-C03"]},
            {"text": "확인할 수 없는 값은 추정하지 않고 확인 불가로 남깁니다.", "chunk_ids": ["M001-C03"]}]})]
        r = execute_tool("search_manual", {"question": "카메라 연결 끊김 알림을 확인한 뒤 어떤 내용을 기록해야 하나요?"}, self.ctx)
        self.assertEqual(r["status"], "ok")
        ev = {e["chunk_id"]: e for e in r["evidence"]}
        self.assertEqual(list(ev), ["M001-C03"])
        self.assertIn("알림 시각, 마지막으로 영상이 확인된 시각, 현재 연결 상태를 기록한다.", ev["M001-C03"]["text"])
        self.assertEqual((ev["M001-C03"]["doc_version"], ev["M001-C03"]["section"]), ("0.1", "3절"))
        self.assertTrue(all(s["chunk_ids"] for s in r["sentences"]))

    def test_manual_no_close_chunk_means_insufficient_without_llm(self):  # FR-QNA-09
        r = execute_tool("search_manual", {"question": "점심 메뉴 추천 좀 해줘 맛집 어디가 좋아?"}, self.ctx)
        self.assertEqual(r["status"], "insufficient_evidence")
        self.assertEqual(r["evidence"], [])
        self.assertEqual(self.llm.calls, [])  # 추측하려고 LLM을 부르지 않는다

    def test_manual_llm_says_not_answerable(self):
        self.llm.script = [self._compose({"answerable": False, "sentences": []})]
        r = execute_tool("search_manual", {"question": "카메라 연결 끊김 알림 후 기록"}, self.ctx)
        self.assertEqual(r["status"], "insufficient_evidence")

    def test_manual_uncited_or_fake_citation_is_retried_then_fails(self):
        bad1 = self._compose({"answerable": True, "sentences": [{"text": "기록합니다.", "chunk_ids": []}]})
        bad2 = self._compose({"answerable": True, "sentences": [{"text": "기록합니다.", "chunk_ids": ["M999-C01"]}]})
        self.llm.script = [bad1, bad2]
        r = execute_tool("search_manual", {"question": "카메라 연결 끊김 알림 후 기록"}, self.ctx)
        self.assertEqual((r["status"], r["error_code"]), ("error", "answer_validation_failed"))
        self.assertEqual(len(self.llm.calls), 2)

    def test_manual_json_in_code_fence_is_accepted(self):
        good = '```json\n' + json.dumps({"answerable": True, "sentences": [
            {"text": "알림 시각과 마지막 영상 확인 시각을 기록합니다.", "chunk_ids": ["M001-C03"]}]}, ensure_ascii=False) + '\n```'
        self.llm.script = [text(good)]
        r = execute_tool("search_manual", {"question": "카메라 연결 끊김 알림 후 기록"}, self.ctx)
        self.assertEqual(r["status"], "ok")

    # ---- 색인 (FR-QNA-07)
    def test_parse_manual_makes_card_per_section(self):
        chunks = parse_manual((Path(__file__).resolve().parent.parent / "data/manual_M001_v0.1.md").read_text(encoding="utf-8"))
        c3 = next(c for c in chunks if c.chunk_id == "M001-C03")
        self.assertEqual((c3.doc_version, c3.section), ("v0.1", "3절"))
        self.assertIn("확인할 수 없는 값은 추정하여 채우지 않고 확인 불가로 남긴다.", c3.text)

    def test_parse_store_inspection_draft(self):
        path = Path(__file__).resolve().parent.parent / "M001_StoreOps_AI_매장점검규정_v0.1.md"
        chunks = parse_manual(path.read_text(encoding="utf-8"))
        self.assertEqual(chunks[0].doc_name, "StoreOps AI 매장 점검 규정")
        self.assertEqual(chunks[0].doc_version, "0.1")
        self.assertEqual(chunks[0].doc_status, "검토용 초안 · 미시행")
        self.assertTrue(any(chunk.chunk_id.startswith("M001-C03") and "연결 끊김" in chunk.text
                            for chunk in chunks))
        ingest_manual(self.conn, self.emb, path)
        row = self.conn.execute("SELECT doc_version, doc_status, text FROM manual_chunks WHERE chunk_id='M001-C03'").fetchone()
        self.assertEqual((row["doc_version"], row["doc_status"]), ("0.1", "검토용 초안 · 미시행"))
        self.assertIn("연결 끊김", row["text"])

    def test_new_version_replaces_text_chunks_and_embeddings_together(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.md"
            p.write_text("doc_id: M001\ndoc_name: 교육용 매장 점검 기록 규정\ndoc_version: v0.2\n\n"
                         "## 1절. 목적\n새 버전 문장이다.\n", encoding="utf-8")
            ingest_manual(self.conn, self.emb, p)
        rows = self.conn.execute("SELECT chunk_id, doc_version, text FROM manual_chunks").fetchall()
        self.assertEqual([(r["chunk_id"], r["doc_version"], r["text"]) for r in rows], [("M001-C01", "v0.2", "새 버전 문장이다.")])


if __name__ == "__main__":
    unittest.main()
