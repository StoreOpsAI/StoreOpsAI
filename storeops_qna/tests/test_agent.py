import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from storeops_qna.api import create_app
from storeops_qna.llm import LLMError
from storeops_qna.testing import make_env, tc, text

Y = {"store_id": "S01", "record_type": "event",
     "date_from": "2026-03-14T00:00:00+09:00", "date_to": "2026-03-15T00:00:00+09:00"}


def logs(conn, sid):
    return [dict(r) for r in conn.execute("SELECT * FROM tool_calls WHERE session_id=? ORDER BY id", (sid,))]


class AgentTests(unittest.TestCase):
    def test_internal_api_requires_token_and_checks_owner_log(self):
        cfg, conn, emb, llm, agent = make_env([tc("query_records", {**Y, "store_id": "S02"}), text("끝")])
        payload = {"question": "어제 사건?", "owner_id": "real-owner", "store_id": "S02",
                   "events": [], "order_drafts": []}
        with mock.patch.dict("os.environ", {"STOREOPS_QNA_TOKEN": "internal-secret"}):
            client = TestClient(create_app(cfg, agent))
            self.assertEqual(client.post("/internal/ask", json=payload).status_code, 403)
            self.assertEqual(client.post("/api/ask", json={"question": "사건?"},
                                         headers={"X-Owner-Id": "owner-01"}).status_code, 403)
            response = client.post("/internal/ask", json=payload, headers={"X-QNA-Token": "internal-secret"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["sources"][0]["count"], 0)
            session_id = response.json()["session_id"]
            log = client.get(f"/internal/ask/{session_id}/log?owner_id=real-owner&store_id=S02",
                             headers={"X-QNA-Token": "internal-secret"})
            self.assertEqual(log.status_code, 200)
            other = client.get(f"/internal/ask/{session_id}/log?owner_id=other&store_id=S02",
                               headers={"X-QNA-Token": "internal-secret"})
            self.assertEqual(other.status_code, 404)

    def test_yesterday_count_shows_conditions_and_logs(self):
        cfg, conn, emb, llm, agent = make_env([
            tc("query_records", Y, content="건수 질문이라 기록 조회를 씁니다."),
            text("어제 사건은 2건입니다. (E014, E015)")])
        r = agent.ask("어제 사건이 몇 건이었나요?", "owner-01")
        self.assertEqual((r.stop_reason, r.tool_call_count), ("completed", 1))
        self.assertIn("2026-03-14T00:00:00+09:00 ~ 2026-03-15T00:00:00+09:00", r.display())  # 조회 조건 표시
        self.assertEqual(r.sources[0]["record_ids"], ["E014", "E015"])
        rows = logs(conn, r.session_id)
        self.assertEqual([(x["tool_name"], x["result_status"]) for x in rows], [("query_records", "ok")])
        self.assertIn("기록 조회", rows[0]["reason"])  # 고른 이유 (FR-QNA-14)
        # 시스템 프롬프트에 오늘 날짜와 '어제' 표가 들어간다
        sysmsg = llm.calls[0]["messages"][0]["content"]
        self.assertIn("2026-03-15", sysmsg)
        self.assertIn("어제: date_from=2026-03-14T00:00:00+09:00", sysmsg)

    def test_two_tools_in_sequence(self):  # FR-QNA-03: 기록 조회로 E015를 찾은 뒤 영상 조회
        cfg, conn, emb, llm, agent = make_env([
            tc("query_records", {**Y, "event_type": "fall"}), tc("get_event_video", {"event_id": "E015"}, "c2"),
            text("검증되지 않은 영상 위치입니다.")])
        r = agent.ask("어제 쓰러짐 사건 영상 보여 줘", "owner-01")
        self.assertEqual((r.stop_reason, r.tool_call_count), ("completed", 2))
        self.assertEqual([x["tool_name"] for x in logs(conn, r.session_id)], ["query_records", "get_event_video"])
        self.assertEqual(r.sources[0]["record_ids"], ["E015"])
        self.assertIn("사건 종류 fall", r.conditions[0])
        self.assertEqual(r.sources[-1], {"type": "video", "event_id": "E015", "video_uri": "clips/E015.mp4",
                         "camera_id": "CAM-01", "clip": {"start_sec": 0, "end_sec": 10, "length_sec": 10}})
        self.assertEqual(r.answer, "E015 영상 위치: clips/E015.mp4 · 카메라: CAM-01 · 영상 구간: 0~10초")
        # 두 번째 LLM 호출에 첫 도구 결과가 들어 있다 (앞 결과를 보고 다음 도구를 고름)
        self.assertTrue(any(m["role"] == "tool" and "E015" in m["content"] for m in llm.calls[1]["messages"]))

    def test_authenticated_snapshot_replaces_demo_records(self):
        cfg, conn, emb, llm, agent = make_env([
            tc("query_records", {**Y, "store_id": "S02", "event_type": "fall"}),
            tc("get_event_video", {"event_id": "E500"}, "c2"), text("끝")])
        snapshot = {"events": [{"event_id": "E500", "store_id": "S02", "source": "behavior_model",
                               "event_type": "fall", "camera_id": "CAM-02", "occurred_at": "2026-03-14T21:12:00+09:00",
                               "clip_uri": "clips/E500.mp4", "clip_length_sec": 12},
                              {"event_id": "E501", "store_id": "S01", "source": "behavior_model",
                               "event_type": "fall", "camera_id": "CAM-01", "occurred_at": "2026-03-14T21:12:00+09:00",
                               "clip_uri": "clips/E501.mp4", "clip_length_sec": 12}], "order_drafts": []}
        result = agent.ask("어제 쓰러짐 사건 영상", "real-owner", store_id="S02", records=snapshot)
        self.assertEqual((result.tool_call_count, result.sources[0]["record_ids"]), (2, ["E500"]))
        self.assertEqual(result.sources[-1]["event_id"], "E500")
        self.assertEqual(agent.get_log(result.session_id, "real-owner", store_id="S02")["session"]["store_id"], "S02")
        with self.assertRaises(PermissionError):
            agent.get_log(result.session_id, "owner-02")

    def test_bad_event_id_is_rejected_and_owner_is_asked(self):  # 검수: 형식 오류
        cfg, conn, emb, llm, agent = make_env([tc("get_event_video", {"event_id": "E15"})])
        r = agent.ask("E15 영상 보여 줘", "owner-01")
        self.assertEqual((r.stop_reason, r.needs_confirmation), ("invalid_input", True))
        self.assertIn("E015가 맞나요?", r.answer)
        row = logs(conn, r.session_id)[0]
        self.assertEqual(row["result_status"], "rejected")
        self.assertIn("입력 검사 실패", row["rejected_reason"])
        self.assertIsNone(row["output"])  # 실행하지 않았다
        self.assertEqual(len(llm.calls), 0)  # 원문 번호 오류는 모델을 호출하기 전에 반려한다

    def test_owner_event_id_is_checked_before_model_can_correct_it(self):
        cfg, conn, emb, llm, agent = make_env([tc("get_event_video", {"event_id": "E001"})])
        r = agent.ask("E01 영상 보여 줘", "owner-01")
        self.assertEqual((r.stop_reason, r.needs_confirmation), ("invalid_input", True))
        self.assertIn("E001이 맞나요?", r.answer)
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(logs(conn, r.session_id)[0]["result_status"], "rejected")

    def test_video_id_not_in_question_or_records_is_not_queried(self):
        cfg, conn, emb, llm, agent = make_env([
            tc("get_event_video", {"event_id": "E015"}),
            tc("query_records", {**Y, "event_type": "fall"}, "c2"),
            tc("get_event_video", {"event_id": "E015"}, "c3"), text("끝")])
        r = agent.ask("어제 쓰러짐 사건 영상 보여 줘", "owner-01")
        self.assertEqual(r.tool_call_count, 3)
        self.assertEqual([(row["tool_name"], row["result_status"]) for row in logs(conn, r.session_id)],
                         [("get_event_video", "rejected"), ("query_records", "ok"), ("get_event_video", "ok")])
        self.assertEqual(r.sources[-1]["event_id"], "E015")

    def test_two_consecutive_failures_stop(self):  # 검수: 종료 기준
        cfg, conn, emb, llm, agent = make_env([tc("query_records", Y), tc("query_records", Y, "c2"), text("안 쓰임")])
        fail = {"status": "error", "message": "기록 조회에 실패했습니다.", "count": None, "record_ids": [], "conditions": Y}
        with mock.patch("storeops_qna.agent.execute_tool", return_value=dict(fail)):
            r = agent.ask("어제 사건 몇 건?", "owner-01")
        self.assertEqual((r.stop_reason, r.needs_confirmation, r.tool_call_count), ("consecutive_failure", True, 2))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(conn.execute("SELECT stop_reason FROM qa_sessions WHERE session_id=?", (r.session_id,)).fetchone()[0],
                         "consecutive_failure")
        self.assertTrue(r.notices)

    def test_failure_streak_resets_when_tool_differs_or_succeeds(self):
        cfg, conn, emb, llm, agent = make_env([
            tc("query_records", Y), tc("get_event_video", {"event_id": "E015"}, "c2"),
            tc("query_records", Y, "c3"), text("끝")])
        seq = [{"status": "error", "message": "x", "tool": "query_records"},
               {"status": "ok", "message": "ok", "event_id": "E015", "video_uri": "u"},
               {"status": "error", "message": "x", "tool": "query_records"}]
        with mock.patch("storeops_qna.agent.execute_tool", side_effect=seq):
            r = agent.ask("E015 영상과 기록 질문", "owner-01")
        self.assertEqual(r.stop_reason, "completed")  # 실패-성공-실패 는 '연속 2회'가 아니다

    def test_more_than_five_calls_stop(self):
        cfg, conn, emb, llm, agent = make_env([tc("query_records", Y, f"c{i}") for i in range(6)])
        r = agent.ask("어제 사건?", "owner-01")
        self.assertEqual((r.stop_reason, r.tool_call_count, r.needs_confirmation), ("call_limit", 5, True))
        rows = logs(conn, r.session_id)
        self.assertEqual(rows[-1]["result_status"], "stopped")
        self.assertIn("호출 한도", rows[-1]["rejected_reason"])
        self.assertEqual(len([x for x in rows if x["result_status"] == "ok"]), 5)

    def test_tool_outside_allowlist_is_rejected(self):  # NFR-05
        cfg, conn, emb, llm, agent = make_env([
            tc("approve_order", {"draft_id": "D001"}),
            text("저는 조회만 할 수 있어 직접 처리할 수 없습니다.")])
        r = agent.ask("발주 승인해 줘", "owner-01")
        row = logs(conn, r.session_id)[0]
        self.assertEqual((row["result_status"], row["rejected_reason"]), ("rejected", "허용 목록에 없는 도구"))
        self.assertIsNone(row["output"])
        self.assertEqual(r.stop_reason, "completed")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM order_approvals").fetchone()[0], 1)  # 시드 1건 그대로

    def test_other_store_query_is_denied(self):
        cfg, conn, emb, llm, agent = make_env([tc("query_records", {**Y, "store_id": "S02"})])
        r = agent.ask("S02 매장 어제 사건?", "owner-01")
        self.assertEqual(r.stop_reason, "permission_denied")
        self.assertEqual(conn.execute("SELECT result_status FROM tool_calls").fetchone()[0], "rejected")

    def test_manual_question_answers_directly_with_evidence(self):  # 검수: 규정 근거
        compose = json.dumps({"answerable": True, "sentences": [
            {"text": "알림 시각과 마지막으로 영상이 확인된 시각을 기록합니다.", "chunk_ids": ["M001-C03"]},
            {"text": "확인할 수 없는 값은 추정하지 않고 확인 불가로 남깁니다.", "chunk_ids": ["M001-C03"]}]}, ensure_ascii=False)
        cfg, conn, emb, llm, agent = make_env([
            tc("search_manual", {"question": "카메라 연결 끊김 알림을 확인한 뒤 어떤 내용을 기록해야 하나요?"}),
            text(compose)])
        r = agent.ask("카메라 연결 끊김 알림을 확인한 뒤 어떤 내용을 기록해야 하나요?", "owner-01")
        self.assertEqual((r.stop_reason, len(llm.calls)), ("completed", 2))  # Agent 판단 1회 + 답변 구성 1회
        self.assertEqual([s["chunk_ids"] for s in r.grounded_sentences], [["M001-C03"], ["M001-C03"]])
        self.assertIn("확인할 수 없는 값은 추정하여 채우지 않고 확인 불가로 남긴다.",
                  r.grounded_sentences[1]["evidence"][0]["text"])
        self.assertEqual(r.grounded_sentences[1]["evidence"][0]["doc_version"], "0.1")
        out = r.display()
        self.assertIn("StoreOps AI 매장 점검 규정 0.1, 3절", out)
        self.assertIn("확인할 수 없는 값은 추정하여 채우지 않고 확인 불가로 남긴다.", out)

    def test_manual_without_evidence_says_insufficient(self):
        cfg, conn, emb, llm, agent = make_env([tc("search_manual", {"question": "점심 메뉴 추천 좀 해줘 맛집 어디가 좋아?"})])
        r = agent.ask("점심 메뉴 추천해 줘", "owner-01")
        self.assertIn("근거", r.answer)
        self.assertEqual(r.sources, [])
        self.assertEqual(len(llm.calls), 1)

    def test_zero_and_not_received_notices(self):
        z = {**Y, "date_from": "2026-03-10T00:00:00+09:00", "date_to": "2026-03-11T00:00:00+09:00"}
        m = {**Y, "date_from": "2026-02-01T00:00:00+09:00", "date_to": "2026-02-02T00:00:00+09:00"}
        for args, needle in ((z, "없습니다"), (m, "받지 못했습니다")):
            cfg, conn, emb, llm, agent = make_env([tc("query_records", args), text("정리한 답")])
            r = agent.ask("사건 몇 건?", "owner-01")
            self.assertTrue(any(needle in n for n in r.notices), r.notices)
            self.assertIn(needle, r.answer)

    def test_model_text_without_tool_cannot_claim_approval(self):
        cfg, conn, emb, llm, agent = make_env([text("발주를 승인했습니다.")])
        r = agent.ask("발주 승인해 줘", "owner-01")
        self.assertIn("조회만", r.answer)
        self.assertNotIn("승인했습니다", r.answer)

    def test_only_first_of_parallel_calls_runs(self):
        two = {"content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "query_records", "arguments": json.dumps(Y)}},
            {"id": "b", "type": "function", "function": {"name": "get_event_video", "arguments": '{"event_id":"E015"}'}}]}
        cfg, conn, emb, llm, agent = make_env([two, text("끝")])
        r = agent.ask("질문", "owner-01")
        self.assertEqual(r.tool_call_count, 1)
        tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["b", "a"])  # 모든 호출 id에 응답이 붙는다

    def test_llm_down_is_reported_not_crashed(self):
        cfg, conn, emb, llm, agent = make_env([LLMError("연결 거부")])
        r = agent.ask("어제 사건?", "owner-01")
        self.assertEqual(r.stop_reason, "llm_error")

    def test_owner_scoping_of_sessions_and_logs(self):  # FR-QNA-12
        cfg, conn, emb, llm, agent = make_env([text("안녕하세요")])
        r = agent.ask("안녕", "owner-01")
        self.assertEqual(agent.get_log(r.session_id, "owner-01")["session"]["store_id"], "S01")
        with self.assertRaises(PermissionError):
            agent.get_log(r.session_id, "owner-02")
        with self.assertRaises(PermissionError):
            agent.ask("안녕", "stranger")

    def test_no_state_carries_over_between_questions(self):  # FR-QNA-13
        cfg, conn, emb, llm, agent = make_env([text("첫 답"), text("둘째 답")])
        agent.ask("첫 질문", "owner-01")
        agent.ask("둘째 질문", "owner-01")
        second_msgs = llm.calls[1]["messages"]
        self.assertEqual([m["role"] for m in second_msgs], ["system", "user"])
        self.assertEqual(second_msgs[1]["content"], "둘째 질문")


if __name__ == "__main__":
    unittest.main()
