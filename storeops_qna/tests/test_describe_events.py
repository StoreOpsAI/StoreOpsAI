"""지난 사건 사진 해석 도구(describe_events)와 사건 목록 안내 문장을 검증한다.

가짜 LLM이 도구 호출을 정해진 순서로 하고, 백엔드 해석 요청은 가짜로 바꿔 실제 VLM 서버 없이 확인한다.
매뉴얼 색인이 필요 없는 흐름만 다룬다(저장소에 규정 문서가 없어도 실행된다).
실행: python -m unittest discover -s storeops_qna/tests -t .
"""
import unittest
import urllib.error
from unittest.mock import patch

from storeops_qna import tools
from storeops_qna.agent import Agent
from storeops_qna.config import Config
from storeops_qna.db import connect, init_schema
from storeops_qna.testing import FakeLLM, tc, text

NOW = "2026-10-08T21:00:00+09:00"
QUERY = {"store_id": "S01", "record_type": "event", "date_from": "2026-10-08T00:00:00+09:00", "date_to": "2026-10-09T00:00:00+09:00"}


def event(event_id: str, store_id: str = "S01", event_type: str = "fall", hour: int = 14) -> dict:
    return {"event_id": event_id, "store_id": store_id, "camera_id": "CAM-01", "source": "behavior_model", "event_type": event_type,
            "occurred_at": f"2026-10-08T{hour:02d}:20:00+09:00", "clip_uri": "output/clips/x.mp4", "clip_length_sec": 10}


def description(event_id: str, status: str = "ok", category: str = "쓰러짐") -> dict:
    row = {"event_id": event_id, "event_type": "fall", "category_ko": category, "camera_id": "CAM-01",
           "occurred_at": "2026-10-08T14:20:00+09:00", "status": status}
    if status == "ok":
        row.update(description="바닥에 사람이 누워 있는 것으로 보임", model_name="vlm")
    return row


class DescribeEventsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.agent.demo_now = NOW
        self.cfg.backend_url = "http://backend"
        self.records = {"events": [event("E001"), event("E002", event_type="littering", hour=16), event("E900", store_id="S02")],
                        "order_drafts": [], "errors": {}}
        self.conn = connect(":memory:")
        init_schema(self.conn)

    def ask(self, script, question="오늘 무슨 일 있었어?"):
        agent = Agent(self.cfg, self.conn, FakeLLM(script), None)
        return agent.ask(question, "owner-01", store_id="S01", records=self.records)

    def test_query_then_describe_answers_with_photo_interpretation(self):
        script = [tc("query_records", QUERY), tc("describe_events", {"event_ids": ["E001", "E002"]}, "c2"), text("끝")]
        rows = [description("E001"), description("E002", "vlm_unavailable", "쓰레기 투기")]
        with patch.object(tools, "_post_backend_describe", return_value=rows) as post:
            result = self.ask(script)
        self.assertEqual(result.stop_reason, "completed")
        self.assertIn("E001 쓰러짐 의심(10/08 14:20, CAM-01): 바닥에 사람이 누워 있는 것으로 보임", result.answer)
        self.assertIn("E002 쓰레기 투기 의심", result.answer)
        self.assertIn("해석 서버를 사용할 수 없어", result.answer)   # 해석 못 한 사건은 사유를 그대로 알린다
        self.assertIn(tools.DESCRIBE_NOTICE, result.answer)
        self.assertEqual([s["type"] for s in result.sources], ["records", "description"])
        self.assertEqual(post.call_args.args[1], ["E001", "E002"])

    def test_query_message_lists_type_time_camera(self):
        result = self.ask([tc("query_records", QUERY), text("끝")])
        self.assertIn("E001 쓰러짐 10/08 14:20 CAM-01", result.answer)
        self.assertIn("E002 쓰레기 투기 10/08 16:20 CAM-01", result.answer)
        self.assertNotIn("E900", result.answer)   # 다른 매장 사건은 보이지 않는다

    def test_unverified_event_ids_are_not_sent_to_backend(self):
        with patch.object(tools, "_post_backend_describe") as post:
            result = self.ask([tc("describe_events", {"event_ids": ["E001"]}), text("끝")])
        post.assert_not_called()   # query_records로 확인되지 않은 번호는 해석하지 않는다
        self.assertNotIn("바닥에", result.answer)

    def test_other_store_event_is_treated_as_missing(self):
        script = [tc("query_records", QUERY), tc("describe_events", {"event_ids": ["E900"]}, "c2"), text("끝")]
        with patch.object(tools, "_post_backend_describe") as post:
            result = self.ask(script, "E900 사건 내용 알려줘")
        post.assert_not_called()
        self.assertIn("E900 번호의 사건 기록이 없습니다", result.answer)

    def test_vlm_server_down_does_not_invent_description(self):
        script = [tc("query_records", QUERY), tc("describe_events", {"event_ids": ["E001"]}, "c2"), text("끝")]
        with patch.object(tools, "_post_backend_describe", side_effect=urllib.error.URLError("연결 거부")):
            result = self.ask(script)
        self.assertIn("연결하지 못해", result.answer)
        self.assertNotIn(tools.DESCRIBE_NOTICE, result.answer)
        self.assertEqual(result.sources[-1]["type"], "records")   # 해석 출처가 없다

    def test_backend_url_missing_is_reported(self):
        self.cfg.backend_url = ""
        script = [tc("query_records", QUERY), tc("describe_events", {"event_ids": ["E001"]}, "c2"), text("끝")]
        result = self.ask(script)
        self.assertIn("주소가 설정되지 않아", result.answer)

    def test_input_validation(self):
        ctx = tools.ToolContext(self.conn, "S01", None, None, self.cfg, self.records)
        for bad in ([], ["E1"], ["E001"] * 1 + ["X"], [f"E00{i}" for i in range(1, 7)], "E001"):
            with self.assertRaises(tools.ToolInputError):
                tools.validate_input("describe_events", {"event_ids": bad}, ctx)
        self.assertEqual(tools.validate_input("describe_events", {"event_ids": ["E001", "E001"]}, ctx), {"event_ids": ["E001"]})


if __name__ == "__main__":
    unittest.main()
