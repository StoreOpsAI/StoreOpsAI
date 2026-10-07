import json
import unittest
from unittest.mock import patch

from models import vlm_analyzer
from models.vlm_analyzer import VLMAnalyzer
from schemas.event_schema import EventCandidate
from utils import notification


class FakeResponse:
    status = 201

    def __init__(self, body=b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


class VLMDeliveryTest(unittest.TestCase):
    def test_request_context_excludes_classifier_decisions(self):
        request = VLMAnalyzer.build_request(
            ["a.jpg", "b.jpg", "c.jpg"],
            {"event_id": "E001", "camera_id": "CAM-01", "category": "싸움", "scores": {"싸움": 0.9}},
        )

        self.assertEqual(request["event_context"], {"event_id": "E001", "camera_id": "CAM-01"})

    def test_analyzer_validates_three_structured_fields(self):
        result = {
            "observation": "사람이 바닥에 앉아 있습니다.",
            "uncertain_points": "넘어지는 장면은 보이지 않습니다.",
            "owner_actions": ["현장 확인", "직원 확인", "카메라 확인"],
        }
        analyzer = VLMAnalyzer(transport=lambda *_: result)

        self.assertEqual(analyzer.analyze(["a.jpg", "b.jpg", "c.jpg"], {}) , result)
        self.assertEqual(result["model_name"], vlm_analyzer.QWEN_VLM_MODEL)
        with self.assertRaises(ValueError):
            analyzer.analyze(["a.jpg", "b.jpg"], {})

    def test_qwen_chat_completion_json_is_normalized_and_validated(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "observation": "출입문 앞에 사람이 서 있습니다.",
                                "uncertain_points": "",
                                "owner_actions": ["현장 확인", "직원 확인", "카메라 확인"],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        result = VLMAnalyzer._normalize_result(
            VLMAnalyzer._parse_json(VLMAnalyzer._extract_response_text(response))
        )

        self.assertEqual(result["uncertain_points"], "없음")
        self.assertTrue(VLMAnalyzer.validate(result))

    def test_qwen_transport_posts_three_images_to_chat_completions(self):
        result = {
            "observation": "사람이 출입문 앞에 서 있습니다.",
            "uncertain_points": "의도는 확인할 수 없습니다.",
            "owner_actions": ["현장 확인", "직원 확인", "카메라 확인"],
        }
        body = json.dumps(
            {"choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}}]},
            ensure_ascii=False,
        ).encode("utf-8")
        analyzer = VLMAnalyzer(transport=lambda *_: result)

        with (
            patch.object(vlm_analyzer, "QWEN_VLM_BASE_URL", "http://vllm.local:8001/v1"),
            patch.object(vlm_analyzer, "QWEN_VLM_MODEL", "Qwen3-VL-8B-Instruct"),
            patch.object(VLMAnalyzer, "_image_data_url", return_value="data:image/jpeg;base64,AA"),
            patch.object(vlm_analyzer, "urlopen", return_value=FakeResponse(body)) as send_request,
        ):
            received = analyzer._qwen_transport(["first", "middle", "last"], {"camera_id": "CAM-01"})

        request = send_request.call_args.args[0]
        payload = json.loads(request.data)
        images = [
            part for part in payload["messages"][1]["content"]
            if part["type"] == "image_url"
        ]
        self.assertEqual(request.full_url, "http://vllm.local:8001/v1/chat/completions")
        self.assertEqual(payload["model"], "Qwen3-VL-8B-Instruct")
        self.assertEqual(len(images), 3)
        self.assertEqual(received, result)

    def test_missing_vllm_url_returns_not_configured(self):
        with patch.object(vlm_analyzer, "QWEN_VLM_BASE_URL", ""):
            analyzer = VLMAnalyzer()
        try:
            result = analyzer.analyze(["a.jpg", "b.jpg", "c.jpg"], {})
        finally:
            analyzer.close()

        self.assertEqual(result["status"], "VLM_NOT_CONFIGURED")

    def test_alert_response_id_is_used_for_vlm_callback(self):
        event = EventCandidate(
            event_id="E007",
            camera_id="CAM-01",
            event_type="행동",
            category="쓰러짐",
            confidence=0.8,
        )
        alert_response = FakeResponse(b'{"event_id":"E104"}')
        with (
            patch.object(notification, "ALERT_WEBHOOK_URL", "http://backend/events"),
            patch.object(notification, "VLM_RESULT_WEBHOOK_URL", "http://backend/events/{event_id}/vlm"),
            patch.object(notification, "urlopen", return_value=alert_response) as send_request,
        ):
            self.assertTrue(notification.send_first_alert(event))

        self.assertEqual(event.backend_event_id, "E104")
        result = {
            "observation": "관찰 내용",
            "uncertain_points": "불확실한 점",
            "owner_actions": ["확인 1", "확인 2", "확인 3"],
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        }
        with (
            patch.object(notification, "VLM_RESULT_WEBHOOK_URL", "http://backend/events/{event_id}/vlm"),
            patch.object(notification, "urlopen", return_value=FakeResponse()) as send_request,
        ):
            self.assertTrue(notification.send_vlm_result(event, "completed", result))

        request = send_request.call_args.args[0]
        self.assertEqual(request.full_url, "http://backend/events/E104/vlm")
        self.assertEqual(json.loads(request.data)["owner_actions"], result["owner_actions"])
        self.assertEqual(json.loads(request.data)["model_name"], result["model_name"])


if __name__ == "__main__":
    unittest.main()