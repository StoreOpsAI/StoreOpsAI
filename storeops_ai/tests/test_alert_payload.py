import json
import unittest
from unittest.mock import patch

from schemas.event_schema import EventCandidate
from utils import notification


def make_event() -> EventCandidate:
    return EventCandidate(
        event_id="E001", camera_id="CAM-01", event_type="행동", category="쓰러짐",
        confidence=0.99, threshold=0.98, scores={"정상": 0.01, "쓰러짐": 0.99}, risk_level="경고", score_logit=4.6,
        clip_path="output/clips/E001.mp4", representative_images=["output/representative_images/E001_1.jpg"],
    )


class AlertPayloadTest(unittest.TestCase):
    def test_internal_scores_are_not_sent_to_owner(self):
        payload = notification.alert_payload(make_event())
        for key in notification.ALERT_INTERNAL_FIELDS:
            self.assertNotIn(key, payload)
        self.assertEqual(
            (payload["event_id"], payload["camera_id"], payload["category"], payload["clip_path"]),
            ("E001", "CAM-01", "쓰러짐", "output/clips/E001.mp4"),
        )
        self.assertEqual(payload["representative_images"], ["output/representative_images/E001_1.jpg"])

    def test_event_json_file_keeps_all_scores_for_audit(self):
        self.assertEqual(make_event().to_dict()["scores"]["쓰러짐"], 0.99)

    def test_webhook_body_has_no_scores(self):
        sent = {}

        class FakeResponse:
            status = 201
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"event_id": "E010"}'

        def fake_urlopen(request, timeout=None):
            sent["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        event = make_event()
        with patch.object(notification, "ALERT_WEBHOOK_URL", "http://backend/api/internal/events"), \
                patch.object(notification, "urlopen", fake_urlopen):
            self.assertTrue(notification.send_first_alert(event))
        self.assertNotIn("scores", sent["body"])
        self.assertNotIn("threshold", sent["body"])
        self.assertEqual(event.backend_event_id, "E010")


class SharedFolderPathTest(unittest.TestCase):
    """탐지 서비스가 다른 컴퓨터의 공유 폴더에 저장해도 백엔드가 읽을 상대 경로로 바뀌는지 확인한다."""

    def test_path_under_media_dir_becomes_output_relative(self):
        import tempfile
        from pathlib import Path
        from schemas import event_schema

        with tempfile.TemporaryDirectory() as share:
            clip = Path(share) / "clips" / "E001_10sec.mp4"
            with patch.object(event_schema, "MEDIA_DIR", Path(share)):
                self.assertEqual(event_schema._portable_output_uri(str(clip)), "output/clips/E001_10sec.mp4")
                # 저장 폴더 밖 경로는 기존 규칙(/output/ 이후만 남김)을 따른다
                self.assertEqual(event_schema._portable_output_uri(r"D:\old\output\clips\E002.mp4"), "output/clips/E002.mp4")


if __name__ == "__main__":
    unittest.main()
