import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
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
    def test_event_identity_and_timezone_survive_json_reload(self):
        """파일을 다시 읽어도 UUID와 절대 시각은 유지되고 새 사건은 다른 UUID를 갖습니다."""

        event = make_event()
        payload = notification.alert_payload(event)
        restored = EventCandidate.model_validate_json(json.dumps(event.to_dict()))
        self.assertEqual(restored.event_uid, event.event_uid)
        self.assertEqual(notification.alert_payload(restored)['event_uid'], payload['event_uid'])
        self.assertNotEqual(make_event().event_uid, event.event_uid)
        sent_at = datetime.fromisoformat(payload['created_at'].replace('Z', '+00:00'))
        self.assertIsNotNone(sent_at.utcoffset())
        self.assertLess(abs((datetime.now(timezone.utc) - sent_at).total_seconds()), 5)

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

    def test_remote_mode_sends_files_after_event_registration(self):
        """분리 배치에서는 로컬 경로를 숨기고 등록된 사건에 파일을 전송합니다."""

        sent = []

        class FakeResponse:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{}'

        def fake_urlopen(request, timeout=None):
            sent.append((request.full_url, request.data.read()))
            return FakeResponse()

        with TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / 'clip.mp4'
            image = Path(temp_dir) / 'frame.jpg'
            video.write_bytes(b'\x00\x00\x00\x18ftypisomtest')
            image.write_bytes(b'\xff\xd8test')
            event = make_event()
            event.backend_event_id = 'E010'
            event.event_video_path = str(video)
            event.representative_images = [str(image)]
            with patch.dict('os.environ', {'STOREOPS_MEDIA_UPLOAD_ENABLED': '1', 'EVENT_INGEST_TOKEN': 'test-token'}), \
                    patch.object(notification, 'ALERT_WEBHOOK_URL', 'https://backend/api/internal/events'), \
                    patch.object(notification, 'urlopen', fake_urlopen):
                self.assertIsNone(notification.alert_payload(event)['event_video_path'])
                self.assertEqual(notification.alert_payload(event)['representative_images'], [])
                self.assertTrue(notification.send_event_media(event))

        self.assertEqual([url.rsplit('/', 1)[-1] for url, _ in sent], ['clip', 'frame_start'])
        self.assertEqual(sent[0][1], b'\x00\x00\x00\x18ftypisomtest')


if __name__ == "__main__":
    unittest.main()
