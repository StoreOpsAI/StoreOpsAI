"""웹캠 프레임 전달 API의 매장 인증과 내부 요청을 검증합니다."""

import json
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.routers.camera import CameraFrameBatch, _forward_frame_batch, _ingest_token


class CameraRouterTests(unittest.TestCase):
    def test_ingest_token_is_selected_by_store(self):
        with patch.dict(
            os.environ,
            {"STOREOPS_AI_INGEST_TOKENS_JSON": '{"token-a":"store-a","token-b":"store-b"}'},
        ):
            self.assertEqual(_ingest_token("store-b"), "token-b")

    def test_missing_store_token_is_rejected(self):
        with patch.dict(os.environ, {"STOREOPS_AI_INGEST_TOKENS_JSON": "{}"}):
            with self.assertRaises(HTTPException) as caught:
                _ingest_token("store-a")
        self.assertEqual(caught.exception.status_code, 503)

    def test_frame_batch_is_forwarded_with_internal_bearer_token(self):
        payload = CameraFrameBatch(
            camera_id="CAM-01",
            captured_at=1792454400.0,
            frames=["encoded-frame"] * 12,
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b'{"issued_event_count":0,"events":[]}'

        with (
            patch.dict(os.environ, {"STOREOPS_AI_URL": "http://detector:8000/"}),
            patch("app.routers.camera.urlopen", return_value=FakeResponse()) as send_request,
        ):
            response = _forward_frame_batch(payload, "store-secret")

        request = send_request.call_args.args[0]
        self.assertEqual(request.full_url, "http://detector:8000/path1/analyze-frames")
        self.assertEqual(request.get_header("Authorization"), "Bearer store-secret")
        self.assertEqual(json.loads(request.data)["camera_id"], "CAM-01")
        self.assertEqual(response["issued_event_count"], 0)


if __name__ == "__main__":
    unittest.main()