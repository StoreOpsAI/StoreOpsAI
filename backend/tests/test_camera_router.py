"""웹캠 프레임 전달 API의 매장 인증과 내부 요청을 검증합니다."""

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.routers.camera import _fetch_live_frame, _ingest_token


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

    def test_live_frame_is_fetched_with_store_token(self):
        class FakeResponse:
            status = 200
            headers = {"X-Frame-Seq": "7"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b"jpeg-bytes"

        with (
            patch.dict(os.environ, {"STOREOPS_AI_LIVE_URL": "http://host.docker.internal:8101/"}),
            patch("app.routers.camera.urlopen", return_value=FakeResponse()) as send_request,
        ):
            response = _fetch_live_frame(6, "store-secret")

        request = send_request.call_args.args[0]
        self.assertEqual(request.full_url, "http://host.docker.internal:8101/live/frame?after=6")
        self.assertEqual(request.get_header("Authorization"), "Bearer store-secret")
        self.assertEqual(response.media_type, "image/jpeg")
        self.assertEqual(response.headers["X-Frame-Seq"], "7")
        self.assertEqual(response.body, b"jpeg-bytes")

    def test_live_frame_without_new_image_returns_204(self):
        class Empty:
            status = 204
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        with patch("app.routers.camera.urlopen", return_value=Empty()):
            self.assertEqual(_fetch_live_frame(3, "t").status_code, 204)

if __name__ == "__main__":
    unittest.main()