"""웹캠 프레임 분석 API의 내부 토큰 검증을 확인합니다."""

import base64
import os
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
from fastapi import HTTPException
import numpy as np

from api import routes_path1
from api.routes_path1 import FrameBatchRequest, analyze_frame_batch


class FrameIngestTests(unittest.TestCase):
    def test_missing_token_is_rejected_before_frame_decoding(self):
        request = FrameBatchRequest(
            camera_id="CAM-01",
            captured_at=1792454400.0,
            frames=["not-base64"] * 12,
        )

        with patch.dict(os.environ, {"EVENT_INGEST_TOKEN": "expected-secret"}):
            with self.assertRaises(HTTPException) as caught:
                analyze_frame_batch(request, authorization=None)

        self.assertEqual(caught.exception.status_code, 401)

    def test_wrong_token_is_rejected(self):
        request = FrameBatchRequest(
            camera_id="CAM-01",
            captured_at=1792454400.0,
            frames=["not-base64"] * 12,
        )

        with patch.dict(os.environ, {"EVENT_INGEST_TOKEN": "expected-secret"}):
            with self.assertRaises(HTTPException) as caught:
                analyze_frame_batch(request, "Bearer wrong-secret")

        self.assertEqual(caught.exception.status_code, 401)

    def test_valid_frames_are_sent_to_existing_pipeline(self):
        success, image = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertTrue(success)
        encoded_frame = base64.b64encode(image).decode("ascii")
        request = FrameBatchRequest(
            camera_id="CAM-01",
            captured_at=1792454400.0,
            frames=[encoded_frame] * 30,
        )

        class FakePipeline:
            issued_events = []

            def analyze(self, video_path, **kwargs):
                self.video_path = Path(video_path)
                self.kwargs = kwargs

        pipeline = FakePipeline()
        with (
            patch.dict(os.environ, {"EVENT_INGEST_TOKEN": "expected-secret"}),
            patch.object(
                routes_path1,
                "_get_camera_pipeline",
                return_value=(pipeline, threading.Lock()),
            ),
        ):
            response = analyze_frame_batch(request, "Bearer expected-secret")

        self.assertEqual(response["issued_event_count"], 0)
        self.assertEqual(pipeline.kwargs, {"time_offset_sec": request.captured_at, "save_tracked": False})
        self.assertFalse(pipeline.video_path.exists())


if __name__ == "__main__":
    unittest.main()