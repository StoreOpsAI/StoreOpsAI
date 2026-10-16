import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.clip_manager import save_frames


class ClipManagerTest(unittest.TestCase):
    def test_saved_mp4_uses_h264(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "event.mp4"
            frames = [np.full((64, 64, 3), value, dtype=np.uint8) for value in (20, 100, 180)]
            save_frames(frames, clip, 3)
            self.assertTrue(clip.is_file())
            capture = cv2.VideoCapture(str(clip))
            try:
                self.assertTrue(capture.isOpened())
                codec = int(capture.get(cv2.CAP_PROP_FOURCC))
                self.assertIn(codec.to_bytes(4, "little").decode("ascii"), ("avc1", "h264"))
            finally:
                capture.release()
            self.assertFalse(clip.with_suffix(".opencv.mp4").exists())
