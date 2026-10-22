import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.config import load_env_file


class EnvFileTest(unittest.TestCase):
    def write_env(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_values_and_skips_comments_and_empty_values(self):
        path = self.write_env(
            "# 주석\n\nEVENT_INGEST_TOKEN=abc123\n"
            'ALERT_WEBHOOK_URL="http://192.168.0.10:8000/api/internal/events"\n'
            "VLM_RESULT_WEBHOOK_URL=http://b:8000/api/internal/events/{event_id}/vlm\n"
            "QWEN_VLM_API_KEY=\nBROKEN_LINE\n"
        )
        with patch.dict(os.environ, {}, clear=True):
            load_env_file(path)
            self.assertEqual(os.environ["EVENT_INGEST_TOKEN"], "abc123")
            self.assertEqual(os.environ["ALERT_WEBHOOK_URL"], "http://192.168.0.10:8000/api/internal/events")
            self.assertIn("{event_id}", os.environ["VLM_RESULT_WEBHOOK_URL"])
            self.assertNotIn("QWEN_VLM_API_KEY", os.environ)   # 빈 값은 설정하지 않아 기본값이 쓰인다

    def test_existing_environment_wins_over_file(self):
        path = self.write_env("EVENT_INGEST_TOKEN=from-file\n")
        with patch.dict(os.environ, {"EVENT_INGEST_TOKEN": "from-shell"}, clear=True):
            load_env_file(path)
            self.assertEqual(os.environ["EVENT_INGEST_TOKEN"], "from-shell")

    def test_missing_file_is_ignored(self):
        load_env_file(Path(tempfile.gettempdir()) / "no-such-dir" / ".env")


if __name__ == "__main__":
    unittest.main()
