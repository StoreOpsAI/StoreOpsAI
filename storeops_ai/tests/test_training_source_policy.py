import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.scripts.extract_multiclass import main as extract_main
from training.scripts.train_i3d import read_manifest


class TrainingSourcePolicyTest(unittest.TestCase):
    def write_manifest(self, directory, data_source, consent_confirmed):
        clip_dir = directory / "clip-001"
        clip_dir.mkdir()
        (clip_dir / "frame_000001.jpg").write_bytes(b"frame")
        with (directory / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["clip_folder", "label", "source_video", "data_source", "consent_confirmed"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "clip_folder": "clip-001",
                    "label": "fight",
                    "source_video": "acting.mp4",
                    "data_source": data_source,
                    "consent_confirmed": consent_confirmed,
                }
            )

    def test_training_rejects_unapproved_or_unverified_sources(self):
        for source, consent in [("nia2019", "false"), ("team_consented", "false")]:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.write_manifest(directory, source, consent)
                with self.assertRaises(ValueError):
                    read_manifest(directory)

    def test_training_accepts_aihub_and_consented_team_video(self):
        for source, consent in [("aihub_public", "false"), ("team_consented", "true")]:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.write_manifest(directory, source, consent)
                self.assertEqual(len(read_manifest(directory)), 1)

    def test_extractor_requires_confirmation_for_team_video(self):
        with patch.object(
            sys,
            "argv",
            ["extract_multiclass.py", "--videos", "videos", "--out", "out", "--data-source", "team_consented"],
        ):
            with self.assertRaises(SystemExit):
                extract_main()


if __name__ == "__main__":
    unittest.main()