import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from models.vlm_analyzer import VLMAnalyzer
from utils import vlm_gate
from utils.vlm_gate import VLMGate

GOOD = {"observation": "사람이 물건을 내려놓고 나감", "uncertain_points": "없음", "owner_actions": ["a", "b", "c"]}


class FakeVLM:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def verify_async(self, images, context):
        self.calls.append(context)
        f = Future()
        f.set_exception(self.error) if self.error else f.set_result(self.result)
        return f


class FakeEvents:
    def __init__(self):
        self.saved, self.updates = [], []

    def save(self, event):
        self.saved.append(event.status)

    def update(self, event_id, **changes):
        self.updates.append(changes)
        return {"event_id": event_id, **changes}


class NowExecutor:
    def submit(self, fn, *args):
        fn(*args)


def run(vlm, fail_open=True, media_first=False):
    event = SimpleNamespace(event_id="E1", camera_id="CAM001", category="쓰레기 투기", status="사건후보", alert_sent=False)
    published, sent, media = [], [], []
    with patch.object(vlm_gate, "send_first_alert", side_effect=lambda e: sent.append(e.event_id) or True), \
            patch.object(vlm_gate, "send_vlm_result", return_value=True), \
            patch.object(vlm_gate, "send_event_media", side_effect=lambda e: media.append(e.event_id)):
        gate = VLMGate(vlm, FakeEvents(), lambda kind, data: published.append(kind), NowExecutor(), fail_open)
        if media_first:
            gate.media_ready(event)
        gate.submit(event, [f"{i}.jpg" for i in range(18)], "정의", 4.0)
        if not media_first:
            gate.media_ready(event)
    return event, gate, published, sent, media


class VLMGateTest(unittest.TestCase):
    def test_confirmed_is_sent_with_media(self):
        event, gate, published, sent, media = run(FakeVLM({**GOOD, "confirmed": True}), media_first=True)
        self.assertEqual((sent, media, published), (["E1"], ["E1"], ["event.created"]))
        self.assertTrue(gate.is_sent("E1"))

    def test_mark_sent_uploads_media_without_vlm(self):   # 물건 우선 경보(VLM 확인 없이 보냄)도 영상은 올라가야 한다
        vlm, media = FakeVLM(), []
        with patch.object(vlm_gate, "send_event_media", side_effect=lambda e: media.append(e.event_id)):
            gate = VLMGate(vlm, FakeEvents(), lambda kind, data: None, NowExecutor())
            gate.mark_sent("E1")
            gate.media_ready(SimpleNamespace(event_id="E1"))
        self.assertEqual((media, vlm.calls), (["E1"], []))
        self.assertTrue(gate.is_sent("E1"))

    def test_rejected_is_not_sent(self):
        event, gate, published, sent, media = run(FakeVLM({**GOOD, "confirmed": False}))
        self.assertEqual((sent, media, published, event.status), ([], [], [], "VLM 기각"))
        self.assertFalse(gate.is_sent("E1"))

    def test_failure_fail_open_and_closed(self):
        _, _, _, sent, media = run(FakeVLM(error=RuntimeError("down")))
        self.assertEqual((sent, media), (["E1"], ["E1"]))
        _, _, _, sent, media = run(FakeVLM(error=RuntimeError("down")), fail_open=False)
        self.assertEqual((sent, media), ([], []))

    def test_not_configured_follows_fail_open(self):
        _, _, _, sent, _ = run(FakeVLM({"status": "VLM_NOT_CONFIGURED"}))
        self.assertEqual(sent, ["E1"])

    def test_vlm_receives_first_middle_and_last_clip_frames(self):
        vlm = FakeVLM({**GOOD, "confirmed": True})
        run(vlm)
        self.assertEqual((vlm.calls[0]["frame_count"], vlm.calls[0]["clip_seconds"]), (18, 4.0))
        seen = []
        VLMAnalyzer(transport=lambda imgs, ctx: seen.append(imgs) or {**GOOD, "confirmed": True}).verify([f"{i}.jpg" for i in range(18)], {})
        self.assertEqual(seen[0], ["0.jpg", "9.jpg", "17.jpg"])

    def test_save_all_images(self):
        import tempfile
        import numpy as np
        from utils.clip_manager import save_all_images
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(18)]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(save_all_images(frames, d, "E1")), 18)
            picked = save_all_images(frames, d, "E2", limit=5)
            self.assertEqual(len(picked), 5)

    def test_analyzer_verify_requires_bool(self):
        ok = VLMAnalyzer(transport=lambda imgs, ctx: {**GOOD, "confirmed": False})
        self.assertFalse(ok.verify(["a", "b", "c"], {"event_id": "E1"})["confirmed"])
        bad = VLMAnalyzer(transport=lambda imgs, ctx: {**GOOD, "confirmed": "yes"})
        with self.assertRaises(ValueError):
            bad.verify(["a", "b", "c"], {"event_id": "E1"})
        seen = []
        VLMAnalyzer(transport=lambda imgs, ctx: seen.append(ctx) or {**GOOD, "confirmed": True}).verify(["a", "b", "c"], {"category": "절도"})
        self.assertEqual(seen[0]["mode"], "verify")


if __name__ == "__main__":
    unittest.main()
