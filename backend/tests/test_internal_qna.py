"""질문 시점 VLM 해석 내부 API와 "VLM 해당 없음" 상태를 검증합니다."""

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.repositories.event_repository import EventRepository
from app.routers import internal_qna
from app.schemas.event import Event, EventClip, EventMediaReference, EventVlm, EventVlmUpdateRequest
from app.services.event_service import EventService
from app.services.vlm_describe import VlmUnavailableError


def make_event(event_id: str, store_id: str, images: list[str], source: str = 'behavior_model') -> Event:
    """대표 이미지 경로를 가진 행동 사건을 만듭니다."""

    media = [
        EventMediaReference(media_type=kind, uri=uri)
        for kind, uri in zip(('frame_start', 'frame_middle', 'frame_end'), images)
    ]
    return Event(
        event_id=event_id, store_id=store_id, camera_id='CAM-01', source=source,
        event_type='fall' if source == 'behavior_model' else 'camera_disconnect',
        occurred_at=datetime(2026, 10, 8, 5, 20, tzinfo=timezone.utc),
        clip=EventClip(uri='output/clips/x.mp4', length_sec=10), representative_images=media,
        vlm=EventVlm(status='not_applicable'), status='unconfirmed',
        gap_sec=60 if source == 'time_rule' else None, threshold_sec=60 if source == 'time_rule' else None,
    )


class InternalQnaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        out = Path(self.tmp.name)
        (out / 'representative_images').mkdir()
        for name in ('a.jpg', 'b.jpg', 'c.jpg'):
            (out / 'representative_images' / name).write_bytes(b'jpg')
        patcher = patch.dict(os.environ, {'STOREOPS_QNA_TOKEN': 'secret', 'STOREOPS_OUTPUT_DIR': str(out)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.service = EventService(EventRepository())
        self.service.ingest(make_event('E001', 'S01', [f'representative_images/{n}.jpg' for n in 'abc']))
        self.service.ingest(make_event('E002', 'S02', [f'representative_images/{n}.jpg' for n in 'abc']))
        self.service.ingest(make_event('E003', 'S01', ['representative_images/missing.jpg']))
        router = internal_qna.create_internal_qna_router(self.service)
        self.describe = next(route.endpoint for route in router.routes if route.path.endswith('/describe'))

    def call(self, ids, token='secret', store='S01'):
        body = internal_qna.DescribeRequest(store_id=store, event_ids=ids)
        return self.describe(body, token)['descriptions']

    def test_wrong_or_missing_token_is_rejected(self):
        for token in (None, '', 'wrong'):
            with self.assertRaises(HTTPException) as caught:
                self.call(['E001'], token)
            self.assertEqual(caught.exception.status_code, 403)

    def test_describes_only_own_store_events(self):
        with patch.object(internal_qna, 'describe_images', return_value={'text': '바닥에 사람이 누워 있음', 'model_name': 'm'}) as fake:
            rows = self.call(['E001', 'E002', 'E999'])
        self.assertEqual([r['status'] for r in rows], ['ok', 'not_found', 'not_found'])
        self.assertEqual(rows[0]['description'], '바닥에 사람이 누워 있음')
        self.assertEqual((rows[0]['category_ko'], rows[0]['camera_id']), ('쓰러짐', 'CAM-01'))
        self.assertEqual(rows[0]['occurred_at'], '2026-10-08T14:20:00+09:00')
        fake.assert_called_once()   # 다른 매장·없는 사건은 VLM을 부르지 않는다
        self.assertEqual(len(fake.call_args.args[0]), 3)

    def test_missing_image_files_are_reported(self):
        with patch.object(internal_qna, 'describe_images') as fake:
            rows = self.call(['E003'])
        self.assertEqual(rows[0]['status'], 'no_images')
        fake.assert_not_called()

    def test_vlm_failure_is_reported_not_hidden(self):
        with patch.object(internal_qna, 'describe_images', side_effect=VlmUnavailableError('서버 꺼짐')):
            rows = self.call(['E001'])
        self.assertEqual((rows[0]['status'], rows[0]['error']), ('vlm_unavailable', '서버 꺼짐'))

    def test_request_limits(self):
        with self.assertRaises(ValidationError):
            internal_qna.DescribeRequest(store_id='S01', event_ids=['E001'] * 6)
        with self.assertRaises(ValidationError):
            internal_qna.DescribeRequest(store_id='S01', event_ids=[])


class VlmStatusTests(unittest.TestCase):
    def test_not_applicable_is_accepted_without_description(self):
        request = EventVlmUpdateRequest(status='not_applicable')
        self.assertEqual((request.observation, request.owner_actions), (None, []))
        service = EventService(EventRepository())
        service.ingest(make_event('E001', 'S01', []).model_copy(update={'vlm': EventVlm(status='pending')}))
        updated = service.update_vlm('E001', EventVlm(status=request.status))
        self.assertEqual(updated.vlm.status, 'not_applicable')


if __name__ == '__main__':
    unittest.main()
