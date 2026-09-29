"""사건 상태 전이와 변경 이력을 검증하는 단위 테스트입니다."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.repositories.event_repository import EventRepository
from app.routers.internal_events import create_internal_events_router, _store_representative_images
from app.schemas.event import Event, EventIngestRequest, EventVlm
from app.services.event_service import (
    EventService,
    InvalidEventStatusTransitionError,
)


class EventServiceTest(unittest.TestCase):
    """메모리 저장소를 사용해 사건 서비스의 업무 규칙을 검증합니다."""

    def setUp(self) -> None:
        """각 테스트가 독립된 미확인 사건에서 시작하도록 준비합니다."""

        repository = EventRepository()
        repository.seed(
            [
                Event(
                    event_id='E015',
                    store_id='S01',
                    camera_id='CAM-01',
                    source='behavior_model',
                    event_type='fall',
                    occurred_at='2026-03-14T21:12:00+09:00',
                    vlm=EventVlm(status='pending'),
                    status='unconfirmed',
                )
            ]
        )
        self.service = EventService(repository)

    def test_allows_confirmed_then_resolved(self) -> None:
        """미확인 사건은 확인 후 처리 완료로 순차 변경할 수 있습니다."""

        self.service.change_status('E015', 'confirmed', 'owner-01')
        event = self.service.change_status('E015', 'resolved', 'owner-01')

        self.assertEqual(event.status, 'resolved')
        history = self.service.history('E015')
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1].changed_at.utcoffset().total_seconds(), 9 * 60 * 60)

    def test_allows_false_alarm_from_unconfirmed(self) -> None:
        """미확인 사건은 오탐으로 바로 분류할 수 있습니다."""

        event = self.service.change_status('E015', 'false_alarm', 'owner-01')

        self.assertEqual(event.status, 'false_alarm')

    def test_rejects_unconfirmed_to_resolved(self) -> None:
        """미확인 사건을 처리 완료로 바로 변경할 수 없습니다."""

        with self.assertRaises(InvalidEventStatusTransitionError):
            self.service.change_status('E015', 'resolved', 'owner-01')

    def test_accepts_vlm_not_applicable_for_disconnect_events(self) -> None:
        """카메라 끊김 사건은 VLM을 적용하지 않는 상태를 허용합니다."""

        event = EventVlm(status='not_applicable')

        self.assertEqual(event.status, 'not_applicable')

    def test_ingests_detector_event_from_token_store_mapping(self) -> None:
        """탐지기 사건의 매장과 ID는 서버 설정과 저장소가 결정합니다."""

        router = create_internal_events_router(self.service)
        ingest = router.routes[0].endpoint
        payload = EventIngestRequest(
            event_id='E101',
            camera_id='CAM-02',
            event_type='fall',
            category='전도',
            confidence=0.91,
            scores={'fall': 0.91},
            occurred_at='2026-09-22T12:00:00+09:00',
        )
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}, clear=False):
            event = ingest(payload, 'Bearer test-token')
            duplicate = ingest(payload, 'Bearer test-token')

        self.assertEqual(event.store_id, 'S02')
        self.assertNotEqual(event.event_id, payload.event_id)
        self.assertEqual(event.event_type, 'fall')
        self.assertEqual(event.status, 'unconfirmed')
        self.assertNotEqual(duplicate.event_id, event.event_id)
        self.assertEqual(len(self.service.list_events('S02')), 2)

    def test_detector_images_keep_db_media_types_and_original_uris(self) -> None:
        """각 프레임 URI는 API 사건 ID와 무관하게 DB 미디어 행에 보존합니다."""

        with TemporaryDirectory() as temp_dir:
            image_root = Path(temp_dir) / 'representative_images'
            image_root.mkdir()
            file_names = ['E101_1.jpg', 'E101_2.jpg', 'E101_3.jpg']
            for file_name in file_names:
                (image_root / file_name).write_bytes(b'image')

            with patch.dict('os.environ', {'STOREOPS_OUTPUT_DIR': temp_dir}, clear=False):
                media_rows = _store_representative_images(
                    [f'output\\representative_images\\{name}' for name in file_names],
                    'E101',
                )

        self.assertEqual(
            [media.media_type for media in media_rows],
            ['frame_start', 'frame_middle', 'frame_end'],
        )
        self.assertEqual(
            [media.uri for media in media_rows],
            [f'representative_images/{name}' for name in file_names],
        )


if __name__ == '__main__':
    unittest.main()