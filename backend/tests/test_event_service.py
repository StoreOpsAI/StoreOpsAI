"""사건 상태 전이와 변경 이력을 검증하는 단위 테스트입니다."""

import unittest
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from fastapi import HTTPException

from app.repositories.event_repository import EventRepository, PostgresEventRepository
from app.routers.internal_events import create_internal_events_router, _store_representative_images
from app.schemas.event import Event, EventIngestRequest, EventVlm, EventVlmUpdateRequest
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
            threshold=0.60,
            scores={'fall': 0.91},
            occurred_at='2026-09-22T12:00:00+09:00',
        )
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}, clear=False):
            event = ingest(payload, 'Bearer test-token')
            duplicate = ingest(payload, 'Bearer test-token')

        self.assertEqual(event.store_id, 'S02')
        self.assertNotEqual(event.event_id, payload.event_id)
        self.assertEqual(event.event_type, 'fall')
        self.assertEqual(event.threshold, 0.60)
        self.assertEqual(event.status, 'unconfirmed')
        self.assertNotEqual(duplicate.event_id, event.event_id)
        self.assertEqual(len(self.service.list_events('S02')), 2)

    def test_maps_live_webcam_categories_to_dashboard_event_types(self) -> None:
        """실시간 웹캠 모델의 한글 카테고리를 사건 화면 종류로 변환합니다."""

        router = create_internal_events_router(self.service)
        ingest = router.routes[0].endpoint
        categories = {
            '쓰러짐': 'fall',
            '싸움': 'fight',
            '파손': 'vandalism',
            '쓰레기 투기': 'littering',
        }
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}, clear=False):
            for index, (category, expected_type) in enumerate(categories.items(), start=1):
                with self.subTest(category=category):
                    payload = EventIngestRequest(
                        event_id=f'E{100 + index}',
                        camera_id='CAM-01',
                        event_type='행동',
                        category=category,
                        confidence=0.8,
                        threshold=0.6,
                        scores={category: 0.8},
                    )
                    event = ingest(payload, 'Bearer test-token')

                    self.assertEqual(event.event_type, expected_type)

    def test_camera_reconnect_is_time_rule_without_vlm(self) -> None:
        """카메라 재연결은 행동 분석/VLM 없이 시간 규칙 사건으로 저장합니다."""

        router = create_internal_events_router(self.service)
        ingest = router.routes[0].endpoint
        payload = EventIngestRequest(
            event_id='E102',
            camera_id='CAM-02',
            event_type='카메라재연결',
            category='카메라재연결',
            disconnect_seconds=90,
            disconnect_threshold_seconds=60,
        )
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}, clear=False):
            event = ingest(payload, 'Bearer test-token')

        self.assertEqual(event.event_type, 'camera_reconnected')
        self.assertEqual(event.source, 'time_rule')
        self.assertEqual(event.vlm.status, 'not_applicable')
        self.assertEqual(event.gap_sec, 90)

    def test_updates_behavior_event_with_structured_vlm_result(self) -> None:
        """비동기 VLM 결과는 인증된 행동 사건에 구조화해 저장합니다."""

        router = create_internal_events_router(self.service)
        update = next(
            route.endpoint for route in router.routes
            if route.path == '/api/internal/events/{event_id}/vlm'
        )
        payload = EventVlmUpdateRequest(
            status='completed',
            observation='사람이 바닥에 앉아 있습니다.',
            uncertain_points='넘어지는 장면은 확인되지 않습니다.',
            owner_actions=['현장 확인', '카메라 시야 점검', '필요 시 직원 확인'],
            model_name='Qwen/Qwen3-VL-8B-Instruct',
        )
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S01"}'}, clear=False):
            updated = update('E015', payload, 'Bearer test-token')

        self.assertEqual(updated.vlm.status, 'completed')
        self.assertEqual(updated.vlm.summary, '사람이 바닥에 앉아 있습니다.')
        self.assertEqual(len(updated.vlm.owner_actions), 3)
        self.assertEqual(updated.vlm.model_name, 'Qwen/Qwen3-VL-8B-Instruct')

    def test_rejects_incomplete_vlm_result(self) -> None:
        """관찰·불확실성·점주 확인 3개 항목이 없으면 완료 결과를 거부합니다."""

        router = create_internal_events_router(self.service)
        update = next(
            route.endpoint for route in router.routes
            if route.path == '/api/internal/events/{event_id}/vlm'
        )
        payload = EventVlmUpdateRequest(
            status='completed', observation='관찰', uncertain_points='불확실', owner_actions=['한 항목'],
        )
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S01"}'}, clear=False):
            with self.assertRaises(HTTPException) as error:
                update('E015', payload, 'Bearer test-token')

        self.assertEqual(error.exception.status_code, 422)

    def test_postgres_event_row_restores_structured_vlm_fields(self) -> None:
        """DB 조회 행을 점주 화면에서 읽을 구조화 VLM 결과로 복원합니다."""

        event = PostgresEventRepository._event_from_row(
            (
                'E105', 'S01', 'CAM-01', 'behavior_model', 'fall',
                datetime.now(timezone.utc), {'fall': 0.8}, 0.6, None, None,
                'completed', '사람이 바닥에 앉아 있습니다.', 'output/clips/E105_10sec.mp4', 10,
                'unconfirmed', 'real', [], '넘어지는 장면은 보이지 않습니다.',
                json.dumps(['현장 확인', '직원 확인', '카메라 확인'], ensure_ascii=False), None,
                'Qwen/Qwen3-VL-8B-Instruct',
            )
        )

        self.assertEqual(event.vlm.uncertain_points, '넘어지는 장면은 보이지 않습니다.')
        self.assertEqual(event.vlm.owner_actions, ['현장 확인', '직원 확인', '카메라 확인'])
        self.assertEqual(event.vlm.model_name, 'Qwen/Qwen3-VL-8B-Instruct')

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