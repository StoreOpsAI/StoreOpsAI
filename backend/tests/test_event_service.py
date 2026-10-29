"""사건 상태 전이와 변경 이력을 검증하는 단위 테스트입니다."""

import unittest
import asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from fastapi import HTTPException
from starlette.requests import Request

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

    def test_event_list_limit_keeps_newest_events(self) -> None:
        """미리보기 개수 제한은 최신 사건부터 적용합니다."""

        repository = EventRepository()
        repository.seed([
            Event(
                event_id=f'E{index:03d}',
                store_id='S01',
                camera_id='CAM-01',
                source='behavior_model',
                event_type='fall',
                occurred_at=f'2026-03-14T21:{index:02d}:00+09:00',
                vlm=EventVlm(status='pending'),
                status='unconfirmed',
            )
            for index in (1, 2, 3)
        ])

        events = repository.list_events(store_id='S01', limit=2)

        self.assertEqual([event.event_id for event in events], ['E003', 'E002'])

    def test_postgres_event_list_passes_limit_to_query(self) -> None:
        """PostgreSQL 조회도 LIMIT 파라미터로 응답 행 수를 제한합니다."""

        repository = PostgresEventRepository()
        with patch('app.repositories.event_repository.connect') as connection:
            cursor = connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
            cursor.fetchall.return_value = []

            events = repository.list_events(store_id='S01', limit=5)

        query, parameters = cursor.execute.call_args.args
        self.assertIn('LIMIT %s', query)
        self.assertEqual(parameters[-1], 5)
        self.assertEqual(events, [])

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
        self.assertEqual(duplicate.event_id, event.event_id)
        self.assertEqual(len(self.service.list_events('S02')), 1)

    def test_uuid_retry_preserves_status_and_does_not_allocate_another_number(self) -> None:
        """재전송 시 점주가 변경한 상태와 사건 번호를 유지합니다."""

        ingest = create_internal_events_router(self.service).routes[0].endpoint
        payload = EventIngestRequest(event_uid=uuid4(), camera_id='CAM-02', event_type='fall')
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}):
            first = ingest(payload, 'Bearer test-token')
            self.service.change_status(first.event_id, 'confirmed', 'owner-01')
            duplicate = ingest(payload, 'Bearer test-token')
            another = ingest(payload.model_copy(update={'event_uid': uuid4()}), 'Bearer test-token')
        self.assertEqual(duplicate.event_id, first.event_id)
        self.assertEqual(duplicate.status, 'confirmed')
        self.assertEqual(another.event_id, 'E002')
        self.assertNotIn('event_uid', duplicate.model_dump())

    def test_concurrent_uuid_retries_create_one_event(self) -> None:
        """동시에 들어온 동일 사건은 한 건만 저장합니다."""

        ingest = create_internal_events_router(self.service).routes[0].endpoint
        payload = EventIngestRequest(event_uid=uuid4(), camera_id='CAM-02', event_type='fall')
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}):
            with ThreadPoolExecutor(max_workers=8) as executor:
                events = list(executor.map(lambda _: ingest(payload, 'Bearer test-token'), range(16)))
        self.assertEqual(len({event.event_id for event in events}), 1)
        self.assertEqual(len(self.service.list_events('S02')), 1)

    def test_uuid_is_scoped_to_store_and_camera(self) -> None:
        """다른 매장이나 카메라의 같은 식별자는 별도 사건으로 처리합니다."""

        ingest = create_internal_events_router(self.service).routes[0].endpoint
        payload = EventIngestRequest(event_uid=uuid4(), camera_id='CAM-02', event_type='fall')
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"first":"S02","second":"S03"}'}):
            first = ingest(payload, 'Bearer first')
            other_store = ingest(payload, 'Bearer second')
            other_camera = ingest(payload.model_copy(update={'camera_id': 'CAM-03'}), 'Bearer first')
        self.assertEqual(len({event.event_id for event in (first, other_store, other_camera)}), 3)

    def test_ingest_keeps_korean_local_time_and_converts_utc(self) -> None:
        """시간대가 명시된 한국 시각과 UTC 시각을 같은 사건 시각으로 저장합니다."""

        ingest = create_internal_events_router(self.service).routes[0].endpoint
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}):
            for value in ('2026-10-12T12:00:00+09:00', '2026-10-12T03:00:00Z'):
                payload = EventIngestRequest(event_uid=uuid4(), camera_id='CAM-02', event_type='fall', created_at=value)
                event = ingest(payload, 'Bearer test-token')
                self.assertEqual(event.occurred_at.isoformat(), '2026-10-12T12:00:00+09:00')

    def test_ingest_requires_an_identity(self) -> None:
        """UUID와 구형 원본 번호가 모두 없으면 재전송을 식별할 수 없어 거부합니다."""

        ingest = create_internal_events_router(self.service).routes[0].endpoint
        with patch.dict('os.environ', {'STOREOPS_AI_INGEST_TOKENS_JSON': '{"test-token":"S02"}'}):
            with self.assertRaises(HTTPException) as error:
                ingest(EventIngestRequest(camera_id='CAM-02', event_type='fall'), 'Bearer test-token')
        self.assertEqual(error.exception.status_code, 422)

    def test_postgres_retry_returns_existing_event_without_allocating_number(self) -> None:
        """DB 재전송 경로는 잠금 후 기존 사건을 반환하고 번호나 상태를 갱신하지 않습니다."""

        repository = PostgresEventRepository()
        event = self.service.get_event('E015').model_copy(update={'event_uid': uuid4()})
        with patch('app.repositories.event_repository.connect') as connection, \
                patch.object(repository, 'get', return_value=event):
            cursor = connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
            cursor.fetchone.return_value = ('E015',)
            self.assertEqual(repository.create(event.model_copy(update={'event_id': ''})), event)
        queries = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertEqual(len(queries), 2)
        self.assertIn('pg_advisory_xact_lock', queries[0])
        self.assertIn('event_uid = %s', queries[1])

    def test_postgres_new_event_stores_uuid_with_event_in_transaction(self) -> None:
        """DB 신규 사건 경로는 번호를 발급하고 UUID를 사건 행에 함께 저장합니다."""

        repository = PostgresEventRepository()
        event = self.service.get_event('E015').model_copy(update={'event_uid': uuid4(), 'event_id': ''})
        saved = event.model_copy(update={'event_id': 'E002'})
        with patch('app.repositories.event_repository.connect') as connection, \
                patch.object(repository, 'get', return_value=saved):
            cursor = connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
            cursor.fetchone.side_effect = [None, ('E002',)]
            self.assertEqual(repository.create(event), saved)
        insert = next(call for call in cursor.execute.call_args_list if 'INSERT INTO events' in call.args[0])
        self.assertEqual(insert.args[1][0], 'E002')
        self.assertEqual(insert.args[1][-1], event.event_uid)

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

    def test_uploads_clip_to_backend_and_rejects_another_store(self) -> None:
        """탐지 토큰의 매장을 확인하고 서버가 정한 경로에 MP4를 저장합니다."""

        upload = next(
            route.endpoint for route in create_internal_events_router(self.service).routes
            if route.path == '/api/internal/events/{event_id}/media/{media_type}'
        )
        body = b'\x00\x00\x00\x18ftypisomtest'

        def request(payload=body, content_type=b'video/mp4'):
            async def receive():
                return {'type': 'http.request', 'body': payload, 'more_body': False}

            return Request(
                {'type': 'http', 'method': 'PUT', 'headers': [(b'content-type', content_type)]},
                receive,
            )
        with TemporaryDirectory() as temp_dir, patch.dict('os.environ', {
            'STOREOPS_OUTPUT_DIR': temp_dir,
            'STOREOPS_AI_INGEST_TOKENS_JSON': '{"owner-token":"S01","other-token":"S02"}',
        }):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(upload('E015', 'clip', request(), 'Bearer other-token'))
            self.assertEqual(error.exception.status_code, 403)

            event = asyncio.run(upload('E015', 'clip', request(), 'Bearer owner-token'))
            self.assertEqual(event.clip.uri, 'clips/E015_10sec.mp4')
            self.assertEqual((Path(temp_dir) / event.clip.uri).read_bytes(), body)

            # 동일 사건의 재전송도 같은 미디어 항목을 갱신합니다.
            event = asyncio.run(upload('E015', 'clip', request(), 'Bearer owner-token'))
            self.assertEqual(event.clip.uri, 'clips/E015_10sec.mp4')

            image = b'\xff\xd8image'
            event = asyncio.run(upload('E015', 'frame_start', request(image, b'image/jpeg'), 'Bearer owner-token'))
            self.assertEqual(event.representative_images[0].uri, 'representative_images/E015_1.jpg')
            self.assertEqual((Path(temp_dir) / event.representative_images[0].uri).read_bytes(), image)

            with self.assertRaises(HTTPException) as error:
                asyncio.run(upload('E015', 'clip', request(b'not-an-mp4'), 'Bearer owner-token'))
            self.assertEqual(error.exception.status_code, 422)
            self.assertEqual((Path(temp_dir) / 'clips/E015_10sec.mp4').read_bytes(), body)


if __name__ == '__main__':
    unittest.main()
