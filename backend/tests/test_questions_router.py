"""인증된 점주의 매장 기록만 질문 서비스에 전달되는지 검증합니다."""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.routers import questions
from app.schemas.auth import AuthUser


class QuestionRouterTests(unittest.TestCase):
    def setUp(self):
        self.user = AuthUser(user_id='owner-a', store_id='S01', store_name='매장',
                             email='owner@example.com', display_name='점주')
        self.service = Mock()
        self.service.list_events.return_value = [
            SimpleNamespace(event_id='E001', store_id='S01', camera_id='CAM-01',
                            source='behavior_model', event_type='fall',
                            occurred_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
                            clip=SimpleNamespace(uri='clips/E001.mp4', length_sec=10)),
            SimpleNamespace(event_id='E999', store_id='S02', camera_id='CAM-02',
                            source='behavior_model', event_type='fall',
                            occurred_at=datetime(2026, 10, 1, tzinfo=timezone.utc), clip=None),
        ]
        router = questions.create_questions_router(self.service, lambda: self.user)
        self.ask = next(route.endpoint for route in router.routes if route.path == '/api/ask')

    def test_forwards_only_authenticated_store_data(self):
        drafts = [{'draft_id': 'D001', 'store_id': 'S01', 'product_id': 'P001', 'target_date': '2026-10-02'},
                  {'draft_id': 'D999', 'store_id': 'S02', 'product_id': 'P999', 'target_date': '2026-10-02'}]
        with (patch.object(questions, '_demand_token', return_value='private-token'),
              patch.object(questions, '_forward', return_value=drafts),
              patch.object(questions, '_call_qna', return_value={'answer': '확인'}) as call):
            result = self.ask(questions.QuestionRequest(question='어제 사건?'), self.user)
        self.assertEqual(result['answer'], '확인')
        self.service.list_events.assert_called_once_with(store_id='S01')
        payload = call.call_args.args[1]
        self.assertEqual((payload['owner_id'], payload['store_id']), ('owner-a', 'S01'))
        self.assertEqual([event['event_id'] for event in payload['events']], ['E001'])
        self.assertEqual([draft['draft_id'] for draft in payload['order_drafts']], ['D001'])
        self.assertEqual(payload['events'][0]['occurred_at'], '2026-10-01T09:00:00+09:00')

    def test_source_failure_is_not_reported_as_zero_records(self):
        self.service.list_events.side_effect = OSError('DB unavailable')
        with (patch.object(questions, '_demand_token', side_effect=ValueError('unavailable')),
              patch.object(questions, '_call_qna', return_value={}) as call):
            self.ask(questions.QuestionRequest(question='어제 사건?'), self.user)
        payload = call.call_args.args[1]
        self.assertEqual(set(payload['errors']), {'event', 'order_draft'})

    def test_missing_session_rejected_by_existing_dependency(self):
        from fastapi import FastAPI, HTTPException
        from app.routers.auth import create_current_user_dependency

        auth_service = Mock()
        from app.services.auth_service import InvalidSessionError
        auth_service.get_current_user.side_effect = InvalidSessionError()

        app = FastAPI()
        current_user = create_current_user_dependency(auth_service)
        app.include_router(questions.create_questions_router(self.service, current_user))
        self.assertTrue(any(route.path == '/api/ask/{question_session_id}/log' for route in app.routes))
        with self.assertRaises(HTTPException) as error:
            current_user(None)
        self.assertEqual(error.exception.status_code, 401)


if __name__ == '__main__':
    unittest.main()