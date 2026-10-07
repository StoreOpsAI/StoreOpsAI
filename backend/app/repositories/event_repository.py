"""메모리 또는 PostgreSQL에 사건과 상태 변경 이력을 보관하는 저장소입니다."""

from datetime import datetime

from psycopg.types.json import Jsonb

from app.schemas.event import Event, EventClip, EventMediaReference, EventStatus, EventStatusHistory, EventVlm
from app.database import connect


class EventRepository:
    """사건 데이터를 조회하고 상태 변경 이력을 기록합니다."""

    def __init__(self) -> None:
        self._events: dict[str, Event] = {}
        self._history: list[EventStatusHistory] = []
        self._next_event_number = 1

    def seed(self, events: list[Event]) -> None:
        """개발 및 시연용 초기 사건을 저장합니다."""

        for event in events:
            self._events[event.event_id] = event

    def create(self, event: Event) -> Event:
        """탐지 서비스 사건을 저장하고 동일 ID 재전송은 기존 값으로 처리합니다."""

        existing = self._events.get(event.event_id)
        if existing is not None:
            return existing
        self._events[event.event_id] = event
        return event

    def next_event_id(self) -> str:
        """화면에 표시하기 적당한 짧은 자리수로 사건 ID를 발급합니다."""

        event_id = f'E{self._next_event_number:03d}'
        self._next_event_number += 1
        return event_id

    def list_events(
        self,
        store_id: str | None = None,
        status: EventStatus | None = None,
    ) -> list[Event]:
        """매장과 상태로 필터링하고 발생 시각이 최신인 순서로 반환합니다."""

        events = list(self._events.values())
        if store_id is not None:
            events = [event for event in events if event.store_id == store_id]
        if status is not None:
            events = [event for event in events if event.status == status]
        return sorted(events, key=lambda event: event.occurred_at, reverse=True)

    def get(self, event_id: str) -> Event | None:
        """사건 번호로 사건을 조회합니다."""

        return self._events.get(event_id)

    def update_vlm(self, event_id: str, vlm: EventVlm) -> Event:
        """메모리 사건의 비동기 VLM 결과를 갱신합니다."""

        self._events[event_id] = self._events[event_id].model_copy(update={'vlm': vlm})
        return self._events[event_id]

    def update_status(
        self,
        event_id: str,
        status: EventStatus,
        changed_by: str,
        changed_at: datetime,
    ) -> Event:
        """사건 상태를 갱신하고 변경 전후 값을 이력으로 남깁니다."""

        event = self._events[event_id]
        previous_status = event.status
        self._events[event_id] = event.model_copy(update={'status': status})
        self._history.append(
            EventStatusHistory(
                event_id=event_id,
                previous_status=previous_status,
                new_status=status,
                changed_by=changed_by,
                changed_at=changed_at,
            )
        )
        return self._events[event_id]

    def history(self, event_id: str) -> list[EventStatusHistory]:
        """지정한 사건에 해당하는 상태 변경 이력만 반환합니다."""

        return [record for record in self._history if record.event_id == event_id]


class PostgresEventRepository:
    """DB 설계서의 events, event_media, vlm_results를 조회하는 저장소입니다."""

    def seed(self, events: list[Event]) -> None:
        """개발·시연용 사건과 필수 매장 기준정보를 중복 없이 저장합니다."""

        with connect() as connection, connection.cursor() as cursor:
            for event in events:
                cursor.execute(
                    "INSERT INTO stores (store_id, name) VALUES (%s, %s) ON CONFLICT (store_id) DO NOTHING",
                    (event.store_id, f'모의 매장 {event.store_id}'),
                )
                cursor.execute(
                    """
                    INSERT INTO cameras (camera_id, store_id, name, is_connected)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (camera_id) DO NOTHING
                    """,
                    (event.camera_id, event.store_id, event.camera_id, event.event_type != 'camera_disconnect'),
                )
                cursor.execute(
                    """
                    INSERT INTO events
                        (event_id, store_id, camera_id, source, event_type, occurred_at,
                        status, scores, threshold, gap_sec, threshold_sec, data_label)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (
                        event.event_id, event.store_id, event.camera_id, event.source, event.event_type,
                        event.occurred_at, event.status, Jsonb(event.scores), event.threshold,
                        event.gap_sec, event.threshold_sec, event.data_label,
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO vlm_results (event_id, status)
                    VALUES (%s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (event.event_id, event.vlm.status),
                )
                if event.clip is not None:
                    cursor.execute(
                        """
                        INSERT INTO event_media (event_id, media_type, uri, length_sec)
                        VALUES (%s, 'clip', %s, %s)
                        ON CONFLICT (event_id, media_type) DO NOTHING
                        """,
                        (event.event_id, event.clip.uri, event.clip.length_sec),
                    )
                self._save_representative_images(cursor, event)

    def create(self, event: Event) -> Event:
        """탐지 서비스 사건과 카메라·VLM 기준정보를 한 트랜잭션에 저장합니다."""

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO cameras (camera_id, store_id, name, is_connected)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (camera_id) DO NOTHING
                """,
                (event.camera_id, event.store_id, event.camera_id, event.event_type != 'camera_disconnect'),
            )
            cursor.execute(
                """
                INSERT INTO events
                    (event_id, store_id, camera_id, source, event_type, occurred_at,
                    status, scores, threshold, gap_sec, threshold_sec, data_label)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    event.event_id, event.store_id, event.camera_id, event.source, event.event_type,
                    event.occurred_at, event.status, Jsonb(event.scores), event.threshold,
                    event.gap_sec, event.threshold_sec, event.data_label,
                ),
            )
            cursor.execute(
                """
                INSERT INTO vlm_results (event_id, status)
                VALUES (%s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (event.event_id, event.vlm.status),
            )
            if event.clip is not None:
                cursor.execute(
                    """
                    INSERT INTO event_media (event_id, media_type, uri, length_sec)
                    VALUES (%s, 'clip', %s, %s)
                    ON CONFLICT (event_id, media_type) DO NOTHING
                    """,
                    (event.event_id, event.clip.uri, event.clip.length_sec),
                )
            self._save_representative_images(cursor, event)
        return self.get(event.event_id) or event

    @staticmethod
    def _save_representative_images(cursor, event: Event) -> None:
        for media in event.representative_images:
            cursor.execute(
                """
                INSERT INTO event_media (event_id, media_type, uri)
                VALUES (%s, %s, %s)
                ON CONFLICT (event_id, media_type) DO NOTHING
                """,
                (event.event_id, media.media_type, media.uri),
            )

    def next_event_id(self) -> str:
        """PostgreSQL 시퀀스에서 운영용 사건 ID를 발급합니다."""

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 'E' || LPAD(nextval('event_number_seq')::text, 3, '0')")
            return cursor.fetchone()[0]

    def list_events(self, store_id: str | None = None, status: EventStatus | None = None) -> list[Event]:
        """매장과 상태 조건이 없으면(NULL) 전체를 조회하고 최신 발생 순으로 반환합니다."""

        query = """
            SELECT e.event_id, e.store_id, e.camera_id, e.source, e.event_type,
                   e.occurred_at, e.scores, e.threshold, e.gap_sec, e.threshold_sec,
                   v.status, v.observed, m.uri, m.length_sec, e.status, e.data_label,
                   (
                       SELECT jsonb_agg(
                           jsonb_build_object('media_type', frame.media_type, 'uri', frame.uri)
                           ORDER BY CASE frame.media_type
                               WHEN 'frame_start' THEN 1
                               WHEN 'frame_middle' THEN 2
                               WHEN 'frame_end' THEN 3
                           END
                       )
                       FROM event_media frame
                       WHERE frame.event_id = e.event_id
                         AND frame.media_type IN ('frame_start', 'frame_middle', 'frame_end')
                   ) AS representative_images,
                   v.uncertain, v.owner_checks, v.error_message, v.model_name
            FROM events e
            LEFT JOIN vlm_results v ON v.event_id = e.event_id
            LEFT JOIN event_media m ON m.event_id = e.event_id AND m.media_type = 'clip'
            WHERE (%s::varchar IS NULL OR e.store_id = %s)
              AND (%s::varchar IS NULL OR e.status = %s)
            ORDER BY e.occurred_at DESC
        """
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, (store_id, store_id, status, status))
            rows = cursor.fetchall()
        return [self._event_from_row(row) for row in rows]

    def get(self, event_id: str) -> Event | None:
        """사건 번호로 사건을 조회합니다."""

        events = self._query_events(event_id)
        return events[0] if events else None

    def update_vlm(self, event_id: str, vlm: EventVlm) -> Event:
        """VLM 결과와 상태를 기존 결과 테이블에 저장합니다."""

        import json

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE vlm_results
                SET status = %s, model_name = %s, observed = %s, uncertain = %s, owner_checks = %s,
                    error_message = %s, completed_at = CASE WHEN %s = 'completed' THEN now() ELSE NULL END,
                    updated_at = now()
                WHERE event_id = %s
                """,
                (
                    vlm.status,
                    vlm.model_name,
                    vlm.summary,
                    vlm.uncertain_points,
                    json.dumps(vlm.owner_actions, ensure_ascii=False),
                    vlm.error,
                    vlm.status,
                    event_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(event_id)
        event = self.get(event_id)
        if event is None:
            raise RuntimeError(f'VLM 결과 저장 후 사건을 찾을 수 없습니다: {event_id}')
        return event

    def update_status(self, event_id: str, status: EventStatus, changed_by: str, changed_at: datetime) -> Event:
        """행 잠금으로 이전 상태를 읽은 뒤 상태를 갱신하고 이력을 함께 저장합니다."""

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute('SELECT status, store_id FROM events WHERE event_id = %s FOR UPDATE', (event_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(event_id)
            cursor.execute(
                'UPDATE events SET status = %s, updated_at = %s WHERE event_id = %s',
                (status, changed_at, event_id),
            )
            cursor.execute(
                """
                INSERT INTO event_status_history
                    (event_id, from_status, to_status, changed_by, changed_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (event_id, row[0], status, changed_by, changed_at),
            )
        updated_event = self.get(event_id)
        if updated_event is None:
            raise RuntimeError(f'상태 변경 후 사건을 찾을 수 없습니다: {event_id}')
        return updated_event

    def history(self, event_id: str) -> list[EventStatusHistory]:
        """지정한 사건의 상태 변경 이력을 변경 시각 순으로 반환합니다."""

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_id, from_status, to_status, changed_by, changed_at
                FROM event_status_history
                WHERE event_id = %s ORDER BY changed_at
                """,
                (event_id,),
            )
            rows = cursor.fetchall()
        return [EventStatusHistory(event_id=row[0], previous_status=row[1], new_status=row[2], changed_by=str(row[3]), changed_at=row[4]) for row in rows]

    def _query_events(self, event_id: str) -> list[Event]:
        """사건 조회에 필요한 컬럼을 관련 테이블과 조인해 가져옵니다."""

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT e.event_id, e.store_id, e.camera_id, e.source, e.event_type,
                       e.occurred_at, e.scores, e.threshold, e.gap_sec, e.threshold_sec,
                       v.status, v.observed, m.uri, m.length_sec, e.status, e.data_label,
                       (
                           SELECT jsonb_agg(
                               jsonb_build_object('media_type', frame.media_type, 'uri', frame.uri)
                               ORDER BY CASE frame.media_type
                                   WHEN 'frame_start' THEN 1
                                   WHEN 'frame_middle' THEN 2
                                   WHEN 'frame_end' THEN 3
                               END
                           )
                           FROM event_media frame
                           WHERE frame.event_id = e.event_id
                             AND frame.media_type IN ('frame_start', 'frame_middle', 'frame_end')
                       ) AS representative_images,
                       v.uncertain, v.owner_checks, v.error_message, v.model_name
                FROM events e
                LEFT JOIN vlm_results v ON v.event_id = e.event_id
                LEFT JOIN event_media m ON m.event_id = e.event_id AND m.media_type = 'clip'
                WHERE e.event_id = %s
                """,
                (event_id,),
            )
            rows = cursor.fetchall()
        return [self._event_from_row(row) for row in rows]

    @staticmethod
    def _event_from_row(row: tuple) -> Event:
        """SQL 조회 결과 한 행을 Event 모델로 변환합니다."""

        import json

        vlm_status = row[10] or 'pending'
        try:
            owner_actions = json.loads(row[18]) if row[18] else []
        except (TypeError, json.JSONDecodeError):
            owner_actions = [row[18]] if row[18] else []
        representative_images = [
            EventMediaReference.model_validate(media)
            for media in (row[16] or [])
        ]
        return Event(
            event_id=row[0], store_id=row[1], camera_id=row[2], source=row[3], event_type=row[4],
            occurred_at=row[5], scores=row[6] or {}, threshold=row[7], gap_sec=row[8], threshold_sec=row[9],
            vlm=EventVlm(
                status='completed' if vlm_status == 'completed' else vlm_status,
                summary=row[11],
                uncertain_points=row[17],
                owner_actions=owner_actions,
                error=row[19],
                model_name=row[20],
            ),
            clip=EventClip(uri=row[12], length_sec=row[13]) if row[12] else None,
            status=row[14], data_label=row[15], representative_images=representative_images,
        )
