"""사건 조회와 상태 전이에 필요한 업무 규칙을 제공합니다."""

from datetime import datetime, timedelta, timezone

from app.repositories.event_repository import EventRepository
from app.schemas.event import Event, EventStatus, EventStatusHistory


ALLOWED_STATUS_TRANSITIONS: dict[EventStatus, set[EventStatus]] = {
    # 점주가 확인할 수 있는 업무 흐름만 허용하고 상태를 되돌리지는 않습니다.
    'unconfirmed': {'confirmed', 'false_alarm'},
    'confirmed': {'resolved'},
    'resolved': set(),
    'false_alarm': set(),
}
KOREA_TIMEZONE = timezone(timedelta(hours=9))


class EventNotFoundError(Exception):
    """요청한 사건이 저장소에 없을 때 발생하는 예외입니다."""

    pass


class InvalidEventStatusTransitionError(Exception):
    """현재 상태에서 요청한 상태로 이동할 수 없을 때 발생합니다."""

    def __init__(self, current_status: EventStatus, requested_status: EventStatus) -> None:
        self.current_status = current_status
        self.requested_status = requested_status
        super().__init__(
            f'상태 전이를 허용하지 않습니다: {current_status} -> {requested_status}'
        )


class EventService:
    """사건 저장소와 API 사이에서 조회 및 상태 변경 규칙을 적용합니다."""

    def __init__(self, repository: EventRepository) -> None:
        self.repository = repository

    def list_events(
        self,
        store_id: str | None = None,
        status: EventStatus | None = None,
    ) -> list[Event]:
        """저장소에 필터 조건을 전달해 사건 목록을 조회합니다."""

        return self.repository.list_events(store_id=store_id, status=status)

    def ingest(self, event: Event) -> Event:
        """탐지 서비스 사건을 저장소에 넣고 재전송 시 기존 사건을 반환합니다."""

        return self.repository.create(event)

    def next_event_id(self) -> str:
        """저장소가 충돌 없이 발급한 사건 식별자를 반환합니다."""

        return self.repository.next_event_id()

    def get_event(self, event_id: str) -> Event:
        """사건을 조회하고 없으면 도메인 예외를 발생시킵니다."""

        event = self.repository.get(event_id)
        if event is None:
            raise EventNotFoundError(event_id)
        return event

    def change_status(
        self,
        event_id: str,
        status: EventStatus,
        changed_by: str,
    ) -> Event:
        """허용된 상태 전이를 검증한 뒤 변경 시각과 함께 저장합니다."""

        event = self.get_event(event_id)
        if status not in ALLOWED_STATUS_TRANSITIONS[event.status]:
            raise InvalidEventStatusTransitionError(event.status, status)
        return self.repository.update_status(
            event_id=event_id,
            status=status,
            changed_by=changed_by,
            changed_at=datetime.now(KOREA_TIMEZONE),
        )

    def history(self, event_id: str) -> list[EventStatusHistory]:
        """사건 존재 여부를 확인한 뒤 상태 변경 이력을 조회합니다."""

        self.get_event(event_id)
        return self.repository.history(event_id)
