"""사건 API의 입력과 출력 형식을 정의하는 Pydantic 모델입니다."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


EventStatus = Literal['unconfirmed', 'confirmed', 'resolved', 'false_alarm']


class EventClip(BaseModel):
    """사건 영상의 위치와 길이입니다."""

    uri: str
    length_sec: int = Field(ge=0)


class EventVlm(BaseModel):
    """VLM 설명 처리 상태와 선택적 요약입니다."""

    status: Literal['pending', 'completed', 'failed', 'not_applicable']
    summary: str | None = None
    uncertain_points: str | None = None
    owner_actions: list[str] = Field(default_factory=list)
    error: str | None = None
    model_name: str | None = None


class EventVlmUpdateRequest(BaseModel):
    """비동기 VLM 작업 결과를 탐지 서비스에서 수신합니다."""

    status: Literal['completed', 'failed', 'not_applicable']
    observation: str | None = Field(default=None, max_length=4000)
    uncertain_points: str | None = Field(default=None, max_length=4000)
    owner_actions: list[str] = Field(default_factory=list, max_length=3)
    error: str | None = Field(default=None, max_length=2000)
    model_name: str | None = Field(default=None, max_length=120)


class EventMediaReference(BaseModel):
    """event_media의 한 프레임 행을 사건 응답에 전달합니다."""

    media_type: Literal['frame_start', 'frame_middle', 'frame_end']
    uri: str


class Event(BaseModel):
    """사건 목록과 상세 조회에서 공통으로 사용하는 사건 모델입니다."""

    event_id: str
    store_id: str
    camera_id: str
    source: Literal['behavior_model', 'time_rule']
    event_type: str
    occurred_at: datetime
    clip: EventClip | None = None
    representative_images: list[EventMediaReference] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    threshold: float | None = Field(default=None, ge=0, le=1)
    gap_sec: int | None = Field(default=None, ge=0)
    threshold_sec: int | None = Field(default=None, ge=0)
    vlm: EventVlm
    status: EventStatus
    data_label: Literal['real', 'synthetic', 'mock'] = 'mock'


class EventStatusChangeRequest(BaseModel):
    """점주가 사건 상태 변경을 요청할 때 사용하는 본문입니다."""

    status: EventStatus
    changed_by: str = Field(min_length=1)


class EventStatusHistory(BaseModel):
    """사건 상태가 언제 누구에 의해 변경되었는지 기록합니다."""

    event_id: str
    previous_status: EventStatus
    new_status: EventStatus
    changed_by: str
    changed_at: datetime


class EventListResponse(BaseModel):
    """사건 목록과 목록 개수를 함께 반환합니다."""

    events: list[Event]
    count: int


class EventIngestRequest(BaseModel):
    """탐지 서비스가 백엔드로 전달하는 사건 후보입니다."""

    event_id: str | None = Field(default=None, min_length=1, max_length=16)
    camera_id: str = Field(min_length=1, max_length=32)
    event_type: str = Field(min_length=1, max_length=32)
    category: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    scores: dict[str, float] = Field(default_factory=dict)
    occurred_at: datetime | None = None
    created_at: datetime | None = None
    clip_path: str | None = None
    event_video_path: str | None = None
    representative_images: list[str] = Field(default_factory=list)
    clip_start_time_sec: float | None = Field(default=None, ge=0)
    clip_end_time_sec: float | None = Field(default=None, ge=0)
    disconnect_seconds: float | None = Field(default=None, ge=0)
    disconnect_threshold_seconds: float | None = Field(default=None, ge=0)
    data_label: Literal['real', 'synthetic', 'mock'] = 'real'
