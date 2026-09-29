from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventCandidate(BaseModel):
    """FR-EVT-06~16에서 저장하는 사건 후보.

    행동 사건: category/confidence/scores/영상/대표 이미지 사용
    카메라 끊김 사건: 영상과 점수는 비워 두고 disconnect 정보만 사용
    """
    event_id: str = Field(..., pattern=r"^E\d{3}$")
    camera_id: str
    event_type: str
    category: str
    confidence: Optional[float] = None
    scores: Dict[str, float] = Field(default_factory=dict)
    track_ids: List[int] = Field(default_factory=list)

    clip_start_frame: Optional[int] = None
    clip_end_frame: Optional[int] = None
    clip_start_time_sec: Optional[float] = None
    clip_end_time_sec: Optional[float] = None

    clip_path: Optional[str] = None
    event_video_path: Optional[str] = None
    representative_images: List[str] = Field(default_factory=list)
    event_video_complete: Optional[bool] = None

    disconnect_seconds: Optional[float] = None
    disconnect_threshold_seconds: Optional[float] = None

    status: str = "사건후보"
    alert_sent: bool = False
    alert_status: str = "PENDING"
    alert_error: Optional[str] = None
    vlm: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_dict(self):
        return self.model_dump(mode="json")
