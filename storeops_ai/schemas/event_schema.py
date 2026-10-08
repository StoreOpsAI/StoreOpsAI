from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def _portable_output_uri(value: str | None) -> str | None:
    """공유 output 경로는 실행 OS가 달라도 컨테이너가 읽을 수 있게 정규화합니다."""
    if not value:
        return value
    normalized = value.replace("\\", "/")
    marker = "/output/"
    if marker in normalized:
        return "output/" + normalized.split(marker, 1)[1]
    if normalized.startswith("output/"):
        return normalized
    return Path(normalized).as_posix()


class EventCandidate(BaseModel):
    """FR-EVT-06~16에서 저장하는 사건 후보.

    행동 사건: category/confidence/scores/영상/대표 이미지 사용
    카메라 끊김 사건: 영상과 점수는 비워 두고 disconnect 정보만 사용
    """
    event_id: str = Field(..., pattern=r"^E\d{3}$")
    backend_event_id: Optional[str] = None
    camera_id: str
    event_type: str
    category: str
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    risk_level: Optional[str] = None      # 위험도 단계(주의/경고/높음). A 단독일 때는 비움
    score_logit: Optional[float] = None   # 그 카테고리의 로그 오즈 점수(확률이 아님)
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
        data = self.model_dump(mode="json")
        data["clip_path"] = _portable_output_uri(data.get("clip_path"))
        data["event_video_path"] = _portable_output_uri(data.get("event_video_path"))
        data["representative_images"] = [
            _portable_output_uri(path) for path in data.get("representative_images", [])
        ]
        return data
