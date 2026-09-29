from fastapi import APIRouter
from typing import Optional

from pydantic import BaseModel
from services.camera_runtime import camera_runtime

router = APIRouter(prefix="/path2", tags=["path2"])

class FrameRequest(BaseModel):
    camera_id: str
    timestamp: Optional[float] = None

class CheckRequest(BaseModel):
    camera_id: str
    timestamp: Optional[float] = None

class StreamRequest(BaseModel):
    camera_id: str
    stream_url: str


def monitor(camera_id):
    return camera_runtime.monitor(camera_id)

@router.post("/frame")
def frame(req: FrameRequest):
    m = monitor(req.camera_id)
    m.frame_received(req.timestamp)
    return m.state()

@router.post("/check")
def check(req: CheckRequest):
    m = monitor(req.camera_id)
    event = m.check(req.timestamp)
    return {"event": event.to_dict() if event else None, "state": m.state()}

@router.get("/state/{camera_id}")
def state(camera_id: str):
    return camera_runtime.stream_state(camera_id)

@router.post("/streams")
def start_stream(req: StreamRequest):
    """RTSP/HTTP 카메라 프레임 수신을 시작한다. Path 2 스케줄러가 끊김을 자동 감시한다."""
    return camera_runtime.start_stream(req.camera_id, req.stream_url)

@router.delete("/streams/{camera_id}")
def stop_stream(camera_id: str):
    return {"camera_id": camera_id, "stopped": camera_runtime.stop_stream(camera_id)}
