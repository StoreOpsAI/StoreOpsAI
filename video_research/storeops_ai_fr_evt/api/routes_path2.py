from fastapi import APIRouter
from pydantic import BaseModel
from pipelines.path2_connection import CameraConnectionMonitor

router = APIRouter(prefix="/path2", tags=["path2"])
_monitors = {}

class FrameRequest(BaseModel):
    camera_id: str
    timestamp: float | None = None

class CheckRequest(BaseModel):
    camera_id: str
    timestamp: float | None = None


def monitor(camera_id):
    if camera_id not in _monitors:
        _monitors[camera_id] = CameraConnectionMonitor(camera_id)
    return _monitors[camera_id]

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
    return monitor(camera_id).state()
