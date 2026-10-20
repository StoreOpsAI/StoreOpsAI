import json
import base64
import binascii
import hmac
import os
import threading
import tempfile
from pathlib import Path
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from config.config import EVENT_DIR
from pipelines.path1_behavior import Path1BehaviorPipeline
from services.realtime import event_hub
from utils.event_manager import EventManager

router = APIRouter(prefix="/path1", tags=["path1"])

class AnalyzeRequest(BaseModel):
    video_path: str
    camera_id: str = "CAM001"
    i3d_weight_path: Optional[str] = None
    debug: bool = False

class FrameBatchRequest(BaseModel):
    camera_id: str = Field(default="CAM-01", min_length=1, max_length=32)
    captured_at: float = Field(gt=0)
    frames: list[str] = Field(min_length=12, max_length=48)

class EventStatusRequest(BaseModel):
    status: Literal["확인", "오탐"]


_pipeline_lock = threading.Lock()
_camera_pipelines = {}


def _get_camera_pipeline(camera_id):
    with _pipeline_lock:
        if camera_id not in _camera_pipelines:
            _camera_pipelines[camera_id] = (Path1BehaviorPipeline(camera_id), threading.Lock())
        return _camera_pipelines[camera_id]

@router.post("/analyze")
def analyze_video(req: AnalyzeRequest):
    if not Path(req.video_path).exists():
        raise HTTPException(status_code=404, detail=f"영상 파일을 찾을 수 없습니다: {req.video_path}")
    pipeline = Path1BehaviorPipeline(req.camera_id, req.i3d_weight_path, req.debug)
    events = pipeline.analyze(req.video_path)
    return {"issued_event_count": len(events), "events": [e.to_dict() for e in events]}


@router.post("/analyze-frames")
def analyze_frame_batch(req: FrameBatchRequest, authorization: str | None = Header(default=None)):
    """웹캠 프레임 묶음을 임시 영상으로 구성해 기존 행동 분석 파이프라인에 전달한다."""
    import cv2
    import numpy as np

    expected_token = os.getenv("EVENT_INGEST_TOKEN", "")
    received_token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not expected_token or not hmac.compare_digest(received_token, expected_token):
        raise HTTPException(status_code=401, detail="탐지 서비스 인증이 필요합니다.")

    decoded_frames = []
    total_bytes = 0
    for encoded_frame in req.frames:
        try:
            frame_bytes = base64.b64decode(encoded_frame, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(status_code=422, detail="프레임 인코딩이 올바르지 않습니다.") from error
        total_bytes += len(frame_bytes)
        if total_bytes > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="프레임 묶음 크기가 제한을 초과했습니다.")
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=422, detail="이미지 프레임을 읽을 수 없습니다.")
        decoded_frames.append(frame)

    height, width = decoded_frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary_file:
        video_path = Path(temporary_file.name)

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (width, height)
    )
    if not writer.isOpened():
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="웹캠 프레임 영상을 만들 수 없습니다.")
    try:
        for frame in decoded_frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    pipeline, camera_lock = _get_camera_pipeline(req.camera_id)
    try:
        with camera_lock:
            previous_event_count = len(pipeline.issued_events)
            pipeline.analyze(
                str(video_path), time_offset_sec=req.captured_at, save_tracked=False
            )
            events = pipeline.issued_events[previous_event_count:]
    finally:
        video_path.unlink(missing_ok=True)

    return {"issued_event_count": len(events), "events": [event.to_dict() for event in events]}

@router.get("/events")
def list_events():
    event_dir = Path(EVENT_DIR)
    event_dir.mkdir(parents=True, exist_ok=True)
    events = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(event_dir.glob("E*.json"), key=lambda p: p.stat().st_mtime, reverse=True)]
    return {"events": events}

@router.get("/events/{event_id}")
def get_event(event_id: str):
    path = Path(EVENT_DIR) / f"{event_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="해당 event_id를 찾을 수 없습니다.")
    return json.loads(path.read_text(encoding="utf-8"))

@router.patch("/events/{event_id}/status")
def update_event_status(event_id: str, req: EventStatusRequest):
    """점주가 대시보드에서 확인 또는 오탐으로 사건 상태를 변경한다."""
    path = Path(EVENT_DIR) / f"{event_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="해당 event_id를 찾을 수 없습니다.")
    event = EventManager(EVENT_DIR).update(event_id, status=req.status)
    event_hub.publish("event.updated", event)
    return event

@router.websocket("/ws/events")
async def event_websocket(websocket: WebSocket):
    """React 대시보드 실시간 사건 이벤트 채널."""
    await event_hub.connect(websocket)
    try:
        await websocket.send_json({"type": "connection.ready"})
        while True:
            # 클라이언트 ping/향후 양방향 명령을 수신한다.
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_hub.disconnect(websocket)
    except Exception:
        event_hub.disconnect(websocket)

@router.get("/events/{event_id}/clip")
def get_event_clip(event_id: str):
    event = get_event(event_id)
    clip = event.get("event_video_path") or event.get("clip_path")
    if not clip or not Path(clip).exists():
        raise HTTPException(status_code=404, detail="이벤트 영상이 없습니다.")
    return FileResponse(clip, media_type="video/mp4", filename=Path(clip).name)

@router.get("/events/{event_id}/images/{index}")
def get_event_image(event_id: str, index: int):
    event = get_event(event_id)
    images = event.get("representative_images", [])
    if index < 0 or index >= len(images):
        raise HTTPException(status_code=404, detail="이미지 인덱스가 잘못되었습니다.")
    p = Path(images[index])
    if not p.exists():
        raise HTTPException(status_code=404, detail="이미지 파일이 없습니다.")
    return FileResponse(p, media_type="image/jpeg", filename=p.name)
