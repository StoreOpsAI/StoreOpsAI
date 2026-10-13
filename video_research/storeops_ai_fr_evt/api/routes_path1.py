import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config.config import EVENT_DIR
from pipelines.path1_behavior import Path1BehaviorPipeline

router = APIRouter(prefix="/path1", tags=["path1"])

class AnalyzeRequest(BaseModel):
    video_path: str
    camera_id: str = "CAM001"
    i3d_weight_path: Optional[str] = None
    debug: bool = False

@router.post("/analyze")
def analyze_video(req: AnalyzeRequest):
    if not Path(req.video_path).exists():
        raise HTTPException(status_code=404, detail=f"영상 파일을 찾을 수 없습니다: {req.video_path}")
    pipeline = Path1BehaviorPipeline(req.camera_id, req.i3d_weight_path, req.debug)
    events = pipeline.analyze(req.video_path)
    return {"issued_event_count": len(events), "events": [e.to_dict() for e in events]}

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
