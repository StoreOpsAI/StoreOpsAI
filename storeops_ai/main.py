from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_path1 import router as path1_router
from api.routes_path2 import router as path2_router
from services.camera_runtime import camera_runtime
from services.input_watcher import input_video_watcher
from services.realtime import event_hub
from config.config import CORS_ALLOW_ORIGINS

app = FastAPI(title="StoreOps AI - FR-EVT-01~16")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(path1_router)
app.include_router(path2_router)

@app.on_event("startup")
def start_camera_runtime():
    event_hub.start()
    camera_runtime.start()
    input_video_watcher.start()

@app.on_event("shutdown")
def stop_camera_runtime():
    camera_runtime.shutdown()
    input_video_watcher.shutdown()

@app.get("/")
def root():
    return {"service": "StoreOps AI", "requirements": "FR-EVT-01~16", "mode": "feature-only-no-model-training"}
