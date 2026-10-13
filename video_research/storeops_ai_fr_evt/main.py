from fastapi import FastAPI
from api.routes_path1 import router as path1_router
from api.routes_path2 import router as path2_router

app = FastAPI(title="StoreOps AI - FR-EVT-01~16")
app.include_router(path1_router)
app.include_router(path2_router)

@app.get("/")
def root():
    return {"service": "StoreOps AI", "requirements": "FR-EVT-01~16", "mode": "feature-only-no-model-training"}
