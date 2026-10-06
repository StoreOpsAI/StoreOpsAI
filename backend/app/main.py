"""StoreOps AI FastAPI 애플리케이션의 진입점입니다."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

from app.repositories.auth_repository import AuthRepository
from app.repositories.auth_repository import PostgresAuthRepository
from app.repositories.event_repository import EventRepository
from app.repositories.event_repository import PostgresEventRepository
from app.database import get_database_url
from app.routers.auth import create_auth_router, create_current_user_dependency
from app.routers.events import create_events_router
from app.routers.demand import create_demand_router
from app.routers.internal_events import create_internal_events_router
from app.routers.questions import create_questions_router
from app.services.auth_service import AuthService
from app.services.event_service import EventService

app = FastAPI(title='StoreOpsAI API')
frontend_origin = os.getenv('FRONTEND_ORIGIN', 'http://localhost:5173')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class EchoRequest(BaseModel):
    """백엔드 연결 확인용 에코 요청 본문입니다."""

    message: str


if get_database_url():
    event_repository = PostgresEventRepository()
    auth_repository = PostgresAuthRepository()
else:
    event_repository = EventRepository()
    auth_repository = AuthRepository()
event_service = EventService(event_repository)
auth_service = AuthService(auth_repository)
current_user = create_current_user_dependency(auth_service)
app.include_router(create_events_router(event_service, current_user))
app.include_router(create_auth_router(auth_service))
app.include_router(create_demand_router(current_user))
app.include_router(create_internal_events_router(event_service))
app.include_router(create_questions_router(event_service, current_user))


@app.get('/api/health')
def healthcheck():
    """프런트엔드가 백엔드 연결 상태를 확인할 수 있도록 응답합니다."""

    return {'service': 'storeopsai-api', 'status': 'ok'}


@app.post('/api/echo')
def echo(payload: EchoRequest):
    """전달받은 메시지를 그대로 되돌려 연결 상태를 확인합니다."""

    return {'message': f'Backend received: {payload.message}'}