from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.schemas.auth import AuthResponse, AuthUser, LoginRequest, SignUpRequest
from app.services.auth_service import (
    AuthService,
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidSessionError,
)


SESSION_HEADER_NAME = 'X-Session-ID'


def create_current_user_dependency(auth_service: AuthService):
    """X-Session-ID 헤더를 검증해 보호 API에서 현재 점주를 제공합니다."""

    def current_user(session_id: str | None = Header(default=None, alias=SESSION_HEADER_NAME)) -> AuthUser:
        try:
            return auth_service.get_current_user(session_id)
        except InvalidSessionError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='로그인이 필요합니다.',
            ) from error

    return current_user


def create_auth_router(auth_service: AuthService) -> APIRouter:
    router = APIRouter(prefix='/api/auth', tags=['auth'])
    current_user = create_current_user_dependency(auth_service)

    @router.post('/signup', response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
    def signup(payload: SignUpRequest) -> AuthResponse:
        try:
            return AuthResponse(user=auth_service.signup(payload))
        except DuplicateEmailError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='이미 가입된 이메일입니다.') from error

    @router.post('/login', response_model=AuthResponse)
    def login(payload: LoginRequest) -> AuthResponse:
        try:
            user, session_id, _ = auth_service.login(payload)
        except InvalidCredentialsError as error:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='이메일 또는 비밀번호가 올바르지 않습니다.') from error
        return AuthResponse(user=user, session_id=session_id)

    @router.post('/logout', status_code=status.HTTP_204_NO_CONTENT)
    def logout(session_id: str | None = Header(default=None, alias=SESSION_HEADER_NAME)) -> None:
        auth_service.logout(session_id)

    @router.get('/me', response_model=AuthUser)
    def me(user: AuthUser = Depends(current_user)) -> AuthUser:
        return user

    return router