"""인증 API의 입력과 출력 형식을 정의하는 Pydantic 모델입니다."""

from pydantic import BaseModel, Field, field_validator


class SignUpRequest(BaseModel):
    """회원가입 시 매장 정보와 점주 계정을 함께 등록하는 요청 본문입니다."""

    store_name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=120)

    @field_validator('email')
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if '@' not in normalized or normalized.startswith('@') or normalized.endswith('@'):
            raise ValueError('올바른 이메일 주소를 입력하세요.')
        return normalized


class LoginRequest(BaseModel):
    """로그인 요청 본문입니다."""

    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1)

    @field_validator('email')
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class AuthUser(BaseModel):
    """응답에 노출할 점주 계정 정보이며 비밀번호는 포함하지 않습니다."""

    user_id: str
    store_id: str
    store_name: str
    email: str
    display_name: str
    role: str = 'owner'


class AuthResponse(BaseModel):
    """회원가입·로그인 성공 시 반환하는 사용자 정보와 세션 ID입니다."""

    user: AuthUser
    session_id: str | None = None
