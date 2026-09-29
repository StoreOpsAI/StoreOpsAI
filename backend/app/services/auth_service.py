import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.repositories.auth_repository import AuthRepository, SessionRecord, UserRecord
from app.schemas.auth import AuthUser, LoginRequest, SignUpRequest


KOREA_TIMEZONE = timezone(timedelta(hours=9))
SESSION_DURATION = timedelta(days=7)


class DuplicateEmailError(Exception):
    """이미 가입된 이메일로 회원가입을 시도했을 때 발생합니다."""


class InvalidCredentialsError(Exception):
    """이메일 또는 비밀번호가 일치하지 않을 때 발생합니다."""


class InvalidSessionError(Exception):
    """세션 ID가 없거나 만료·폐기되었을 때 발생합니다."""


def _hash_password(password: str, salt: bytes | None = None) -> str:
    password_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), password_salt, 310_000)
    return 'pbkdf2_sha256$310000$' + base64.b64encode(password_salt).decode() + '$' + base64.b64encode(digest).decode()


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        _, iterations, encoded_salt, encoded_digest = stored_hash.split('$')
        salt = base64.b64decode(encoded_salt)
        expected = base64.b64decode(encoded_digest)
        actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AuthService:
    """회원가입, 로그인, 세션 검증에 필요한 인증 규칙을 제공합니다."""

    def __init__(self, repository: AuthRepository) -> None:
        self.repository = repository

    def signup(self, payload: SignUpRequest) -> AuthUser:
        """새 매장과 점주 계정을 만들고 비밀번호는 해시로만 저장합니다."""
        store_id = self.repository.next_store_id()
        store_name = payload.store_name
        user = UserRecord(
            user_id=str(uuid4()),
            store_id=store_id,
            store_name=store_name,
            email=payload.email,
            password_hash=_hash_password(payload.password),
            display_name=payload.display_name,
        )
        try:
            self.repository.create_user(user)
        except ValueError as error:
            raise DuplicateEmailError from error
        return self._to_auth_user(user)

    def login(self, payload: LoginRequest) -> tuple[AuthUser, str, datetime]:
        """자격 증명을 검증하고 클라이언트에 전달할 세션 ID를 발급합니다."""

        user = self.repository.get_user_by_email(payload.email)
        if user is None or not user.is_active or not _verify_password(payload.password, user.password_hash):
            raise InvalidCredentialsError

        session_id = str(uuid4())
        now = datetime.now(KOREA_TIMEZONE)
        self.repository.save_session(
            SessionRecord(
                session_id=session_id,
                user_id=user.user_id,
                token_hash=_hash_session_token(session_id),
                expires_at=now + SESSION_DURATION,
                last_seen_at=now,
            )
        )
        return self._to_auth_user(user), session_id, now + SESSION_DURATION

    def logout(self, session_id: str | None) -> None:
        """세션 ID를 폐기해 이후 요청을 무효화합니다."""

        if session_id:
            self.repository.revoke_session(_hash_session_token(session_id), datetime.now(KOREA_TIMEZONE))

    def get_current_user(self, session_id: str | None) -> AuthUser:
        """세션 ID를 검증하고 최근 사용 시각을 갱신합니다."""

        if not session_id:
            raise InvalidSessionError
        session_hash = _hash_session_token(session_id)
        session = self.repository.get_session(session_hash)
        now = datetime.now(KOREA_TIMEZONE)
        if session is None or session.revoked_at is not None or session.expires_at <= now:
            raise InvalidSessionError
        self.repository.touch_session(session_hash, now)
        user = self.repository.get_user(session.user_id)
        if user is None or not user.is_active:
            raise InvalidSessionError
        return self._to_auth_user(user)

    @staticmethod
    def _to_auth_user(user: UserRecord) -> AuthUser:
        return AuthUser(
            user_id=user.user_id,
            store_id=user.store_id,
            store_name=user.store_name,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
        )
