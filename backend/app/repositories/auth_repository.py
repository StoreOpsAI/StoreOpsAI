"""현재는 메모리 또는 PostgreSQL에 점주 계정과 세션을 보관하는 인증 저장소입니다."""

from dataclasses import dataclass
from datetime import datetime

from app.database import connect


@dataclass
class UserRecord:
    """점주 계정 한 명의 정보와 비밀번호 해시입니다."""

    user_id: str
    store_id: str
    store_name: str
    email: str
    password_hash: str
    display_name: str
    role: str = 'owner'
    is_active: bool = True


@dataclass
class SessionRecord:
    """로그인 세션 하나의 토큰 해시와 만료·폐기 시각입니다."""

    session_id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    revoked_at: datetime | None = None
    last_seen_at: datetime | None = None


class AuthRepository:
    """DATABASE_URL이 없을 때 사용하는 메모리 기반 인증 저장소입니다."""

    def __init__(self) -> None:
        self._users_by_email: dict[str, UserRecord] = {}
        self._users_by_id: dict[str, UserRecord] = {}
        self._sessions_by_token_hash: dict[str, SessionRecord] = {}
        self._next_store_number = 1

    def create_user(self, user: UserRecord) -> UserRecord:
        # 이메일은 로그인 식별자이므로 중복 가입을 저장 전에 차단합니다.
        if user.email in self._users_by_email:
            raise ValueError('이미 가입된 이메일입니다.')
        self._users_by_email[user.email] = user
        self._users_by_id[user.user_id] = user
        return user

    def next_store_id(self) -> str:
        store_id = f'S{self._next_store_number:02d}'
        self._next_store_number += 1
        return store_id

    def get_user_by_email(self, email: str) -> UserRecord | None:
        return self._users_by_email.get(email)

    def get_user(self, user_id: str) -> UserRecord | None:
        return self._users_by_id.get(user_id)

    def save_session(self, session: SessionRecord) -> None:
        self._sessions_by_token_hash[session.token_hash] = session

    def get_session(self, token_hash: str) -> SessionRecord | None:
        return self._sessions_by_token_hash.get(token_hash)

    def revoke_session(self, token_hash: str, revoked_at: datetime) -> None:
        session = self._sessions_by_token_hash.get(token_hash)
        if session is not None:
            session.revoked_at = revoked_at

    def touch_session(self, token_hash: str, seen_at: datetime) -> None:
        session = self._sessions_by_token_hash.get(token_hash)
        if session is not None:
            session.last_seen_at = seen_at


class PostgresAuthRepository:
    """DB 설계서의 stores, users, auth_sessions를 사용하는 인증 저장소입니다."""

    def create_user(self, user: UserRecord) -> UserRecord:
        # 매장과 점주 계정을 한 트랜잭션으로 저장해 매장만 남는 상태를 방지합니다.
        try:
            with connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'INSERT INTO stores (store_id, name) VALUES (%s, %s) ON CONFLICT (store_id) DO NOTHING',
                        (user.store_id, user.store_name),
                    )
                    cursor.execute(
                        """
                        INSERT INTO users (user_id, store_id, email, password_hash, display_name)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (user.user_id, user.store_id, user.email, user.password_hash, user.display_name),
                    )
        except Exception as error:
            if 'duplicate key' in str(error).lower() or 'unique' in str(error).lower():
                raise ValueError('이미 가입된 이메일입니다.') from error
            raise
        return user

    def next_store_id(self) -> str:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 'S' || LPAD(nextval('store_number_seq')::text, 2, '0')")
            return cursor.fetchone()[0]

    def get_user_by_email(self, email: str) -> UserRecord | None:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT u.user_id, u.store_id, s.name, u.email, u.password_hash,
                       u.display_name, u.role, u.is_active
                FROM users u JOIN stores s ON s.store_id = u.store_id
                WHERE u.email = %s
                """,
                (email,),
            )
            row = cursor.fetchone()
        return self._user_from_row(row)

    def get_user(self, user_id: str) -> UserRecord | None:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT u.user_id, u.store_id, s.name, u.email, u.password_hash,
                       u.display_name, u.role, u.is_active
                FROM users u JOIN stores s ON s.store_id = u.store_id
                WHERE u.user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
        return self._user_from_row(row)

    def save_session(self, session: SessionRecord) -> None:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO auth_sessions
                    (session_id, user_id, session_token_hash, expires_at, last_seen_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session.session_id, session.user_id, session.token_hash, session.expires_at, session.last_seen_at),
            )

    def get_session(self, token_hash: str) -> SessionRecord | None:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, user_id, session_token_hash, expires_at, revoked_at, last_seen_at
                FROM auth_sessions WHERE session_token_hash = %s
                """,
                (token_hash,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return SessionRecord(*row)

    def revoke_session(self, token_hash: str, revoked_at: datetime) -> None:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                'UPDATE auth_sessions SET revoked_at = %s WHERE session_token_hash = %s',
                (revoked_at, token_hash),
            )

    def touch_session(self, token_hash: str, seen_at: datetime) -> None:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                'UPDATE auth_sessions SET last_seen_at = %s WHERE session_token_hash = %s',
                (seen_at, token_hash),
            )

    @staticmethod
    def _user_from_row(row: tuple | None) -> UserRecord | None:
        if row is None:
            return None
        return UserRecord(
            user_id=str(row[0]),
            store_id=row[1],
            store_name=row[2],
            email=row[3],
            password_hash=row[4],
            display_name=row[5],
            role=row[6],
            is_active=row[7],
        )
