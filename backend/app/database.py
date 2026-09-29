"""환경변수에서 PostgreSQL 연결 정보를 읽고 연결을 생성합니다."""

import os

import psycopg


def get_database_url() -> str | None:
    """DATABASE_URL이 없으면 메모리 저장소를 사용하도록 None을 반환합니다."""

    return os.getenv('DATABASE_URL')


def connect() -> psycopg.Connection:
    """요청 단위로 PostgreSQL 연결을 생성합니다."""

    database_url = get_database_url()
    if not database_url:
        raise RuntimeError('DATABASE_URL이 설정되지 않았습니다.')
    return psycopg.connect(database_url)
