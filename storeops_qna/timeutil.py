"""한국 시간(UTC+9) 처리. NFR-09: 모든 시각은 +09:00 ISO 8601."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def now_kst(demo_now: str | None = None) -> datetime:
    if demo_now:
        return parse_kst_iso(demo_now)
    return datetime.now(KST)


def parse_kst_iso(s: str) -> datetime:
    """ISO 8601 문자열을 읽는다. +09:00 이 붙어 있지 않으면 ValueError (임의로 고치지 않음)."""
    if not isinstance(s, str):
        raise ValueError("날짜는 문자열이어야 합니다")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"날짜 형식이 ISO 8601이 아닙니다: {s!r}") from e
    if dt.tzinfo is None:
        raise ValueError(f"한국 시간 표시(+09:00)가 없습니다: {s!r}")
    if dt.utcoffset() != timedelta(hours=9):
        raise ValueError(f"한국 시간(+09:00)이 아닙니다: {s!r}")
    return dt


def iso(dt: datetime) -> str:
    return dt.astimezone(KST).isoformat(timespec="seconds")


def _day_start(dt: datetime) -> datetime:
    return dt.astimezone(KST).replace(hour=0, minute=0, second=0, microsecond=0)


def date_hints(now: datetime) -> list[tuple[str, str, str]]:
    """'어제', '이번 주' 같은 표현을 [시작, 끝) 범위로 바꾼 표. 시스템 프롬프트에 넣는다.
    끝 시각은 포함하지 않는다(다음 날 0시). 한 주는 월요일에 시작한다."""
    today = _day_start(now)
    tomorrow = today + timedelta(days=1)
    yesterday = today - timedelta(days=1)
    week_start = today - timedelta(days=today.weekday())
    last_week_start = week_start - timedelta(days=7)
    month_start = today.replace(day=1)
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    return [
        ("오늘", iso(today), iso(tomorrow)),
        ("어제", iso(yesterday), iso(today)),
        ("이번 주(월~일)", iso(week_start), iso(week_start + timedelta(days=7))),
        ("지난주(월~일)", iso(last_week_start), iso(week_start)),
        ("최근 7일(오늘 포함)", iso(today - timedelta(days=6)), iso(tomorrow)),
        ("이번 달", iso(month_start), iso(next_month)),
    ]
