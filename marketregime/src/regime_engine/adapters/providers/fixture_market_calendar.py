"""Fixture NYSE trading-calendar adapter (CLAUDE.md section 10 fixture-first requirement).

Sourced from `fixtures/nyse_calendar.csv`, an exceptions-only list of full
closures and early closes verified against NYSE Group's official holiday
announcements; see docs/data_dictionary.md for the citation. Any weekday
in the coverage window not listed defaults to a standard open session;
weekends default to closed. Session resolution converts `as_of` to
America/New_York via `zoneinfo` so DST transitions are handled correctly
without manual offset arithmetic.
"""

import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from regime_engine.domain.enums import CalendarDayStatus, MarketSession
from regime_engine.domain.errors import CalendarCoverageError
from regime_engine.domain.models import SessionBounds
from regime_engine.ports.market_data import MarketCalendarPort

NY_TZ = ZoneInfo("America/New_York")

PRE_MARKET_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE_TIME = time(13, 0)
AFTER_HOURS_CLOSE = time(20, 0)

COVERAGE_START = date(2024, 1, 1)
COVERAGE_END = date(2026, 12, 31)

_DEFAULT_CSV_PATH = Path(__file__).parent / "fixtures" / "nyse_calendar.csv"
_MAX_SEARCH_DAYS = 14
_SATURDAY = 5


def _parse_close_time(raw: str) -> time:
    hour_str, minute_str = raw.split(":")
    return time(int(hour_str), int(minute_str))


def _load_overrides(csv_path: Path) -> dict[date, SessionBounds]:
    overrides: dict[date, SessionBounds] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            status = CalendarDayStatus(row["status"])
            label = row["label"] or None
            if status is CalendarDayStatus.EARLY_CLOSE:
                overrides[day] = SessionBounds(
                    day=day,
                    status=status,
                    pre_market_open=PRE_MARKET_OPEN,
                    regular_open=REGULAR_OPEN,
                    regular_close=_parse_close_time(row["close_time"]),
                    after_hours_close=AFTER_HOURS_CLOSE,
                    label=label,
                )
            else:
                overrides[day] = SessionBounds(
                    day=day,
                    status=CalendarDayStatus.CLOSED,
                    pre_market_open=None,
                    regular_open=None,
                    regular_close=None,
                    after_hours_close=None,
                    label=label,
                )
    return overrides


class FixtureMarketCalendarAdapter(MarketCalendarPort):
    """MarketCalendarPort implementation backed by the checked-in NYSE fixture CSV."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self._path = csv_path or _DEFAULT_CSV_PATH
        self._overrides = _load_overrides(self._path)

    def session_bounds(self, day: date) -> SessionBounds:
        if not (COVERAGE_START <= day <= COVERAGE_END):
            raise CalendarCoverageError(day)
        if day in self._overrides:
            return self._overrides[day]
        if day.weekday() >= _SATURDAY:  # Saturday, Sunday
            return SessionBounds(
                day=day,
                status=CalendarDayStatus.CLOSED,
                pre_market_open=None,
                regular_open=None,
                regular_close=None,
                after_hours_close=None,
                label=None,
            )
        return SessionBounds(
            day=day,
            status=CalendarDayStatus.OPEN,
            pre_market_open=PRE_MARKET_OPEN,
            regular_open=REGULAR_OPEN,
            regular_close=REGULAR_CLOSE,
            after_hours_close=AFTER_HOURS_CLOSE,
            label=None,
        )

    def is_trading_day(self, day: date) -> bool:
        return self.session_bounds(day).status is not CalendarDayStatus.CLOSED

    def session_for(self, as_of: datetime) -> MarketSession:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        local = as_of.astimezone(NY_TZ)
        bounds = self.session_bounds(local.date())
        if bounds.status is CalendarDayStatus.CLOSED:
            return MarketSession.CLOSED

        assert bounds.pre_market_open is not None
        assert bounds.regular_open is not None
        assert bounds.regular_close is not None
        assert bounds.after_hours_close is not None

        local_time = local.time()
        if local_time < bounds.pre_market_open:
            return MarketSession.CLOSED
        if local_time < bounds.regular_open:
            return MarketSession.PRE_MARKET
        if local_time < bounds.regular_close:
            return MarketSession.REGULAR
        if local_time < bounds.after_hours_close:
            return MarketSession.AFTER_HOURS
        return MarketSession.CLOSED

    def next_trading_day(self, day: date) -> date:
        cursor = day + timedelta(days=1)
        for _ in range(_MAX_SEARCH_DAYS):
            if self.is_trading_day(cursor):
                return cursor
            cursor += timedelta(days=1)
        raise CalendarCoverageError(cursor)

    def previous_trading_day(self, day: date) -> date:
        cursor = day - timedelta(days=1)
        for _ in range(_MAX_SEARCH_DAYS):
            if self.is_trading_day(cursor):
                return cursor
            cursor -= timedelta(days=1)
        raise CalendarCoverageError(cursor)
