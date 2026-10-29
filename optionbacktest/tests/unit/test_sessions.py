from datetime import date, datetime
from zoneinfo import ZoneInfo

from spx_0dte_backtest.market import SessionService


def test_dst_offsets_are_new_york_offsets() -> None:
    sessions = SessionService()
    winter = sessions.bounds(date(2026, 3, 6))
    summer = sessions.bounds(date(2026, 3, 9))
    assert winter is not None and winter.open.utcoffset().total_seconds() == -5 * 3600
    assert summer is not None and summer.open.utcoffset().total_seconds() == -4 * 3600


def test_holiday_has_no_session() -> None:
    assert SessionService().bounds(date(2026, 12, 25)) is None


def test_early_close_grid() -> None:
    sessions = SessionService()
    bounds = sessions.bounds(date(2025, 11, 28))
    assert bounds is not None and bounds.is_early_close
    assert (bounds.close.hour, bounds.close.minute) == (13, 0)
    grid = sessions.expected_timestamps(date(2025, 11, 28))
    assert (grid[-1].hour, grid[-1].minute) == (12, 55)


def test_containing_session_rejects_off_hours() -> None:
    timestamp = datetime(2026, 8, 12, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    assert SessionService().containing_session(timestamp) is None
