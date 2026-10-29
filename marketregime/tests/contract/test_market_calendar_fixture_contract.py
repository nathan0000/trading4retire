"""Contract tests for FixtureMarketCalendarAdapter against MarketCalendarPort."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from regime_engine.adapters.providers.fixture_market_calendar import (
    COVERAGE_END,
    COVERAGE_START,
    FixtureMarketCalendarAdapter,
)
from regime_engine.domain.enums import CalendarDayStatus, MarketSession
from regime_engine.domain.errors import CalendarCoverageError

NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture
def calendar() -> FixtureMarketCalendarAdapter:
    return FixtureMarketCalendarAdapter()


def _ny(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY_TZ)


def test_regular_weekday_is_open(calendar: FixtureMarketCalendarAdapter) -> None:
    bounds = calendar.session_bounds(date(2024, 1, 3))
    assert bounds.status is CalendarDayStatus.OPEN
    assert calendar.is_trading_day(date(2024, 1, 3))


def test_weekend_is_closed_without_fixture_row(calendar: FixtureMarketCalendarAdapter) -> None:
    bounds = calendar.session_bounds(date(2024, 1, 6))  # Saturday
    assert bounds.status is CalendarDayStatus.CLOSED
    assert not calendar.is_trading_day(date(2024, 1, 6))


@pytest.mark.parametrize(
    "holiday",
    [
        date(2024, 1, 1),
        date(2024, 7, 4),
        date(2024, 12, 25),
        date(2025, 1, 9),  # National Day of Mourning
        date(2026, 7, 3),  # Independence Day observed (July 4, 2026 is a Saturday)
    ],
)
def test_known_holidays_are_closed(calendar: FixtureMarketCalendarAdapter, holiday: date) -> None:
    assert calendar.session_bounds(holiday).status is CalendarDayStatus.CLOSED
    assert not calendar.is_trading_day(holiday)


@pytest.mark.parametrize(
    "early_close_day", [date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)]
)
def test_known_early_closes(calendar: FixtureMarketCalendarAdapter, early_close_day: date) -> None:
    bounds = calendar.session_bounds(early_close_day)
    assert bounds.status is CalendarDayStatus.EARLY_CLOSE
    assert bounds.regular_close is not None
    assert bounds.regular_close.hour == 13


def test_out_of_coverage_date_raises(calendar: FixtureMarketCalendarAdapter) -> None:
    with pytest.raises(CalendarCoverageError):
        calendar.session_bounds(COVERAGE_END + timedelta(days=1))
    with pytest.raises(CalendarCoverageError):
        calendar.session_bounds(COVERAGE_START - timedelta(days=1))


def test_session_for_walks_pre_market_through_closed(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    trading_day = date(2024, 1, 3)
    assert calendar.session_for(_ny(2024, 1, 3, 3, 0)) is MarketSession.CLOSED
    assert calendar.session_for(_ny(2024, 1, 3, 4, 0)) is MarketSession.PRE_MARKET
    assert calendar.session_for(_ny(2024, 1, 3, 9, 0)) is MarketSession.PRE_MARKET
    assert calendar.session_for(_ny(2024, 1, 3, 9, 30)) is MarketSession.REGULAR
    assert calendar.session_for(_ny(2024, 1, 3, 12, 0)) is MarketSession.REGULAR
    assert calendar.session_for(_ny(2024, 1, 3, 16, 0)) is MarketSession.AFTER_HOURS
    assert calendar.session_for(_ny(2024, 1, 3, 19, 59)) is MarketSession.AFTER_HOURS
    assert calendar.session_for(_ny(2024, 1, 3, 20, 0)) is MarketSession.CLOSED
    assert calendar.is_trading_day(trading_day)


def test_session_for_on_early_close_day_shortens_regular_session(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    assert calendar.session_for(_ny(2024, 7, 3, 12, 59)) is MarketSession.REGULAR
    assert calendar.session_for(_ny(2024, 7, 3, 13, 0)) is MarketSession.AFTER_HOURS


def test_session_for_on_full_holiday_is_closed_all_day(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    assert calendar.session_for(_ny(2024, 12, 25, 10, 0)) is MarketSession.CLOSED


def test_session_for_rejects_naive_datetime(calendar: FixtureMarketCalendarAdapter) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.session_for(datetime(2024, 1, 3, 10, 0))


def test_session_for_handles_utc_input_across_dst_spring_forward(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    # DST spring-forward happens 2024-03-10 (a Sunday, not a trading day) at
    # 02:00 local time, switching EST (UTC-5) to EDT (UTC-4). The same UTC
    # instant resolves to a different session before vs. after the shift:
    # Friday 2024-03-08, 13:30 UTC = 08:30 EST -> still PRE_MARKET.
    before_transition = datetime(2024, 3, 8, 13, 30, tzinfo=UTC)
    assert calendar.session_for(before_transition) is MarketSession.PRE_MARKET
    # Monday 2024-03-11, 13:30 UTC = 09:30 EDT -> REGULAR open, one hour
    # "earlier" in UTC terms than the prior Friday's open would have been.
    after_transition = datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert calendar.session_for(after_transition) is MarketSession.REGULAR


def test_session_for_handles_utc_input_across_dst_fall_back(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    # After 2024-11-03 fall-back, 9:30am ET regular open is UTC-5 (EST).
    regular_open_utc = datetime(2024, 11, 4, 14, 30, tzinfo=UTC)  # 9:30 EST
    assert calendar.session_for(regular_open_utc) is MarketSession.REGULAR


def test_next_trading_day_skips_weekend_and_holiday(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    # Fri 2024-12-20 -> next trading day should skip the Christmas Day
    # holiday and the weekend around it, landing on 2024-12-26.
    assert calendar.next_trading_day(date(2024, 12, 24)) == date(2024, 12, 26)


def test_previous_trading_day_skips_weekend_and_holiday(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    assert calendar.previous_trading_day(date(2024, 12, 26)) == date(2024, 12, 24)


def test_next_trading_day_across_new_year_holiday_weekend(
    calendar: FixtureMarketCalendarAdapter,
) -> None:
    # 2024-12-31 (Tue) -> 2025-01-01 is a holiday -> next trading day 2025-01-02.
    assert calendar.next_trading_day(date(2024, 12, 31)) == date(2025, 1, 2)
