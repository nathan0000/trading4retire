"""Unit tests for the Clock port implementations."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from regime_engine.ports.clock import FixedClock, SystemClock


def test_system_clock_returns_timezone_aware_utc_now() -> None:
    before = datetime.now(UTC)
    observed = SystemClock().now()
    after = datetime.now(UTC)

    assert observed.tzinfo is not None
    assert before <= observed <= after


def test_fixed_clock_returns_the_configured_time() -> None:
    fixed = datetime(2026, 1, 5, 10, 45, tzinfo=UTC)
    clock = FixedClock(fixed)
    assert clock.now() == fixed


def test_fixed_clock_normalizes_non_utc_input_to_utc() -> None:
    minus_five = timezone(timedelta(hours=-5))
    local = datetime(2026, 1, 5, 5, 45, tzinfo=minus_five)
    clock = FixedClock(local)
    assert clock.now() == datetime(2026, 1, 5, 10, 45, tzinfo=UTC)
    assert clock.now().tzinfo == UTC


def test_fixed_clock_rejects_naive_datetime_on_construction() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FixedClock(datetime(2026, 1, 5, 10, 45))


def test_fixed_clock_set_advances_the_clock() -> None:
    clock = FixedClock(datetime(2026, 1, 5, 10, 45, tzinfo=UTC))
    clock.set(datetime(2026, 1, 5, 11, 0, tzinfo=UTC))
    assert clock.now() == datetime(2026, 1, 5, 11, 0, tzinfo=UTC)


def test_fixed_clock_set_rejects_naive_datetime() -> None:
    clock = FixedClock(datetime(2026, 1, 5, 10, 45, tzinfo=UTC))
    with pytest.raises(ValueError, match="timezone-aware"):
        clock.set(datetime(2026, 1, 5, 11, 0))
