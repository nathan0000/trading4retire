"""Contract tests for FixturePriceBarsAdapter, focused on point-in-time filtering."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from regime_engine.adapters.providers.fixture_price_bars import FixturePriceBarsAdapter
from regime_engine.domain.enums import Quality

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "price_bars" / "spx_daily_sample.csv"
)

WIDE_START = datetime(2024, 1, 1, tzinfo=UTC)
WIDE_END = datetime(2024, 2, 1, tzinfo=UTC)


@pytest.fixture
def adapter() -> FixturePriceBarsAdapter:
    return FixturePriceBarsAdapter(FIXTURE_PATH, publish_lag=timedelta(minutes=5))


def test_returns_all_bars_when_as_of_is_far_in_the_future(
    adapter: FixturePriceBarsAdapter,
) -> None:
    bars = adapter.get_bars(
        "SPX", WIDE_START, WIDE_END, "1d", as_of=datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert len(bars) == 10


def test_bars_are_ordered_ascending_by_event_time(adapter: FixturePriceBarsAdapter) -> None:
    bars = adapter.get_bars(
        "SPX", WIDE_START, WIDE_END, "1d", as_of=datetime(2030, 1, 1, tzinfo=UTC)
    )
    event_times = [obs.event_time for obs in bars]
    assert event_times == sorted(event_times)


def test_bar_provenance_fields_are_populated(adapter: FixturePriceBarsAdapter) -> None:
    bars = adapter.get_bars(
        "SPX", WIDE_START, WIDE_END, "1d", as_of=datetime(2030, 1, 1, tzinfo=UTC)
    )
    first = bars[0]
    assert first.source == "fixture"
    assert first.source_symbol == "SPX"
    assert first.quality is Quality.GOOD
    assert first.value is not None
    assert first.available_time == first.event_time + timedelta(minutes=5)


def test_bar_excluded_when_as_of_before_available_time(adapter: FixturePriceBarsAdapter) -> None:
    all_bars = adapter.get_bars(
        "SPX", WIDE_START, WIDE_END, "1d", as_of=datetime(2030, 1, 1, tzinfo=UTC)
    )
    first_bar = all_bars[0]

    just_before_available = first_bar.available_time - timedelta(seconds=1)
    filtered = adapter.get_bars("SPX", WIDE_START, WIDE_END, "1d", as_of=just_before_available)
    assert first_bar.event_time not in [obs.event_time for obs in filtered]

    exactly_available = first_bar.available_time
    filtered_at_available = adapter.get_bars(
        "SPX", WIDE_START, WIDE_END, "1d", as_of=exactly_available
    )
    assert first_bar.event_time in [obs.event_time for obs in filtered_at_available]


def test_filters_by_symbol_and_date_range(adapter: FixturePriceBarsAdapter) -> None:
    narrow_start = datetime(2024, 1, 8, tzinfo=UTC)
    # Daily bars are stamped at the 16:00 New York close (21:00 UTC), so the
    # range must extend past that instant to include the 2024-01-12 bar.
    narrow_end = datetime(2024, 1, 13, tzinfo=UTC)
    bars = adapter.get_bars(
        "SPX", narrow_start, narrow_end, "1d", as_of=datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert len(bars) == 5

    no_match = adapter.get_bars(
        "MISSING_SYMBOL", WIDE_START, WIDE_END, "1d", as_of=datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert no_match == ()
