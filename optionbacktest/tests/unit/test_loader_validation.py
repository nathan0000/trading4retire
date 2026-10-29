from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from spx_0dte_backtest.config import AppConfig, DataConfig
from spx_0dte_backtest.data.loaders import SpxBarLoader

HEADER = "timestamp,open,high,low,close,volume\n"


def test_loads_and_normalizes_aware_csv(config, write_csv) -> None:
    path = write_csv(
        HEADER
        + "2026-08-12T13:30:00+00:00,6000,6002,5999,6001,\n"
        + "2026-08-12T13:35:00+00:00,6001,6003,6000,6002,10\n"
    )
    result = SpxBarLoader(config).load(path)
    assert result.validation.is_valid
    assert result.validation.accepted_rows == 2
    assert str(result.bars["timestamp"].dt.tz) == "America/New_York"
    assert result.bars.iloc[0]["timestamp"].hour == 9


def test_naive_timestamp_requires_declared_source_timezone(config, write_csv) -> None:
    path = write_csv(HEADER + "2026-08-12 09:30:00,6000,6002,5999,6001,1\n")
    result = SpxBarLoader(config).load(path)
    assert result.validation.rejected_rows == 1
    assert result.validation.counts_by_code == {"NAIVE_TIMESTAMP": 1}

    localized_config = AppConfig(data=DataConfig(source_timezone="America/New_York"))
    localized = SpxBarLoader(localized_config).load(path)
    assert localized.validation.is_valid
    assert localized.validation.accepted_rows == 1


def test_reports_duplicate_invalid_ohlc_and_internal_gap(config, write_csv) -> None:
    path = write_csv(
        HEADER
        + "2026-08-12T09:30:00-04:00,6000,6002,5999,6001,1\n"
        + "2026-08-12T09:30:00-04:00,6000,6002,5999,6001,1\n"
        + "2026-08-12T09:35:00-04:00,6000,5998,5999,6001,1\n"
        + "2026-08-12T09:40:00-04:00,6001,6003,6000,6002,1\n"
        + "2026-08-12T09:50:00-04:00,6002,6004,6001,6003,1\n"
    )
    result = SpxBarLoader(config).load(path)
    counts = result.validation.counts_by_code
    assert counts["DUPLICATE_TIMESTAMP"] == 2
    assert counts["INVALID_OHLC"] == 1
    assert counts["MISSING_BAR"] == 1
    assert result.validation.rejected_rows == 3


def test_holiday_and_early_close_are_rejected(config, write_csv) -> None:
    holiday = write_csv(HEADER + "2026-12-25T09:30:00-05:00,6000,6002,5999,6001,1\n", "holiday.csv")
    early = write_csv(HEADER + "2025-11-28T13:00:00-05:00,6000,6002,5999,6001,1\n", "early.csv")
    assert "MARKET_HOLIDAY" in SpxBarLoader(config).load(holiday).validation.counts_by_code
    assert "EARLY_CLOSE_VIOLATION" in SpxBarLoader(config).load(early).validation.counts_by_code


def test_cutoff_prevents_future_rows_from_entering_context(config, write_csv) -> None:
    prefix = HEADER + "2026-08-12T09:30:00-04:00,6000,6002,5999,6001,1\n"
    prefix_path = write_csv(prefix, "prefix.csv")
    future_path = write_csv(
        prefix + "2026-08-12T09:35:00-04:00,6001,6003,6000,6002,1\n", "future.csv"
    )
    cutoff = datetime(2026, 8, 12, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    first = SpxBarLoader(config).load(prefix_path, end=cutoff).bars
    second = SpxBarLoader(config).load(future_path, end=cutoff).bars
    assert first.drop(columns="source").equals(second.drop(columns="source"))


def test_unsorted_input_is_reported_but_output_is_sorted(config, write_csv) -> None:
    path = write_csv(
        HEADER
        + "2026-08-12T09:35:00-04:00,6001,6003,6000,6002,1\n"
        + "2026-08-12T09:30:00-04:00,6000,6002,5999,6001,1\n"
    )
    result = SpxBarLoader(config).load(path)
    assert "UNSORTED_INPUT" in result.validation.counts_by_code
    assert result.bars["timestamp"].is_monotonic_increasing
