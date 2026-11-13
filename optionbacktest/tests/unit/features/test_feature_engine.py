from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.features import FeatureEngine


def make_bars(periods: int = 30, day: str = "2026-08-12", volume: float = 100.0) -> pd.DataFrame:
    timestamps = pd.date_range(f"{day} 09:30", periods=periods, freq="5min", tz="America/New_York")
    close = 6000.0 + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPX",
            "open": close - 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
            "source": "fixture",
        }
    )


def test_feature_engine_has_required_phase_two_columns() -> None:
    result = FeatureEngine(AppConfig()).compute(make_bars())
    required = {
        "return_5m",
        "return_15m",
        "return_30m",
        "return_60m",
        "vwap",
        "vwap_slope",
        "distance_from_vwap",
        "ema",
        "ema_slope",
        "adx",
        "atr",
        "realized_vol_5m",
        "realized_vol_30m",
        "realized_vol_60m",
        "intraday_range",
        "range_expansion",
        "opening_range_high",
        "opening_range_low",
        "opening_range_breakout",
        "previous_day_high",
        "previous_day_low",
        "previous_day_close",
        "recent_swing_high",
        "recent_swing_low",
        "consecutive_up_bars",
        "consecutive_down_bars",
        "atr_normalized_return",
        "trend_classification",
        "momentum_classification",
        "vix",
        "vix_change",
        "expected_move",
        "data_cutoff_timestamp",
    }
    assert required.issubset(result.columns)
    assert result["data_cutoff_timestamp"].equals(result["timestamp"])


def test_bar_start_rows_are_labelled_when_their_close_is_known() -> None:
    bars = make_bars(3)
    result = FeatureEngine(AppConfig()).compute(bars)
    # The 09:30 bar covers 09:30-09:35, so its close is first known at 09:35.
    assert result["bar_start"].equals(bars["timestamp"])
    assert result["timestamp"].equals(bars["timestamp"] + pd.Timedelta(minutes=5))
    assert result["data_cutoff_timestamp"].equals(result["timestamp"])
    assert result.iloc[0]["close"] == pytest.approx(bars.iloc[0]["close"])


def test_bar_end_rows_keep_their_timestamps() -> None:
    bars = make_bars(3)
    config = AppConfig(data=AppConfig().data.model_copy(update={"timestamp_semantics": "bar_end"}))
    result = FeatureEngine(config).compute(bars)
    assert "bar_start" not in result.columns
    assert result["timestamp"].equals(bars["timestamp"])


def test_future_rows_cannot_change_features_at_timestamp() -> None:
    bars = make_bars(40)
    engine = FeatureEngine(AppConfig())
    through_t = engine.compute(bars.iloc[:20])
    with_future = engine.compute(bars)
    pd.testing.assert_series_equal(
        through_t.iloc[-1], with_future.iloc[19], check_names=False, check_dtype=False
    )


def test_features_reset_at_session_boundary_and_previous_levels_use_prior_day() -> None:
    first = make_bars(6, "2026-08-11")
    second = make_bars(3, "2026-08-12")
    second.loc[:, "close"] += 20
    second.loc[:, "open"] += 20
    second.loc[:, "high"] += 20
    second.loc[:, "low"] += 20
    result = FeatureEngine(AppConfig()).compute(pd.concat([first, second], ignore_index=True))
    first_second_day = result.iloc[6]
    assert pd.isna(first_second_day["return_5m"])
    assert first_second_day["intraday_range"] == pytest.approx(2.0)
    assert first_second_day["previous_day_high"] == pytest.approx(first["high"].max())
    assert first_second_day["previous_day_low"] == pytest.approx(first["low"].min())
    assert first_second_day["previous_day_close"] == pytest.approx(first.iloc[-1]["close"])


def test_opening_range_is_expanding_until_window_finishes_then_frozen() -> None:
    result = FeatureEngine(AppConfig()).compute(make_bars(10))
    assert result.iloc[0]["opening_range_high"] == pytest.approx(result.iloc[0]["high"])
    assert result.iloc[4]["opening_range_high"] == pytest.approx(result.iloc[:5]["high"].max())
    assert result.iloc[8]["opening_range_high"] == pytest.approx(result.iloc[:6]["high"].max())


def test_vwap_is_unavailable_without_valid_volume() -> None:
    bars = make_bars(3)
    bars["volume"] = pd.NA
    result = FeatureEngine(AppConfig()).compute(bars)
    assert result["vwap"].isna().all()
    assert result["distance_from_vwap"].isna().all()


def test_vix_alignment_uses_only_last_known_observation() -> None:
    bars = make_bars(4)
    vix = pd.DataFrame(
        {
            "timestamp": [bars.iloc[0]["timestamp"], bars.iloc[2]["timestamp"]],
            "vix": [20.0, 21.0],
        }
    )
    result = FeatureEngine(AppConfig()).compute(bars, vix=vix)
    assert result["vix"].tolist() == [20.0, 20.0, 21.0, 21.0]
    assert result.iloc[0]["expected_move"] == pytest.approx(6000 * 0.20 / np.sqrt(252))


def test_future_only_vix_is_not_backfilled() -> None:
    bars = make_bars(2)
    vix = pd.DataFrame({"timestamp": [bars.iloc[1]["timestamp"]], "vix": [20.0]})
    result = FeatureEngine(AppConfig()).compute(bars, vix=vix)
    assert pd.isna(result.iloc[0]["vix"])
    assert result.iloc[1]["vix"] == 20.0


def test_input_must_be_sorted_unique_and_timezone_aware() -> None:
    engine = FeatureEngine(AppConfig())
    unsorted = make_bars(2).iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        engine.compute(unsorted)
    duplicate = pd.concat([make_bars(2), make_bars(2).iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        engine.compute(duplicate)
    naive = make_bars(2)
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.compute(naive)
