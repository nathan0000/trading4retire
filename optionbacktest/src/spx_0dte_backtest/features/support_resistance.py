"""Point-in-time price levels and opening-range features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_0dte_backtest.features.settings import FeatureSettings


def add_intraday_levels(group: pd.DataFrame, settings: FeatureSettings) -> pd.DataFrame:
    result = group.copy()
    count = max(1, settings.opening_range_minutes // 5)
    opening_high = result["high"].iloc[:count].cummax()
    opening_low = result["low"].iloc[:count].cummin()
    result["opening_range_high"] = opening_high.reindex(result.index).ffill()
    result["opening_range_low"] = opening_low.reindex(result.index).ffill()
    result["current_day_open"] = result["open"].iloc[0]
    result["opening_range_breakout"] = np.select(
        [
            result["close"] > result["opening_range_high"],
            result["close"] < result["opening_range_low"],
        ],
        ["ABOVE", "BELOW"],
        default="INSIDE",
    )
    result["recent_swing_high"] = (
        result["high"].rolling(settings.swing_window_bars, min_periods=1).max()
    )
    result["recent_swing_low"] = (
        result["low"].rolling(settings.swing_window_bars, min_periods=1).min()
    )
    # SPX cash bars cannot provide defensible overnight levels.
    result["overnight_high"] = np.nan
    result["overnight_low"] = np.nan
    return result


def add_previous_day_levels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    summaries = result.groupby("session_date", sort=True).agg(
        day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")
    )
    prior = summaries.shift(1).rename(
        columns={
            "day_high": "previous_day_high",
            "day_low": "previous_day_low",
            "day_close": "previous_day_close",
        }
    )
    return result.join(prior, on="session_date")
