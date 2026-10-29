"""Causal intraday trend features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_0dte_backtest.features.common import safe_ratio, true_range
from spx_0dte_backtest.features.settings import FeatureSettings


def add_trend_features(group: pd.DataFrame, settings: FeatureSettings) -> pd.DataFrame:
    result = group.copy()
    close = result["close"]
    result["return_5m"] = close.pct_change(1, fill_method=None)
    result["return_15m"] = close.pct_change(3, fill_method=None)
    result["return_30m"] = close.pct_change(6, fill_method=None)
    result["return_60m"] = close.pct_change(12, fill_method=None)

    typical = (result["high"] + result["low"] + result["close"]) / 3.0
    valid_volume = pd.to_numeric(result["volume"], errors="coerce")
    if valid_volume.notna().all() and (valid_volume > 0).any():
        cumulative_volume = valid_volume.cumsum()
        result["vwap"] = (typical * valid_volume).cumsum().div(cumulative_volume)
    else:
        result["vwap"] = np.nan
    result["vwap_slope"] = result["vwap"].diff(settings.slope_bars)
    result["distance_from_vwap"] = result["close"] - result["vwap"]

    result["ema"] = close.ewm(span=settings.ema_span, adjust=False).mean()
    result["ema_slope"] = result["ema"].diff(settings.slope_bars)

    high_delta = result["high"].diff()
    low_delta = -result["low"].diff()
    plus_dm = pd.Series(
        np.where((high_delta > low_delta) & (high_delta > 0), high_delta, 0.0), index=result.index
    )
    minus_dm = pd.Series(
        np.where((low_delta > high_delta) & (low_delta > 0), low_delta, 0.0), index=result.index
    )
    atr = true_range(result).ewm(alpha=1 / settings.adx_period, adjust=False).mean()
    plus_di = 100 * safe_ratio(plus_dm.ewm(alpha=1 / settings.adx_period, adjust=False).mean(), atr)
    minus_di = 100 * safe_ratio(
        minus_dm.ewm(alpha=1 / settings.adx_period, adjust=False).mean(), atr
    )
    dx = 100 * safe_ratio((plus_di - minus_di).abs(), plus_di + minus_di)
    result["adx"] = dx.ewm(alpha=1 / settings.adx_period, adjust=False).mean()
    return result


def classify_trend(frame: pd.DataFrame, settings: FeatureSettings) -> pd.Series:
    atr = frame["atr"].replace(0, np.nan)
    score = frame["return_30m"].fillna(0) * 100
    score += (frame["distance_from_vwap"] / atr).fillna(0) * 0.5
    score += np.sign(frame["ema_slope"].fillna(0)) * 0.25
    return pd.Series(
        np.select(
            [
                score >= settings.trend_strong_threshold,
                score >= settings.trend_threshold,
                score <= -settings.trend_strong_threshold,
                score <= -settings.trend_threshold,
            ],
            ["STRONG_BULL", "BULL", "STRONG_BEAR", "BEAR"],
            default="NEUTRAL",
        ),
        index=frame.index,
        dtype="string",
    )
