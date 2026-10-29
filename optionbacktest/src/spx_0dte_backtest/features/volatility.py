"""Causal realized-volatility and range features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_0dte_backtest.features.common import safe_ratio, true_range
from spx_0dte_backtest.features.settings import FeatureSettings


def add_volatility_features(group: pd.DataFrame, settings: FeatureSettings) -> pd.DataFrame:
    result = group.copy()
    returns = result["close"].pct_change(fill_method=None)
    annualizer = np.sqrt(settings.annualization_periods)
    for label, bars in (("5m", 1), ("30m", 6), ("60m", 12)):
        result[f"realized_vol_{label}"] = (
            returns.rolling(bars, min_periods=bars).std(ddof=0) * annualizer
        )
    ranges = true_range(result)
    result["atr"] = ranges.ewm(alpha=1 / settings.atr_period, adjust=False).mean()
    result["intraday_range"] = result["high"].cummax() - result["low"].cummin()
    trailing_range = ranges.rolling(settings.atr_period, min_periods=1).mean()
    result["range_expansion"] = safe_ratio(ranges, trailing_range)
    return result


def align_vix(bars: pd.DataFrame, vix: pd.DataFrame | None) -> pd.DataFrame:
    result = bars.copy()
    if vix is None or vix.empty:
        result["vix"] = np.nan
        result["vix_change"] = np.nan
        result["expected_move"] = np.nan
        return result
    required = {"timestamp", "vix"}
    missing = required.difference(vix.columns)
    if missing:
        raise ValueError(f"VIX frame missing required columns: {sorted(missing)}")
    observations = vix[["timestamp", "vix"]].copy()
    if observations["timestamp"].dt.tz is None:
        raise ValueError("VIX timestamps must be timezone-aware")
    if not observations["timestamp"].is_monotonic_increasing:
        raise ValueError("VIX timestamps must be sorted")
    if not observations["timestamp"].is_unique:
        raise ValueError("VIX timestamps must be unique")
    observations["vix"] = pd.to_numeric(observations["vix"], errors="coerce")
    observations["vix_change"] = observations["vix"].pct_change(fill_method=None)
    result = pd.merge_asof(result, observations, on="timestamp", direction="backward")
    result["expected_move"] = result["close"] * (result["vix"] / 100.0) / np.sqrt(252)
    return result
