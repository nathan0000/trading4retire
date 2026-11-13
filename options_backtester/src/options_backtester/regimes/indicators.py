"""Causal indicator helpers. Every function uses only rows at or before each index."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    ranges = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1)
    return ranges.max(axis=1, skipna=False).fillna(high - low)


def _wilder(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), window)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (0-100)."""
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    average_range = atr(high, low, close, window)
    plus_di = 100.0 * _wilder(plus_dm, window) / average_range
    minus_di = 100.0 * _wilder(minus_dm, window) / average_range
    total = (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder(100.0 * (plus_di - minus_di).abs() / total, window)


def realized_volatility(close: pd.Series, window: int = 20, periods: int = 252) -> pd.Series:
    """Annualized close-to-close volatility (decimal)."""
    returns = np.log(close).diff()
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods)


def iv_rank(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Position of the current value in its trailing high-low range, 0-100."""
    low = series.rolling(window, min_periods=min_periods).min()
    high = series.rolling(window, min_periods=min_periods).max()
    span = (high - low).replace(0.0, np.nan)
    return 100.0 * (series - low) / span


def iv_percentile(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Percentage of trailing observations strictly below the current value, 0-100."""

    def below(values: np.ndarray) -> float:
        return 100.0 * float(np.mean(values[:-1] < values[-1])) if len(values) > 1 else np.nan

    return series.rolling(window, min_periods=min_periods).apply(below, raw=True)


def term_structure_ratio(front: pd.Series, back: pd.Series) -> pd.Series:
    """Front / back implied vol: above 1 is backwardation, below 1 is contango."""
    return front / back.replace(0.0, np.nan)


def hurst_exponent(close: pd.Series, window: int = 100, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent from the scaling of lagged log-price differences.

    Below 0.5 suggests mean reversion, near 0.5 a random walk, above 0.5 trending.
    """
    lags = np.arange(2, max_lag + 1)
    log_lags = np.log(lags)

    def estimate(values: np.ndarray) -> float:
        spread = [np.std(values[lag:] - values[:-lag]) for lag in lags]
        if min(spread) <= 0:
            return np.nan
        return float(np.polyfit(log_lags, np.log(spread), 1)[0])

    return np.log(close).rolling(window, min_periods=window).apply(estimate, raw=True)
