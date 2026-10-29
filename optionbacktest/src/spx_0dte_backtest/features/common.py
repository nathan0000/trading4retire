"""Shared causal feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def session_dates(frame: pd.DataFrame, timezone: str) -> pd.Series:
    return frame["timestamp"].dt.tz_convert(timezone).dt.date


def true_range(group: pd.DataFrame) -> pd.Series:
    previous_close = group["close"].shift(1)
    return pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - previous_close).abs(),
            (group["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def consecutive_counts(condition: pd.Series) -> pd.Series:
    """Count the current consecutive True run without observing later rows."""

    groups = (~condition).cumsum()
    return condition.astype(int).groupby(groups).cumsum().astype(int)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))
