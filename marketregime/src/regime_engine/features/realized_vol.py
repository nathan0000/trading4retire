"""Realized-volatility primitives: true range and Wilder ATR (CLAUDE.md section 7)."""

import numpy as np
from numpy.typing import NDArray


def true_range(
    high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-bar true range. The first bar has no prior close, so its range is high - low.

    Only the prior bar's close is used, never a later one.
    """
    ranges = high - low
    if ranges.size > 1:
        prior_close = close[:-1]
        ranges[1:] = np.maximum.reduce(
            [ranges[1:], np.abs(high[1:] - prior_close), np.abs(low[1:] - prior_close)]
        )
    return ranges


def atr_last(ranges: NDArray[np.float64], period: int) -> float | None:
    """Wilder ATR over true ranges: seeded with the mean of the first `period`, then
    smoothed with alpha = 1 / period. Returns None until `period` ranges exist.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    if ranges.size < period:
        return None
    value = float(ranges[:period].mean())
    for current in ranges[period:].tolist():
        value = (value * (period - 1) + current) / period
    return value
