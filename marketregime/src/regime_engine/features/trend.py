"""Trend and momentum primitives: EMA and Wilder RSI (CLAUDE.md section 7).

Pure numeric functions over closes ordered oldest-first. Each returns the
value as of the LAST element only, so a caller that passes the bars known
at `as_of` cannot leak later data. `None` means "not warmed up"; it is never
silently replaced by zero or a neutral value (CLAUDE.md section 2.6).
"""

import numpy as np
from numpy.typing import NDArray

_NEUTRAL_RSI = 50.0
_MAX_RSI = 100.0


def ema_last(closes: NDArray[np.float64], period: int) -> float | None:
    """Exponential moving average of `closes`, seeded with the SMA of the first `period` values.

    alpha = 2 / (period + 1). Returns None until `period` values exist.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    if closes.size < period:
        return None
    alpha = 2.0 / (period + 1)
    value = float(closes[:period].mean())
    for close in closes[period:].tolist():
        value += alpha * (close - value)
    return value


def rsi_last(closes: NDArray[np.float64], period: int) -> float | None:
    """Wilder RSI: averages seeded with the mean of the first `period` changes, then
    smoothed with alpha = 1 / period. Needs `period + 1` closes.

    A window with no losses reads 100, no gains reads 0, and no movement at all reads 50.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    if closes.size < period + 1:
        return None
    changes = np.diff(closes)
    gains = np.clip(changes, 0.0, None)
    losses = np.clip(-changes, 0.0, None)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for gain, loss in zip(gains[period:].tolist(), losses[period:].tolist(), strict=True):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0.0:
        return _MAX_RSI if avg_gain > 0.0 else _NEUTRAL_RSI
    return _MAX_RSI - _MAX_RSI / (1.0 + avg_gain / avg_loss)
