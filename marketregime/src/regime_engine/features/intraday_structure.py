"""Intraday structure primitives: session-anchored VWAP (CLAUDE.md section 7).

The caller is responsible for anchoring: pass only the bars of one regular
session, starting with the 09:30 ET bar and ending with the last bar
completed at `as_of`.
"""

import numpy as np
from numpy.typing import NDArray


def session_vwap(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
) -> float | None:
    """Volume-weighted average of the typical price (H + L + C) / 3 over the given bars.

    Returns None when the bars carry no volume (cash indices such as SPX report zero
    volume at many vendors); VWAP is undefined there rather than equal-weighted.
    """
    total_volume = float(volume.sum())
    if total_volume <= 0.0:
        return None
    typical = (high + low + close) / 3.0
    return float((typical * volume).sum() / total_volume)
