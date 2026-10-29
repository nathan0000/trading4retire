"""Causal intraday momentum features and classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_0dte_backtest.features.common import consecutive_counts, safe_ratio
from spx_0dte_backtest.features.settings import FeatureSettings


def add_momentum_features(group: pd.DataFrame) -> pd.DataFrame:
    result = group.copy()
    up = result["close"] > result["open"]
    down = result["close"] < result["open"]
    result["consecutive_up_bars"] = consecutive_counts(up)
    result["consecutive_down_bars"] = consecutive_counts(down)
    result["atr_normalized_return"] = safe_ratio(result["close"] - result["open"], result["atr"])
    return result


def classify_momentum(frame: pd.DataFrame, settings: FeatureSettings) -> pd.Series:
    score = frame["atr_normalized_return"].fillna(0)
    score += frame["return_15m"].fillna(0) * 100
    score += (frame["consecutive_up_bars"] - frame["consecutive_down_bars"]) * 0.1
    return pd.Series(
        np.select(
            [
                score >= settings.momentum_strong_threshold,
                score >= settings.momentum_threshold,
                score <= -settings.momentum_strong_threshold,
                score <= -settings.momentum_threshold,
            ],
            ["STRONG_UP", "UP", "STRONG_DOWN", "DOWN"],
            default="NEUTRAL",
        ),
        index=frame.index,
        dtype="string",
    )
