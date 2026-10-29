"""Orchestration for point-in-time Phase 2 market features."""

from __future__ import annotations

import pandas as pd

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.features.common import session_dates
from spx_0dte_backtest.features.momentum import add_momentum_features, classify_momentum
from spx_0dte_backtest.features.options_positioning import add_unavailable_positioning
from spx_0dte_backtest.features.settings import FeatureSettings
from spx_0dte_backtest.features.support_resistance import (
    add_intraday_levels,
    add_previous_day_levels,
)
from spx_0dte_backtest.features.trend import add_trend_features, classify_trend
from spx_0dte_backtest.features.volatility import add_volatility_features, align_vix


class FeatureEngine:
    """Calculate causal features from validated, canonical market bars."""

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(self, config: AppConfig, settings: FeatureSettings | None = None):
        self._timezone = config.project.timezone
        self._settings = settings or FeatureSettings()

    def compute(self, bars: pd.DataFrame, *, vix: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return features without mutating input or observing a later row."""

        self._validate_input(bars)
        if bars.empty:
            return bars.copy()
        result = bars.copy().reset_index(drop=True)
        result["session_date"] = session_dates(result, self._timezone)

        sessions: list[pd.DataFrame] = []
        for _, group in result.groupby("session_date", sort=False):
            featured = add_trend_features(group.copy(), self._settings)
            featured = add_volatility_features(featured, self._settings)
            featured = add_intraday_levels(featured, self._settings)
            featured = add_momentum_features(featured)
            sessions.append(featured)
        result = pd.concat(sessions).sort_index()
        result = add_previous_day_levels(result)
        result["trend_classification"] = classify_trend(result, self._settings)
        result["momentum_classification"] = classify_momentum(result, self._settings)
        result = align_vix(result, vix)
        result = add_unavailable_positioning(result)
        result["data_cutoff_timestamp"] = result["timestamp"]
        return result.reset_index(drop=True)

    def _validate_input(self, bars: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS.difference(bars.columns)
        if missing:
            raise ValueError(f"market bars missing required columns: {sorted(missing)}")
        if bars.empty:
            return
        if not pd.api.types.is_datetime64_any_dtype(bars["timestamp"]):
            raise ValueError("market bar timestamps must be datetime values")
        if bars["timestamp"].dt.tz is None:
            raise ValueError("market bar timestamps must be timezone-aware")
        if not bars["timestamp"].is_monotonic_increasing:
            raise ValueError("market bar timestamps must be sorted")
        if not bars["timestamp"].is_unique:
            raise ValueError("market bar timestamps must be unique")
