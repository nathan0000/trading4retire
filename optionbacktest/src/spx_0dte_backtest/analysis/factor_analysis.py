"""Descriptive attribution of realized P&L to point-in-time factor scores."""

from __future__ import annotations

import pandas as pd

from spx_0dte_backtest.backtest.metrics import grouped_analysis

ATTRIBUTION_COLUMNS = (
    "factor_score",
    "trend_score",
    "volatility_score",
    "sr_score",
    "momentum_score",
    "gamma_score",
    "regime",
)


def factor_performance(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return performance grouped by every Phase 10 score and entry regime.

    This is descriptive attribution only. It does not infer causality and does
    not select, optimize, or rank factor thresholds.
    """

    missing = set(ATTRIBUTION_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    return {column: grouped_analysis(data, column) for column in ATTRIBUTION_COLUMNS}
