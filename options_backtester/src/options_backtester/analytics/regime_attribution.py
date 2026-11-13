"""Regime attribution: which market states the strategy earns (or loses) its P&L in.

Trades are grouped by the regime that was known *at entry*. "Ideal" regimes
need a minimum number of trades and a mean trade P&L whose t-statistic clears a
threshold, so a regime is not labelled ideal on two lucky trades or on noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from options_backtester.analytics.metrics import profit_factor

DIMENSIONS = {
    "trend": ["entry_trend"],
    "volatility": ["entry_volatility"],
    "vol_dynamics": ["entry_vol_dynamics"],
    "trend_x_volatility": ["entry_trend", "entry_volatility"],
}


@dataclass
class RegimeAttribution:
    tables: dict[str, pd.DataFrame]
    ideal: pd.DataFrame
    avoid: pd.DataFrame
    min_trades: int
    min_t_stat: float = 1.0

    def ideal_labels(self) -> list[str]:
        return self.ideal["regime"].tolist()

    def avoid_labels(self) -> list[str]:
        return self.avoid["regime"].tolist()


def _summarize(group: pd.DataFrame) -> dict[str, float]:
    pnl = group["net_pnl"].astype(float)
    n = len(pnl)
    deviation = float(pnl.std(ddof=1)) if n > 1 else np.nan
    sharpe = float(pnl.mean() / deviation) if deviation and deviation > 0 else np.nan
    return {
        "trades": n,
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "pnl_std": deviation,
        "trade_sharpe": sharpe,
        "t_stat": sharpe * sqrt(n) if sharpe == sharpe else np.nan,
        "profit_factor": profit_factor(pnl),
        "avg_days_in_trade": float(group["days_in_trade"].mean()),
    }


def attribute(
    trades: pd.DataFrame, min_trades: int = 10, min_t_stat: float = 1.0
) -> RegimeAttribution:
    tables: dict[str, pd.DataFrame] = {}
    for name, keys in DIMENSIONS.items():
        rows = []
        if not trades.empty:
            for label, group in trades.groupby(keys, sort=True):
                parts = label if isinstance(label, tuple) else (label,)
                rows.append({"dimension": name, "regime": "|".join(parts), **_summarize(group)})
        tables[name] = pd.DataFrame(rows)
    combined = pd.concat(tables.values(), ignore_index=True) if tables else pd.DataFrame()
    if combined.empty:
        empty = pd.DataFrame(columns=["dimension", "regime", "trades", "avg_pnl", "t_stat"])
        return RegimeAttribution(tables, empty, empty.copy(), min_trades, min_t_stat)
    enough = combined[combined["trades"] >= min_trades]
    ideal = enough[enough["t_stat"] >= min_t_stat].sort_values("t_stat", ascending=False)
    avoid = enough[enough["t_stat"] <= -min_t_stat].sort_values("t_stat")
    return RegimeAttribution(
        tables,
        ideal.reset_index(drop=True),
        avoid.reset_index(drop=True),
        min_trades,
        min_t_stat,
    )
