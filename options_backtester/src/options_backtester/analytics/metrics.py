"""Risk/return metrics from an equity curve and a trade log."""

from __future__ import annotations

from math import inf, nan, sqrt

import numpy as np
import pandas as pd


def _periods_per_year(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 252.0
    years = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / (365.25 * 86400)
    return (len(timestamps) - 1) / years if years > 0 else 252.0


def drawdown(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (<= 0)."""
    return equity / equity.cummax() - 1.0


def profit_factor(pnl: pd.Series) -> float:
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if losses == 0:
        return inf if gains > 0 else nan
    return float(gains / losses)


def compute_metrics(
    equity: pd.DataFrame, trades: pd.DataFrame, risk_free_rate: float = 0.0
) -> dict[str, float]:
    """CAGR, Sharpe, Sortino, drawdown, trade statistics and tail risk.

    Return-based ratios use period returns of the mark-to-market equity curve,
    annualized by the observed sampling frequency. VaR/CVaR are historical, at
    95%, reported as positive loss fractions of equity per period.
    """
    metrics: dict[str, float] = {}
    curve = equity["equity"].astype(float).reset_index(drop=True)
    stamps = equity["timestamp"].reset_index(drop=True)
    periods = _periods_per_year(stamps)
    returns = curve.pct_change().dropna()
    years = (stamps.iloc[-1] - stamps.iloc[0]).total_seconds() / (365.25 * 86400)
    growth = curve.iloc[-1] / curve.iloc[0]
    metrics["total_return"] = float(growth - 1.0)
    metrics["cagr"] = float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else nan
    excess = returns - risk_free_rate / periods
    deviation = returns.std(ddof=1) if len(returns) > 1 else nan
    metrics["annual_volatility"] = (
        float(deviation * sqrt(periods)) if deviation == deviation else nan
    )
    metrics["sharpe"] = (
        float(excess.mean() / deviation * sqrt(periods)) if deviation and deviation > 0 else nan
    )
    downside = sqrt(float((np.minimum(excess, 0.0) ** 2).mean())) if len(excess) else nan
    metrics["sortino"] = (
        float(excess.mean() / downside * sqrt(periods)) if downside and downside > 0 else nan
    )
    depth = drawdown(curve)
    metrics["max_drawdown"] = float(depth.min())
    metrics["max_drawdown_dollars"] = float((curve - curve.cummax()).min())
    if len(returns):
        var = float(-np.quantile(returns, 0.05))
        tail = returns[returns <= -var]
        metrics["var_95"] = var
        metrics["cvar_95"] = float(-tail.mean()) if len(tail) else var
    else:
        metrics["var_95"] = metrics["cvar_95"] = nan

    metrics["trades"] = float(len(trades))
    if trades.empty:
        for name in (
            "win_rate",
            "profit_factor",
            "expectancy",
            "avg_win",
            "avg_loss",
            "avg_days_in_trade",
            "net_pnl",
            "trade_var_95",
            "trade_cvar_95",
            "worst_trade",
            "return_on_margin",
        ):
            metrics[name] = nan
        metrics["net_pnl"] = 0.0
        return metrics
    pnl = trades["net_pnl"].astype(float)
    metrics["net_pnl"] = float(pnl.sum())
    metrics["win_rate"] = float((pnl > 0).mean())
    metrics["profit_factor"] = profit_factor(pnl)
    metrics["expectancy"] = float(pnl.mean())
    metrics["avg_win"] = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else nan
    metrics["avg_loss"] = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else nan
    metrics["avg_days_in_trade"] = float(trades["days_in_trade"].mean())
    metrics["worst_trade"] = float(pnl.min())
    cutoff = float(np.quantile(pnl, 0.05))
    metrics["trade_var_95"] = -cutoff
    metrics["trade_cvar_95"] = float(-pnl[pnl <= cutoff].mean())
    margin = trades["margin"].astype(float)
    metrics["return_on_margin"] = float((pnl / margin.where(margin > 0)).mean())
    return metrics
