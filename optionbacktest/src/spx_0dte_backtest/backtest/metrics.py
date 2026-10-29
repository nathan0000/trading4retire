"""Deterministic Phase 8 P&L, risk, drawdown, and grouped metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from math import inf, nan, sqrt

import numpy as np
import pandas as pd

from spx_0dte_backtest.backtest.position import TradeRecord

NORMALIZED_COLUMNS = [
    "entry_timestamp",
    "exit_timestamp",
    "gross_pnl",
    "commission",
    "fees",
    "net_pnl",
    "initial_credit",
    "exit_debit",
    "exit_reason",
    "strategy",
    "spx_entry",
    "short_put_delta",
    "short_call_delta",
    "regime",
    "trend_score",
    "volatility_score",
    "sr_score",
    "momentum_score",
    "gamma_score",
    "factor_score",
    "factor_score_version",
    "vix",
    "expected_move",
    "expected_move_distance",
]

REQUIRED_COLUMNS = {
    "entry_timestamp",
    "exit_timestamp",
    "gross_pnl",
    "commission",
    "fees",
    "net_pnl",
}


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = nan
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    average_trade: float = nan
    median_trade: float = nan
    profit_factor: float = nan
    expectancy: float = nan
    sharpe_ratio: float = nan
    sortino_ratio: float = nan
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    worst_trade: float = nan
    worst_day: float = nan
    worst_week: float = nan
    worst_month: float = nan
    largest_consecutive_loss: float = 0.0
    largest_consecutive_losing_days: int = 0
    var_95: float = nan
    cvar_95: float = nan

    def as_dict(self) -> dict[str, int | float]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def records_to_frame(records: Iterable[TradeRecord]) -> pd.DataFrame:
    """Normalize completed records without discarding traceable strategy fields."""

    rows: list[dict[str, object]] = []
    for record in records:
        candidate = record.candidate
        features = candidate.context_features
        rows.append(
            {
                "entry_timestamp": record.entry_timestamp,
                "exit_timestamp": record.exit_timestamp,
                "gross_pnl": record.gross_pnl,
                "commission": record.commission,
                "fees": record.fees,
                "net_pnl": record.net_pnl,
                "initial_credit": record.initial_credit,
                "exit_debit": record.exit_debit,
                "exit_reason": record.exit_reason.value,
                "strategy": candidate.strategy.value,
                "spx_entry": candidate.spx_price,
                "short_put_delta": candidate.short_put_delta,
                "short_call_delta": candidate.short_call_delta,
                "regime": features.get("regime"),
                "trend_score": features.get("trend_score"),
                "volatility_score": features.get("volatility_score"),
                "sr_score": features.get("sr_score"),
                "momentum_score": features.get("momentum_score"),
                "gamma_score": features.get("gamma_score"),
                "factor_score": features.get("factor_score"),
                "factor_score_version": features.get("factor_score_version"),
                "vix": features.get("vix"),
                "expected_move": features.get("expected_move"),
                "expected_move_distance": features.get("expected_move_distance"),
            }
        )
    return pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)


def calculate_metrics(data: pd.DataFrame | Iterable[TradeRecord]) -> BacktestMetrics:
    frame = _frame(data)
    if frame.empty:
        return BacktestMetrics()
    pnl = frame["net_pnl"].astype(float)
    daily = daily_summary(frame)
    equity = _trade_equity(frame)
    positive = pnl[pnl > 0].sum()
    negative = pnl[pnl < 0].sum()
    profit_factor = (
        inf
        if negative == 0 and positive > 0
        else (positive / abs(negative) if negative < 0 else nan)
    )
    daily_pnl = daily["net_pnl"].astype(float)
    sharpe = _sharpe(daily_pnl)
    sortino = _sortino(daily_pnl)
    threshold = float(pnl.quantile(0.05))
    tail = pnl[pnl <= threshold]
    return BacktestMetrics(
        total_trades=len(frame),
        winning_trades=int((pnl > 0).sum()),
        losing_trades=int((pnl < 0).sum()),
        win_rate=float((pnl > 0).sum() / len(frame)),
        gross_pnl=float(frame["gross_pnl"].sum()),
        net_pnl=float(pnl.sum()),
        average_trade=float(pnl.mean()),
        median_trade=float(pnl.median()),
        profit_factor=float(profit_factor),
        expectancy=float(pnl.mean()),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=float(-equity["drawdown"].min()),
        max_drawdown_duration=_max_drawdown_duration(equity["drawdown"]),
        worst_trade=float(pnl.min()),
        worst_day=float(daily_pnl.min()),
        worst_week=float(_period_pnl(frame, "W").min()),
        worst_month=float(_period_pnl(frame, "M").min()),
        largest_consecutive_loss=_largest_consecutive_loss(pnl),
        largest_consecutive_losing_days=_largest_losing_run(daily_pnl),
        var_95=max(0.0, -threshold),
        cvar_95=max(0.0, -float(tail.mean())),
    )


def daily_summary(data: pd.DataFrame | Iterable[TradeRecord]) -> pd.DataFrame:
    frame = _frame(data)
    columns = ["date", "trades", "gross_pnl", "commission", "fees", "net_pnl"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    local = frame["exit_timestamp"].dt.tz_convert("America/New_York")
    working = frame.assign(date=local.dt.date)
    result = working.groupby("date", sort=True).agg(
        trades=("net_pnl", "size"),
        gross_pnl=("gross_pnl", "sum"),
        commission=("commission", "sum"),
        fees=("fees", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    return result.reset_index()[columns]


def equity_and_drawdown(data: pd.DataFrame | Iterable[TradeRecord]) -> pd.DataFrame:
    frame = _frame(data)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "net_pnl", "equity", "drawdown"])
    return _trade_equity(frame)


def entry_time_analysis(data: pd.DataFrame | Iterable[TradeRecord]) -> pd.DataFrame:
    frame = _frame(data)
    if frame.empty:
        return _empty_breakdown("entry_time")
    values = frame["entry_timestamp"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    schedule = pd.date_range("2000-01-01 10:45", "2000-01-01 15:15", freq="15min").strftime("%H:%M")
    labels = pd.Series(pd.Categorical(values, categories=schedule, ordered=True), index=frame.index)
    return _group_metrics(frame, labels, "entry_time", include_empty=True)


def weekday_analysis(data: pd.DataFrame | Iterable[TradeRecord]) -> pd.DataFrame:
    frame = _frame(data)
    if frame.empty:
        return _empty_breakdown("weekday")
    labels = frame["entry_timestamp"].dt.tz_convert("America/New_York").dt.day_name()
    return _group_metrics(frame, labels, "weekday")


def delta_analysis(data: pd.DataFrame | Iterable[TradeRecord]) -> pd.DataFrame:
    frame = _frame(data)
    _require(frame, {"short_put_delta", "short_call_delta"})
    actual = pd.concat(
        [frame["short_put_delta"].abs(), frame["short_call_delta"].abs()], axis=1
    ).mean(axis=1)
    labels = pd.cut(
        actual,
        bins=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        labels=["0.05–0.08", "0.08–0.10", "0.10–0.12", "0.12–0.15", "0.15–0.20"],
        include_lowest=True,
        right=True,
    )
    return _group_metrics(frame, labels, "delta_bucket", include_empty=True)


def credit_analysis(data: pd.DataFrame | Iterable[TradeRecord], bins: list[float]) -> pd.DataFrame:
    frame = _frame(data)
    _require(frame, {"initial_credit"})
    labels = pd.cut(frame["initial_credit"], bins=bins, include_lowest=True)
    return _group_metrics(frame, labels, "credit_bucket")


def expected_move_analysis(data: pd.DataFrame | Iterable[TradeRecord]) -> pd.DataFrame:
    frame = _frame(data)
    _require(frame, {"expected_move_distance"})
    labels = pd.cut(
        frame["expected_move_distance"],
        bins=[-np.inf, 0.50, 0.75, 1.00, 1.25, 1.50, np.inf],
        labels=[
            "< 0.50 EM",
            "0.50–0.75 EM",
            "0.75–1.00 EM",
            "1.00–1.25 EM",
            "1.25–1.50 EM",
            "> 1.50 EM",
        ],
        right=False,
    )
    return _group_metrics(frame, labels, "expected_move_bucket", include_empty=True)


def grouped_analysis(data: pd.DataFrame | Iterable[TradeRecord], column: str) -> pd.DataFrame:
    """Analyze enriched future fields such as VIX or regime without guessing them."""

    frame = _frame(data)
    _require(frame, {column})
    return _group_metrics(frame, frame[column], column)


def _frame(data: pd.DataFrame | Iterable[TradeRecord]) -> pd.DataFrame:
    frame = data.copy() if isinstance(data, pd.DataFrame) else records_to_frame(data)
    _require(frame, REQUIRED_COLUMNS)
    if frame.empty:
        return frame
    result = frame.copy()
    for column in ("entry_timestamp", "exit_timestamp"):
        if not pd.api.types.is_datetime64_any_dtype(result[column]) or result[column].dt.tz is None:
            raise ValueError(f"{column} must be timezone-aware datetime data")
    return result.sort_values(["exit_timestamp", "entry_timestamp"], kind="stable").reset_index(
        drop=True
    )


def _require(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def _trade_equity(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["exit_timestamp", "entry_timestamp"], kind="stable")
    equity = ordered["net_pnl"].astype(float).cumsum()
    peaks = equity.cummax().clip(lower=0)
    return pd.DataFrame(
        {
            "timestamp": ordered["exit_timestamp"].to_numpy(),
            "net_pnl": ordered["net_pnl"].to_numpy(),
            "equity": equity.to_numpy(),
            "drawdown": (equity - peaks).to_numpy(),
        }
    )


def _period_pnl(frame: pd.DataFrame, frequency: str) -> pd.Series:
    local = frame["exit_timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return frame["net_pnl"].groupby(local.dt.to_period(frequency)).sum()


def _sharpe(values: pd.Series) -> float:
    if len(values) < 2 or values.std(ddof=1) == 0:
        return nan
    return float(values.mean() / values.std(ddof=1) * sqrt(252))


def _sortino(values: pd.Series) -> float:
    downside = values[values < 0]
    if downside.empty:
        return inf if values.mean() > 0 else nan
    deviation = sqrt(float((downside**2).mean()))
    return float(values.mean() / deviation * sqrt(252)) if deviation else nan


def _largest_consecutive_loss(values: pd.Series) -> float:
    worst = 0.0
    current = 0.0
    for value in values:
        current = current + float(value) if value < 0 else 0.0
        worst = min(worst, current)
    return worst


def _largest_losing_run(values: pd.Series) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value < 0 else 0
        best = max(best, current)
    return best


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    best = current = 0
    for value in drawdown:
        current = current + 1 if value < 0 else 0
        best = max(best, current)
    return best


def _group_metrics(
    frame: pd.DataFrame,
    labels: pd.Series,
    name: str,
    *,
    include_empty: bool = False,
) -> pd.DataFrame:
    working = frame.assign(_group=labels)
    rows: list[dict[str, object]] = []
    observed = False if include_empty else True
    for label, group in working.groupby("_group", observed=observed, sort=True):
        pnl = group["net_pnl"].astype(float)
        positive, negative = pnl[pnl > 0].sum(), pnl[pnl < 0].sum()
        factor = (
            inf
            if negative == 0 and positive > 0
            else (positive / abs(negative) if negative < 0 else nan)
        )
        drawdown = _trade_equity(group)["drawdown"] if not group.empty else pd.Series(dtype=float)
        rows.append(
            {
                name: str(label),
                "trades": len(group),
                "win_rate": float((pnl > 0).mean()) if len(group) else nan,
                "average_pnl": float(pnl.mean()) if len(group) else nan,
                "profit_factor": float(factor),
                "max_drawdown": float(-drawdown.min()) if len(group) else nan,
                "tail_loss": float(pnl.quantile(0.05)) if len(group) else nan,
            }
        )
    return pd.DataFrame(rows)


def _empty_breakdown(name: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            name,
            "trades",
            "win_rate",
            "average_pnl",
            "profit_factor",
            "max_drawdown",
            "tail_loss",
        ]
    )
