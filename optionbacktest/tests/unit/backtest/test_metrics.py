from __future__ import annotations

import math

import pandas as pd
import pytest

from spx_0dte_backtest.backtest.metrics import (
    calculate_metrics,
    daily_summary,
    delta_analysis,
    entry_time_analysis,
    equity_and_drawdown,
    expected_move_analysis,
)


def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                [
                    "2026-08-10 10:45-04:00",
                    "2026-08-10 11:00-04:00",
                    "2026-08-11 10:45-04:00",
                    "2026-08-12 11:00-04:00",
                    "2026-08-13 10:45-04:00",
                ],
                utc=True,
            ).tz_convert("America/New_York"),
            "exit_timestamp": pd.to_datetime(
                [
                    "2026-08-10 12:00-04:00",
                    "2026-08-10 12:30-04:00",
                    "2026-08-11 13:00-04:00",
                    "2026-08-12 13:00-04:00",
                    "2026-08-13 13:00-04:00",
                ],
                utc=True,
            ).tz_convert("America/New_York"),
            "gross_pnl": [110.0, -40.0, -90.0, 210.0, -40.0],
            "commission": [5.0] * 5,
            "fees": [5.0] * 5,
            "net_pnl": [100.0, -50.0, -100.0, 200.0, -50.0],
            "short_put_delta": [-0.07, -0.09, -0.11, -0.14, -0.17],
            "short_call_delta": [0.07, 0.09, 0.11, 0.14, 0.17],
            "initial_credit": [0.8, 1.1, 1.4, 1.8, 2.4],
            "expected_move_distance": [0.4, 0.6, 0.9, 1.1, 1.6],
        }
    )


def test_overall_metrics_match_hand_calculation() -> None:
    metrics = calculate_metrics(trades())
    assert metrics.total_trades == 5
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 3
    assert metrics.win_rate == pytest.approx(0.4)
    assert metrics.gross_pnl == pytest.approx(150.0)
    assert metrics.net_pnl == pytest.approx(100.0)
    assert metrics.average_trade == pytest.approx(20.0)
    assert metrics.median_trade == pytest.approx(-50.0)
    assert metrics.profit_factor == pytest.approx(300 / 200)
    assert metrics.expectancy == pytest.approx(20.0)


def test_risk_metrics_and_streaks_match_hand_calculation() -> None:
    metrics = calculate_metrics(trades())
    assert metrics.worst_trade == pytest.approx(-100.0)
    assert metrics.worst_day == pytest.approx(-100.0)
    assert metrics.worst_week == pytest.approx(100.0)
    assert metrics.worst_month == pytest.approx(100.0)
    assert metrics.largest_consecutive_loss == pytest.approx(-150.0)
    assert metrics.largest_consecutive_losing_days == 1
    assert metrics.max_drawdown == pytest.approx(150.0)
    assert metrics.max_drawdown_duration == 2
    assert metrics.var_95 >= 50
    assert metrics.cvar_95 >= metrics.var_95


def test_daily_summary_aggregates_overlapping_trades() -> None:
    summary = daily_summary(trades())
    assert len(summary) == 4
    first = summary.iloc[0]
    assert first["trades"] == 2
    assert first["net_pnl"] == pytest.approx(50.0)
    assert first["gross_pnl"] == pytest.approx(70.0)


def test_equity_curve_is_anchored_at_zero_and_drawdown_is_nonpositive() -> None:
    curve = equity_and_drawdown(trades())
    assert curve["equity"].tolist() == pytest.approx([100, 50, -50, 150, 100])
    assert curve["drawdown"].tolist() == pytest.approx([0, -50, -150, 0, -50])


def test_entry_time_breakdown_has_requested_statistics() -> None:
    result = entry_time_analysis(trades())
    row = result.loc[result["entry_time"] == "10:45"].iloc[0]
    assert row["trades"] == 3
    assert row["win_rate"] == pytest.approx(1 / 3)
    assert {"average_pnl", "profit_factor", "max_drawdown"}.issubset(result.columns)


def test_delta_bins_use_actual_short_leg_absolute_delta() -> None:
    result = delta_analysis(trades())
    assert result["trades"].sum() == 5
    assert result.loc[result["delta_bucket"] == "0.05–0.08", "trades"].iloc[0] == 1
    assert result.loc[result["delta_bucket"] == "0.15–0.20", "trades"].iloc[0] == 1


def test_expected_move_bins_follow_specified_boundaries() -> None:
    result = expected_move_analysis(trades())
    assert result["trades"].sum() == 5
    assert result.loc[result["expected_move_bucket"] == "< 0.50 EM", "trades"].iloc[0] == 1
    assert result.loc[result["expected_move_bucket"] == "> 1.50 EM", "trades"].iloc[0] == 1


def test_no_loss_and_zero_variance_edges_are_explicit() -> None:
    frame = trades().iloc[:1].copy()
    metrics = calculate_metrics(frame)
    assert math.isinf(metrics.profit_factor)
    assert math.isnan(metrics.sharpe_ratio)
    assert math.isinf(metrics.sortino_ratio)
    assert metrics.max_drawdown == 0


def test_empty_input_returns_empty_metrics_without_crashing() -> None:
    metrics = calculate_metrics(trades().iloc[0:0])
    assert metrics.total_trades == 0
    assert metrics.net_pnl == 0
    assert math.isnan(metrics.win_rate)


def test_empty_record_iterable_is_supported() -> None:
    metrics = calculate_metrics([])
    assert metrics.total_trades == 0


def test_missing_required_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        calculate_metrics(pd.DataFrame({"net_pnl": [1.0]}))
