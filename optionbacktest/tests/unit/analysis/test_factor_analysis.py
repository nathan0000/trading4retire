import pandas as pd
import pytest

from spx_0dte_backtest.analysis import factor_performance


def trades() -> pd.DataFrame:
    timestamps = pd.date_range("2026-08-10 10:45", periods=4, freq="1D", tz="America/New_York")
    return pd.DataFrame(
        {
            "entry_timestamp": timestamps,
            "exit_timestamp": timestamps + pd.Timedelta("1h"),
            "gross_pnl": [110, -90, 210, -40],
            "commission": [5] * 4,
            "fees": [5] * 4,
            "net_pnl": [100, -100, 200, -50],
            "factor_score": [8, -5, 8, 0],
            "trend_score": [2, -1, 2, 0],
            "volatility_score": [2, -2, 2, 0],
            "sr_score": [2, -2, 2, 0],
            "momentum_score": [2, -2, 2, 0],
            "gamma_score": [0, 2, 0, 0],
            "regime": ["RANGE", "TREND_UP", "RANGE", "UNKNOWN"],
        }
    )


def test_factor_performance_returns_each_component_and_regime() -> None:
    result = factor_performance(trades())
    assert set(result) == {
        "factor_score",
        "trend_score",
        "volatility_score",
        "sr_score",
        "momentum_score",
        "gamma_score",
        "regime",
    }
    range_row = result["regime"].loc[result["regime"]["regime"] == "RANGE"].iloc[0]
    assert range_row["trades"] == 2
    assert range_row["average_pnl"] == pytest.approx(150)


def test_missing_attribution_columns_are_not_fabricated() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        factor_performance(trades().drop(columns=["gamma_score"]))
