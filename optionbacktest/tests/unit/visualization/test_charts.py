from __future__ import annotations

from pathlib import Path

import pandas as pd

from spx_0dte_backtest.analysis import SensitivityResult
from spx_0dte_backtest.visualization import ChartGenerator, ChartSettings


def trades() -> pd.DataFrame:
    entry = pd.date_range("2025-01-02 10:45", periods=8, freq="31D", tz="America/New_York")
    pnl = [100.0, -50.0, 75.0, -25.0, 120.0, -80.0, 40.0, 60.0]
    return pd.DataFrame(
        {
            "entry_timestamp": entry,
            "exit_timestamp": entry + pd.Timedelta(hours=2),
            "gross_pnl": [value + 10 for value in pnl],
            "commission": [5.0] * 8,
            "fees": [5.0] * 8,
            "net_pnl": pnl,
            "short_put_delta": [-0.08, -0.09, -0.10, -0.11, -0.12, -0.13, -0.14, -0.15],
            "short_call_delta": [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
            "expected_move_distance": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            "vix_regime": ["LOW", "LOW", "NORMAL", "NORMAL", "HIGH", "HIGH", "LOW", "NORMAL"],
            "factor_score": [4, 2, 5, 0, 6, -2, 3, 4],
            "spx_entry": [6000, 6010, 6020, 6005, 6040, 6025, 6060, 6070],
            "spx_exit": [6005, 5990, 6030, 5995, 6060, 6000, 6070, 6080],
        }
    )


def sensitivity() -> SensitivityResult:
    surface = pd.DataFrame(
        {
            "delta": [0.10, 0.10, 0.15, 0.15],
            "stop": [0.75, 0.90, 0.75, 0.90],
            "expectancy": [10.0, 15.0, 8.0, 12.0],
            "possible_overfit": [False] * 4,
        }
    )
    return SensitivityResult(surface, {"delta": 0.10, "stop": 0.90}, 15.0, False, "expectancy")


def test_generate_all_creates_major_backtest_charts(tmp_path: Path) -> None:
    manifest = ChartGenerator(ChartSettings(dpi=72)).generate_all(
        trades(), tmp_path, sensitivity=sensitivity()
    )

    expected = {
        "equity_curve",
        "drawdown_curve",
        "pnl_by_entry_time",
        "pnl_by_vix_regime",
        "pnl_by_delta",
        "pnl_by_expected_move",
        "pnl_distribution",
        "factor_score_performance",
        "monthly_pnl_heatmap",
        "entry_exit_examples",
        "sensitivity_surface",
    }
    assert set(manifest.generated) == expected
    assert manifest.skipped == {}
    assert all(path.exists() and path.stat().st_size > 0 for path in manifest.generated.values())


def test_missing_optional_inputs_are_reported_not_fabricated(tmp_path: Path) -> None:
    minimal = trades().drop(
        columns=[
            "vix_regime",
            "expected_move_distance",
            "factor_score",
            "spx_exit",
        ]
    )
    manifest = ChartGenerator().generate_all(minimal, tmp_path)
    assert "equity_curve" in manifest.generated
    assert set(manifest.skipped) == {
        "pnl_by_vix_regime",
        "pnl_by_expected_move",
        "factor_score_performance",
        "entry_exit_examples",
        "sensitivity_surface",
    }
    assert "vix_regime" in manifest.skipped["pnl_by_vix_regime"]


def test_chart_generation_does_not_mutate_trade_data(tmp_path: Path) -> None:
    source = trades()
    original = source.copy(deep=True)
    ChartGenerator().generate_all(source, tmp_path)
    pd.testing.assert_frame_equal(source, original)


def test_invalid_chart_settings_are_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        ChartSettings(dpi=0)
