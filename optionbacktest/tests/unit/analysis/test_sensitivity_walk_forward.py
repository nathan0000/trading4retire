from __future__ import annotations

import pandas as pd

from spx_0dte_backtest.analysis import (
    ParameterGrid,
    SensitivityAnalyzer,
    WalkForwardRunner,
    WalkForwardSettings,
)


def test_training_analysis_integrates_with_walk_forward_fit_callback() -> None:
    timestamps = pd.date_range("2021-01-01", periods=19, freq="MS", tz="UTC")
    data = pd.DataFrame({"entry_timestamp": timestamps, "net_pnl": range(19)})
    analyzer = SensitivityAnalyzer()
    grid = ParameterGrid({"delta": [0.1, 0.15]})

    results = WalkForwardRunner(
        WalkForwardSettings(train_months=12, test_months=3, step_months=3)
    ).run(
        data,
        fit=lambda training: (
            analyzer.analyze_training(
                training,
                grid,
                lambda sample, parameters: {
                    "expectancy": float(sample["net_pnl"].mean()) + parameters["delta"]
                },
            ).best_parameters
        ),
        evaluate=lambda test, parameters: float(test["net_pnl"].mean()) + parameters["delta"],
    )

    assert len(results) == 2
    assert all(result.parameters == {"delta": 0.15} for result in results)
