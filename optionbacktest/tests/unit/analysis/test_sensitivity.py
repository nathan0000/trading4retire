from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from spx_0dte_backtest.analysis import (
    FixedSamples,
    ParameterGrid,
    SampleLabel,
    SensitivityAnalyzer,
    SensitivitySettings,
    evaluate_selected,
)


def frame(label: SampleLabel, values: list[int]) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "entry_timestamp": timestamps,
            "net_pnl": values,
            "sample": label,
        }
    )


def samples() -> FixedSamples:
    return FixedSamples(
        training=frame(SampleLabel.IN_SAMPLE, [1, 2, 3]),
        validation=frame(SampleLabel.VALIDATION, [100, 200]),
        out_of_sample=frame(SampleLabel.OUT_OF_SAMPLE, [1000]),
    )


def test_one_dimensional_surface_reports_every_value_and_selects_best() -> None:
    grid = ParameterGrid({"stop_multiplier": [0.5, 0.9, 1.25]})

    result = SensitivityAnalyzer().analyze(
        samples(),
        grid,
        lambda _data, parameters: {
            "expectancy": {0.5: 2.0, 0.9: 5.0, 1.25: 3.0}[parameters["stop_multiplier"]],
            "trades": 10.0,
        },
    )

    assert result.surface["stop_multiplier"].tolist() == [0.5, 0.9, 1.25]
    assert result.surface["expectancy"].tolist() == [2.0, 5.0, 3.0]
    assert result.best_parameters == {"stop_multiplier": 0.9}
    assert result.best_objective == pytest.approx(5.0)


def test_two_dimensional_surface_reports_cartesian_product() -> None:
    grid = ParameterGrid(
        {
            "short_delta": [0.10, 0.15],
            "stop_multiplier": [0.75, 0.90, 1.25],
        }
    )
    result = SensitivityAnalyzer().analyze(
        samples(),
        grid,
        lambda _data, parameters: {
            "expectancy": parameters["short_delta"] * 100 - parameters["stop_multiplier"]
        },
    )
    assert len(result.surface) == 6
    assert set(result.surface.columns) >= {
        "short_delta",
        "stop_multiplier",
        "expectancy",
        "possible_overfit",
    }


def test_selection_callback_receives_training_sample_only() -> None:
    observed: list[set[SampleLabel]] = []

    def evaluate(data: pd.DataFrame, _parameters: Mapping[str, object]) -> dict[str, float]:
        observed.append(set(data["sample"]))
        return {"expectancy": float(data["net_pnl"].mean())}

    SensitivityAnalyzer().analyze(samples(), ParameterGrid({"delta": [0.1, 0.15]}), evaluate)
    assert observed == [{SampleLabel.IN_SAMPLE}, {SampleLabel.IN_SAMPLE}]


def test_analysis_does_not_mutate_training_data() -> None:
    source = samples()
    original = source.training.copy(deep=True)

    def evaluate(data: pd.DataFrame, _parameters: Mapping[str, object]) -> dict[str, float]:
        data.loc[:, "net_pnl"] = -999
        return {"expectancy": 1.0}

    SensitivityAnalyzer().analyze(source, ParameterGrid({"delta": [0.1]}), evaluate)
    pd.testing.assert_frame_equal(source.training, original)


def test_isolated_best_point_is_flagged_as_possible_overfit() -> None:
    objectives = {0.5: 1.0, 0.9: 10.0, 1.25: 1.5}
    result = SensitivityAnalyzer(SensitivitySettings(isolation_threshold=0.25)).analyze(
        samples(),
        ParameterGrid({"stop": list(objectives)}),
        lambda _data, parameters: {"expectancy": objectives[parameters["stop"]]},
    )
    assert result.possible_overfit
    assert result.surface.loc[result.surface["stop"] == 0.9, "possible_overfit"].item()


def test_plateau_is_not_flagged_as_possible_overfit() -> None:
    objectives = {0.5: 1.0, 0.9: 10.0, 1.25: 9.5}
    result = SensitivityAnalyzer(SensitivitySettings(isolation_threshold=0.25)).analyze(
        samples(),
        ParameterGrid({"stop": list(objectives)}),
        lambda _data, parameters: {"expectancy": objectives[parameters["stop"]]},
    )
    assert not result.possible_overfit


def test_minimization_objective_is_supported() -> None:
    result = SensitivityAnalyzer(
        SensitivitySettings(objective="max_drawdown", maximize=False)
    ).analyze(
        samples(),
        ParameterGrid({"delta": [0.1, 0.15]}),
        lambda _data, parameters: {"max_drawdown": {0.1: 100.0, 0.15: 80.0}[parameters["delta"]]},
    )
    assert result.best_parameters == {"delta": 0.15}


def test_ties_select_first_grid_point_deterministically() -> None:
    result = SensitivityAnalyzer().analyze(
        samples(),
        ParameterGrid({"delta": [0.1, 0.15]}),
        lambda _data, _parameters: {"expectancy": 1.0},
    )
    assert result.best_parameters == {"delta": 0.1}


def test_selected_parameters_are_immutable_and_surface_is_defensive() -> None:
    result = SensitivityAnalyzer().analyze(
        samples(),
        ParameterGrid({"delta": [0.1]}),
        lambda _data, _parameters: {"expectancy": 1.0},
    )
    with pytest.raises(TypeError):
        result.best_parameters["delta"] = 0.2  # type: ignore[index]
    surface = result.surface
    surface.loc[:, "expectancy"] = 999
    assert result.surface["expectancy"].item() == 1.0


def test_evaluate_selected_uses_fixed_parameters_without_reselection() -> None:
    result = SensitivityAnalyzer().analyze(
        samples(),
        ParameterGrid({"delta": [0.1, 0.15]}),
        lambda _data, parameters: {"expectancy": parameters["delta"]},
    )
    observed: dict[str, object] = {}

    def evaluate(data: pd.DataFrame, parameters: Mapping[str, object]) -> dict[str, float]:
        observed.update(parameters)
        return {"expectancy": float(data["net_pnl"].mean())}

    metrics = evaluate_selected(samples().validation, result, evaluate)
    assert observed == {"delta": 0.15}
    assert metrics["expectancy"] == pytest.approx(150.0)


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"a": [1], "b": [2], "c": [3]},
        {"a": []},
        {"a": [1, 1]},
    ],
)
def test_grid_rejects_empty_large_or_duplicate_dimensions(
    values: dict[str, list[int]],
) -> None:
    with pytest.raises(ValueError):
        ParameterGrid(values)


def test_missing_non_numeric_or_nonfinite_objective_is_rejected() -> None:
    analyzer = SensitivityAnalyzer()
    grid = ParameterGrid({"delta": [0.1]})
    with pytest.raises(ValueError, match="objective"):
        analyzer.analyze(samples(), grid, lambda _data, _parameters: {"trades": 1.0})
    with pytest.raises(ValueError, match="finite numeric"):
        analyzer.analyze(samples(), grid, lambda _data, _parameters: {"expectancy": float("nan")})
