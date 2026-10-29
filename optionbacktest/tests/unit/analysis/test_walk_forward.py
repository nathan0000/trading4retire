from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from spx_0dte_backtest.analysis import (
    FixedSplitSettings,
    SampleLabel,
    WalkForwardRunner,
    WalkForwardSettings,
    split_fixed_samples,
)


def monthly_data(start: str = "2021-01-01", periods: int = 60) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="MS", tz="America/New_York")
    return pd.DataFrame({"entry_timestamp": timestamps, "net_pnl": range(periods)})


def test_fixed_split_produces_disjoint_named_samples() -> None:
    settings = FixedSplitSettings(
        start=date(2021, 1, 1),
        validation_start=date(2024, 1, 1),
        out_of_sample_start=date(2025, 1, 1),
        end=date(2026, 1, 1),
    )

    result = split_fixed_samples(monthly_data(), settings)

    assert len(result.training) == 36
    assert len(result.validation) == 12
    assert len(result.out_of_sample) == 12
    assert result.training["sample"].eq(SampleLabel.IN_SAMPLE).all()
    assert result.validation["sample"].eq(SampleLabel.VALIDATION).all()
    assert result.out_of_sample["sample"].eq(SampleLabel.OUT_OF_SAMPLE).all()
    assert set(result.training.index).isdisjoint(result.validation.index)
    assert set(result.validation.index).isdisjoint(result.out_of_sample.index)


def test_fixed_split_uses_half_open_boundaries() -> None:
    settings = FixedSplitSettings(
        start="2021-01-01",
        validation_start="2024-01-01",
        out_of_sample_start="2025-01-01",
        end="2026-01-01",
    )
    result = split_fixed_samples(monthly_data(), settings)
    assert result.training["entry_timestamp"].max() < pd.Timestamp(
        "2024-01-01", tz="America/New_York"
    )
    assert result.validation["entry_timestamp"].min() == pd.Timestamp(
        "2024-01-01", tz="America/New_York"
    )
    assert result.out_of_sample["entry_timestamp"].min() == pd.Timestamp(
        "2025-01-01", tz="America/New_York"
    )


def test_fixed_split_rejects_invalid_chronology() -> None:
    with pytest.raises(ValueError, match="chronological"):
        FixedSplitSettings(
            start="2024-01-01",
            validation_start="2023-01-01",
            out_of_sample_start="2025-01-01",
            end="2026-01-01",
        )


def test_split_rejects_naive_timestamps() -> None:
    data = monthly_data()
    data["entry_timestamp"] = data["entry_timestamp"].dt.tz_localize(None)
    settings = FixedSplitSettings(
        start="2021-01-01",
        validation_start="2024-01-01",
        out_of_sample_start="2025-01-01",
        end="2026-01-01",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        split_fixed_samples(data, settings)


def test_rolling_windows_are_chronological_and_non_overlapping_tests() -> None:
    runner = WalkForwardRunner(WalkForwardSettings(train_months=12, test_months=3, step_months=3))

    windows = runner.windows(monthly_data(periods=24))

    assert len(windows) == 4
    assert windows[0].train_start == pd.Timestamp("2021-01-01", tz="America/New_York")
    assert windows[0].train_end == windows[0].test_start
    assert windows[0].test_end == pd.Timestamp("2022-04-01", tz="America/New_York")
    assert windows[1].train_start == pd.Timestamp("2021-04-01", tz="America/New_York")
    assert all(
        left.test_end <= right.test_start for left, right in zip(windows, windows[1:], strict=False)
    )


def test_expanding_windows_keep_original_training_start() -> None:
    runner = WalkForwardRunner(
        WalkForwardSettings(
            train_months=12,
            test_months=3,
            step_months=3,
            expanding=True,
        )
    )
    windows = runner.windows(monthly_data(periods=24))
    assert {window.train_start for window in windows} == {
        pd.Timestamp("2021-01-01", tz="America/New_York")
    }
    assert windows[1].train_end == pd.Timestamp("2022-04-01", tz="America/New_York")


def test_runner_fits_only_training_and_evaluates_following_test() -> None:
    observed: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fit(training: pd.DataFrame) -> dict[str, int]:
        return {"maximum_training_value": int(training["net_pnl"].max())}

    def evaluate(test: pd.DataFrame, parameters: dict[str, int]) -> int:
        observed.append((test["entry_timestamp"].min(), test["entry_timestamp"].max()))
        assert test["net_pnl"].min() > parameters["maximum_training_value"]
        return int(test["net_pnl"].sum())

    runner = WalkForwardRunner(WalkForwardSettings(train_months=12, test_months=3, step_months=3))
    results = runner.run(monthly_data(periods=24), fit=fit, evaluate=evaluate)

    assert len(results) == 4
    assert len(observed) == 4
    assert results[0].parameters["maximum_training_value"] == 11
    assert results[0].evaluation == 12 + 13 + 14
    assert results[0].training_rows == 12
    assert results[0].test_rows == 3


def test_callbacks_cannot_mutate_source_or_saved_parameters() -> None:
    data = monthly_data(periods=18)
    original = data.copy(deep=True)

    def fit(training: pd.DataFrame) -> dict[str, int]:
        training.loc[:, "net_pnl"] = -1
        return {"choice": 1}

    def evaluate(test: pd.DataFrame, parameters: dict[str, int]) -> int:
        test.loc[:, "net_pnl"] = -2
        parameters["choice"] = 2
        return 0

    result = WalkForwardRunner(
        WalkForwardSettings(train_months=12, test_months=3, step_months=3)
    ).run(data, fit=fit, evaluate=evaluate)

    pd.testing.assert_frame_equal(data, original)
    assert result[0].parameters["choice"] == 1
    with pytest.raises(TypeError):
        result[0].parameters["choice"] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    "settings",
    [
        WalkForwardSettings(train_months=1, test_months=1, step_months=2),
        WalkForwardSettings(train_months=12, test_months=3, step_months=1),
    ],
)
def test_overlapping_or_gapped_test_windows_are_rejected(
    settings: WalkForwardSettings,
) -> None:
    with pytest.raises(ValueError, match="step_months"):
        WalkForwardRunner(settings)


def test_partial_final_test_window_is_excluded_by_default() -> None:
    data = monthly_data(periods=16)
    runner = WalkForwardRunner(WalkForwardSettings(train_months=12, test_months=3, step_months=3))
    assert len(runner.windows(data)) == 1


def test_missing_or_unsorted_timestamps_are_rejected() -> None:
    runner = WalkForwardRunner(WalkForwardSettings(train_months=12, test_months=3, step_months=3))
    with pytest.raises(ValueError, match="missing required column"):
        runner.windows(pd.DataFrame({"net_pnl": [1]}))
    with pytest.raises(ValueError, match="sorted"):
        runner.windows(monthly_data(periods=18).iloc[::-1])
