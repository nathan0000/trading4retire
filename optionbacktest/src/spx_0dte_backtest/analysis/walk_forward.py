"""Chronological fixed-sample and rolling walk-forward evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeVar

import pandas as pd

EvaluationT = TypeVar("EvaluationT")


class SampleLabel(StrEnum):
    IN_SAMPLE = "IN_SAMPLE"
    VALIDATION = "VALIDATION"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"


def _date(value: date | str) -> date:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        raise ValueError("split boundaries must be timezone-naive calendar dates")
    return parsed.date()


@dataclass(frozen=True, slots=True)
class FixedSplitSettings:
    """Contiguous half-open boundaries for three research samples."""

    start: date | str
    validation_start: date | str
    out_of_sample_start: date | str
    end: date | str
    timestamp_column: str = "entry_timestamp"
    timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        for name in ("start", "validation_start", "out_of_sample_start", "end"):
            object.__setattr__(self, name, _date(getattr(self, name)))
        if not self.timestamp_column:
            raise ValueError("timestamp column cannot be empty")
        if not (self.start < self.validation_start < self.out_of_sample_start < self.end):
            raise ValueError("fixed split boundaries must be strictly chronological")


@dataclass(frozen=True, slots=True)
class FixedSamples:
    training: pd.DataFrame
    validation: pd.DataFrame
    out_of_sample: pd.DataFrame

    def combined(self) -> pd.DataFrame:
        return pd.concat([self.training, self.validation, self.out_of_sample], axis=0).sort_values(
            "entry_timestamp", kind="stable"
        )


def split_fixed_samples(data: pd.DataFrame, settings: FixedSplitSettings) -> FixedSamples:
    """Split trades into disjoint in-sample, validation, and OOS frames."""

    timestamps = _validated_timestamps(data, settings.timestamp_column)
    local = timestamps.dt.tz_convert(settings.timezone)
    boundaries = [
        pd.Timestamp(value).tz_localize(settings.timezone)
        for value in (
            settings.start,
            settings.validation_start,
            settings.out_of_sample_start,
            settings.end,
        )
    ]

    def sample(start: pd.Timestamp, end: pd.Timestamp, label: SampleLabel) -> pd.DataFrame:
        selected = data.loc[(local >= start) & (local < end)].copy(deep=True)
        selected["sample"] = label
        return selected

    return FixedSamples(
        training=sample(boundaries[0], boundaries[1], SampleLabel.IN_SAMPLE),
        validation=sample(boundaries[1], boundaries[2], SampleLabel.VALIDATION),
        out_of_sample=sample(boundaries[2], boundaries[3], SampleLabel.OUT_OF_SAMPLE),
    )


@dataclass(frozen=True, slots=True)
class WalkForwardSettings:
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3
    expanding: bool = False
    timestamp_column: str = "entry_timestamp"
    timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        for name in ("train_months", "test_months", "step_months"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.timestamp_column:
            raise ValueError("timestamp column cannot be empty")


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    sequence: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end == self.test_start < self.test_end:
            raise ValueError("walk-forward window boundaries must be chronological")


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    window: WalkForwardWindow
    parameters: Mapping[str, Any]
    evaluation: Any
    training_rows: int
    test_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


class WalkForwardRunner:
    """Fit on each training window and evaluate only the following test window."""

    def __init__(self, settings: WalkForwardSettings | None = None) -> None:
        self.settings = settings or WalkForwardSettings()
        if self.settings.step_months != self.settings.test_months:
            raise ValueError(
                "step_months must equal test_months to avoid overlapping or gapped test samples"
            )

    def windows(self, data: pd.DataFrame) -> tuple[WalkForwardWindow, ...]:
        timestamps = _validated_timestamps(data, self.settings.timestamp_column)
        if data.empty:
            return ()
        local = timestamps.dt.tz_convert(self.settings.timezone)
        start = local.iloc[0]
        coverage_end = local.iloc[-1] + pd.offsets.MonthBegin(1)
        train_end = start + pd.DateOffset(months=self.settings.train_months)
        result: list[WalkForwardWindow] = []
        sequence = 1
        while train_end + pd.DateOffset(months=self.settings.test_months) <= coverage_end:
            rolling_start = train_end - pd.DateOffset(months=self.settings.train_months)
            result.append(
                WalkForwardWindow(
                    sequence=sequence,
                    train_start=start if self.settings.expanding else rolling_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=train_end + pd.DateOffset(months=self.settings.test_months),
                )
            )
            sequence += 1
            train_end += pd.DateOffset(months=self.settings.step_months)
        return tuple(result)

    def run(
        self,
        data: pd.DataFrame,
        *,
        fit: Callable[[pd.DataFrame], Mapping[str, Any]],
        evaluate: Callable[[pd.DataFrame, dict[str, Any]], EvaluationT],
    ) -> tuple[WalkForwardResult, ...]:
        """Run callbacks on defensive copies and preserve fitted parameters."""

        timestamps = _validated_timestamps(data, self.settings.timestamp_column)
        local = timestamps.dt.tz_convert(self.settings.timezone)
        results: list[WalkForwardResult] = []
        for window in self.windows(data):
            training = data.loc[(local >= window.train_start) & (local < window.train_end)].copy(
                deep=True
            )
            test = data.loc[(local >= window.test_start) & (local < window.test_end)].copy(
                deep=True
            )
            if training.empty or test.empty:
                raise ValueError("walk-forward windows cannot contain empty samples")
            fitted = dict(fit(training.copy(deep=True)))
            saved = dict(fitted)
            evaluation = evaluate(test.copy(deep=True), fitted)
            results.append(
                WalkForwardResult(
                    window=window,
                    parameters=saved,
                    evaluation=evaluation,
                    training_rows=len(training),
                    test_rows=len(test),
                )
            )
        return tuple(results)


def _validated_timestamps(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        raise ValueError(f"missing required column: {column}")
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        raise ValueError(f"{column} must contain pandas timestamps")
    timestamps = data[column]
    if timestamps.dt.tz is None:
        raise ValueError(f"{column} must be timezone-aware")
    if timestamps.isna().any():
        raise ValueError(f"{column} cannot contain missing timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{column} must be sorted chronologically")
    return timestamps
