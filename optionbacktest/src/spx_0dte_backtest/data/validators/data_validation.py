"""Structured validation for canonical five-minute market bars."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

import pandas as pd

from spx_0dte_backtest.market.sessions import SessionService


class Severity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationCode(StrEnum):
    MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    NAIVE_TIMESTAMP = "NAIVE_TIMESTAMP"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    INVALID_TIMEZONE = "INVALID_TIMEZONE"
    OUTSIDE_SESSION = "OUTSIDE_SESSION"
    UNEXPECTED_INTERVAL = "UNEXPECTED_INTERVAL"
    MISSING_BAR = "MISSING_BAR"
    NON_FINITE_PRICE = "NON_FINITE_PRICE"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    INVALID_OHLC = "INVALID_OHLC"
    MARKET_HOLIDAY = "MARKET_HOLIDAY"
    EARLY_CLOSE_VIOLATION = "EARLY_CLOSE_VIOLATION"
    UNSORTED_INPUT = "UNSORTED_INPUT"
    INVALID_VOLUME = "INVALID_VOLUME"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: ValidationCode
    severity: Severity
    message: str
    row_number: int | None = None
    timestamp: str | None = None
    field: str | None = None
    observed_value: Any = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    total_rows: int
    accepted_rows: int
    rejected_rows: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    @property
    def counts_by_code(self) -> dict[str, int]:
        return dict(Counter(issue.code.value for issue in self.issues))


def _issue(
    code: ValidationCode,
    message: str,
    row_number: int | None = None,
    timestamp: object | None = None,
    field: str | None = None,
    observed_value: Any = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=Severity.ERROR,
        message=message,
        row_number=row_number,
        timestamp=None if timestamp is None else str(timestamp),
        field=field,
        observed_value=observed_value,
    )


def validate_market_bars(
    frame: pd.DataFrame,
    sessions: SessionService,
    *,
    frequency_minutes: int = 5,
    timestamp_semantics: str = "bar_start",
    initial_issues: tuple[ValidationIssue, ...] = (),
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate normalized bars and return valid rows plus a structured report.

    Missing-bar detection is bounded by the first and last observed bar per session.
    This permits intentionally partial input ranges without inventing missing data at
    their boundaries while still detecting internal gaps.
    """

    required = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "source"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"canonical frame missing columns: {sorted(missing_columns)}")

    issues = list(initial_issues)
    rejected: set[int] = set()
    working = frame.copy()

    if not working["timestamp"].dropna().is_monotonic_increasing:
        issues.append(_issue(ValidationCode.UNSORTED_INPUT, "timestamps are not sorted"))

    duplicate_mask = working["timestamp"].notna() & working["timestamp"].duplicated(keep=False)
    for index in working.index[duplicate_mask]:
        rejected.add(int(index))
        issues.append(
            _issue(
                ValidationCode.DUPLICATE_TIMESTAMP,
                "duplicate timestamp",
                int(index),
                working.at[index, "timestamp"],
                "timestamp",
            )
        )

    for index, row in working.iterrows():
        row_number = int(index)
        timestamp = row["timestamp"]
        if pd.isna(timestamp):
            rejected.add(row_number)
            if not any(issue.row_number == row_number for issue in initial_issues):
                issues.append(
                    _issue(ValidationCode.MISSING_TIMESTAMP, "timestamp is missing", row_number)
                )
            continue

        local_timestamp = pd.Timestamp(timestamp)
        if local_timestamp.tzinfo is None:
            rejected.add(row_number)
            issues.append(
                _issue(
                    ValidationCode.NAIVE_TIMESTAMP,
                    "timestamp has no timezone",
                    row_number,
                    timestamp,
                    "timestamp",
                )
            )
            continue

        bounds = sessions.bounds(local_timestamp.tz_convert(sessions.timezone).date())
        if bounds is None:
            rejected.add(row_number)
            issues.append(
                _issue(
                    ValidationCode.MARKET_HOLIDAY,
                    "timestamp falls on a non-session date",
                    row_number,
                    timestamp,
                    "timestamp",
                )
            )
        else:
            expected = sessions.expected_timestamps(
                bounds.session_date, frequency_minutes, timestamp_semantics
            )
            if local_timestamp.tz_convert(sessions.timezone) not in expected:
                rejected.add(row_number)
                code = (
                    ValidationCode.EARLY_CLOSE_VIOLATION
                    if bounds.is_early_close and local_timestamp >= bounds.close
                    else ValidationCode.OUTSIDE_SESSION
                )
                issues.append(
                    _issue(
                        code, "timestamp is outside the expected bar grid", row_number, timestamp
                    )
                )

        prices: dict[str, float] = {}
        bad_price = False
        for field in ("open", "high", "low", "close"):
            try:
                value = float(row[field])
            except (TypeError, ValueError):
                value = float("nan")
            prices[field] = value
            if not isfinite(value):
                rejected.add(row_number)
                bad_price = True
                issues.append(
                    _issue(
                        ValidationCode.NON_FINITE_PRICE,
                        "price must be finite",
                        row_number,
                        timestamp,
                        field,
                        row[field],
                    )
                )
            elif value <= 0:
                rejected.add(row_number)
                bad_price = True
                issues.append(
                    _issue(
                        ValidationCode.NEGATIVE_PRICE,
                        "price must be positive",
                        row_number,
                        timestamp,
                        field,
                        value,
                    )
                )
        if not bad_price and (
            prices["high"] < max(prices.values()) or prices["low"] > min(prices.values())
        ):
            rejected.add(row_number)
            issues.append(
                _issue(
                    ValidationCode.INVALID_OHLC,
                    "high/low are inconsistent with OHLC values",
                    row_number,
                    timestamp,
                )
            )

        if pd.notna(row["volume"]):
            try:
                volume = float(row["volume"])
            except (TypeError, ValueError):
                volume = float("nan")
            if not isfinite(volume) or volume < 0:
                rejected.add(row_number)
                issues.append(
                    _issue(
                        ValidationCode.INVALID_VOLUME,
                        "volume must be finite and non-negative",
                        row_number,
                        timestamp,
                        "volume",
                        row["volume"],
                    )
                )

    valid_for_gaps = working.loc[~working.index.isin(rejected)].copy()
    if not valid_for_gaps.empty:
        local = valid_for_gaps["timestamp"].dt.tz_convert(sessions.timezone)
        valid_for_gaps["_session_date"] = local.dt.date
        frequency = pd.Timedelta(f"{frequency_minutes}min")
        for _, group in valid_for_gaps.groupby("_session_date", sort=True):
            observed = pd.DatetimeIndex(group["timestamp"].sort_values())
            observed_local = observed.tz_convert(sessions.timezone)
            expected_between = pd.date_range(
                observed_local.min(), observed_local.max(), freq=frequency
            )
            for missing in expected_between.difference(observed_local):
                issues.append(
                    _issue(
                        ValidationCode.MISSING_BAR,
                        "expected bar is missing between observed bars",
                        timestamp=missing,
                        field="timestamp",
                    )
                )

    accepted = working.loc[~working.index.isin(rejected)].copy()
    accepted = accepted.sort_values("timestamp").reset_index(drop=True)
    report = ValidationReport(
        total_rows=len(working),
        accepted_rows=len(accepted),
        rejected_rows=len(rejected),
        issues=tuple(issues),
    )
    return accepted, report
