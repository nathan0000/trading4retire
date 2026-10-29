"""Structured quality checks for historical SPX option snapshots."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from spx_0dte_backtest.data.validators.data_validation import (
    Severity,
    ValidationIssue,
    ValidationReport,
)
from spx_0dte_backtest.market import SessionService


class OptionValidationCode(StrEnum):
    DUPLICATE_OPTION_QUOTE = "DUPLICATE_OPTION_QUOTE"
    INVALID_OPTION_TYPE = "INVALID_OPTION_TYPE"
    MISSING_BID_ASK = "MISSING_BID_ASK"
    CROSSED_MARKET = "CROSSED_MARKET"
    ZERO_BID = "ZERO_BID"
    NEGATIVE_OPTION_PRICE = "NEGATIVE_OPTION_PRICE"
    INVALID_STRIKE = "INVALID_STRIKE"
    INVALID_SIZE = "INVALID_SIZE"
    MISSING_GREEKS = "MISSING_GREEKS"
    INVALID_GREEKS = "INVALID_GREEKS"
    INVALID_IV = "INVALID_IV"
    EXPIRED_CONTRACT = "EXPIRED_CONTRACT"
    INCORRECT_EXPIRATION = "INCORRECT_EXPIRATION"
    OUTSIDE_SESSION = "OUTSIDE_SESSION"


def _issue(code: StrEnum, message: str, index: int, row: pd.Series) -> ValidationIssue:
    # ValidationIssue accepts enum-like codes at runtime and serializes their stable value.
    return ValidationIssue(
        code=code,  # type: ignore[arg-type]
        severity=Severity.ERROR,
        message=message,
        row_number=index,
        timestamp=None if pd.isna(row["timestamp"]) else str(row["timestamp"]),
    )


def validate_option_quotes(
    frame: pd.DataFrame,
    sessions: SessionService,
    *,
    initial_issues: tuple[ValidationIssue, ...] = (),
    require_same_day_expiration: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    issues = list(initial_issues)
    rejected = {issue.row_number for issue in initial_issues if issue.row_number is not None}
    working = frame.copy()
    identity = ["timestamp", "expiration", "strike", "option_type", "contract_symbol"]
    duplicates = working[identity].notna().all(axis=1) & working.duplicated(identity, keep=False)

    for index, row in working.iterrows():
        idx = int(index)
        if duplicates.loc[index]:
            rejected.add(idx)
            issues.append(
                _issue(
                    OptionValidationCode.DUPLICATE_OPTION_QUOTE,
                    "duplicate contract snapshot",
                    idx,
                    row,
                )
            )
        if pd.isna(row["timestamp"]):
            continue
        timestamp = pd.Timestamp(row["timestamp"])
        local = timestamp.tz_convert(sessions.timezone)
        if sessions.containing_session(timestamp) is None:
            rejected.add(idx)
            issues.append(
                _issue(
                    OptionValidationCode.OUTSIDE_SESSION,
                    "quote is outside the cash session",
                    idx,
                    row,
                )
            )
        expiration = row["expiration"]
        if pd.isna(expiration):
            rejected.add(idx)
            issues.append(
                _issue(OptionValidationCode.INCORRECT_EXPIRATION, "expiration is missing", idx, row)
            )
        elif expiration < local.date():
            rejected.add(idx)
            issues.append(
                _issue(
                    OptionValidationCode.EXPIRED_CONTRACT, "contract expired before quote", idx, row
                )
            )
        elif require_same_day_expiration and expiration != local.date():
            rejected.add(idx)
            issues.append(
                _issue(OptionValidationCode.INCORRECT_EXPIRATION, "contract is not 0DTE", idx, row)
            )
        if row["option_type"] not in {"CALL", "PUT"}:
            rejected.add(idx)
            issues.append(
                _issue(
                    OptionValidationCode.INVALID_OPTION_TYPE,
                    "option type must be CALL or PUT",
                    idx,
                    row,
                )
            )
        if not np.isfinite(row["strike"]) or row["strike"] <= 0:
            rejected.add(idx)
            issues.append(
                _issue(OptionValidationCode.INVALID_STRIKE, "strike must be positive", idx, row)
            )
        if pd.isna(row["bid"]) or pd.isna(row["ask"]):
            rejected.add(idx)
            issues.append(
                _issue(OptionValidationCode.MISSING_BID_ASK, "bid and ask are required", idx, row)
            )
        else:
            if row["bid"] < 0 or row["ask"] < 0:
                rejected.add(idx)
                issues.append(
                    _issue(
                        OptionValidationCode.NEGATIVE_OPTION_PRICE,
                        "option prices cannot be negative",
                        idx,
                        row,
                    )
                )
            elif row["bid"] == 0:
                rejected.add(idx)
                issues.append(
                    _issue(OptionValidationCode.ZERO_BID, "zero bid is not executable", idx, row)
                )
            if row["bid"] > row["ask"]:
                rejected.add(idx)
                issues.append(
                    _issue(OptionValidationCode.CROSSED_MARKET, "bid exceeds ask", idx, row)
                )
        for field in ("volume", "open_interest"):
            if pd.notna(row[field]) and (row[field] < 0 or float(row[field]).is_integer() is False):
                rejected.add(idx)
                issues.append(
                    _issue(
                        OptionValidationCode.INVALID_SIZE,
                        f"{field} must be a non-negative integer",
                        idx,
                        row,
                    )
                )
        greeks = row[["delta", "gamma", "theta", "vega"]]
        if greeks.isna().any():
            rejected.add(idx)
            issues.append(
                _issue(
                    OptionValidationCode.MISSING_GREEKS, "historical Greeks are required", idx, row
                )
            )
        elif not np.isfinite(greeks.astype(float)).all() or not -1 <= row["delta"] <= 1:
            rejected.add(idx)
            issues.append(
                _issue(OptionValidationCode.INVALID_GREEKS, "Greeks are invalid", idx, row)
            )
        if pd.isna(row["implied_volatility"]) or row["implied_volatility"] < 0:
            rejected.add(idx)
            issues.append(
                _issue(OptionValidationCode.INVALID_IV, "implied volatility is invalid", idx, row)
            )

    accepted = (
        working.loc[~working.index.isin(rejected)].sort_values(identity).reset_index(drop=True)
    )
    return accepted, ValidationReport(
        total_rows=len(working),
        accepted_rows=len(accepted),
        rejected_rows=len(rejected),
        issues=tuple(issues),
    )
