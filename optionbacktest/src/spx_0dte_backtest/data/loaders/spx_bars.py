"""CSV and Parquet loader for canonical SPX five-minute bars."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.data.loaders.base import LoadResult
from spx_0dte_backtest.data.validators import (
    Severity,
    ValidationCode,
    ValidationIssue,
    validate_market_bars,
)
from spx_0dte_backtest.market import SessionService


class SpxBarLoader:
    """Load SPX bars without guessing ambiguous source conventions."""

    REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close")

    def __init__(self, config: AppConfig, sessions: SessionService | None = None):
        self._config = config
        self._sessions = sessions or SessionService(config.market.calendar, config.project.timezone)

    def load(
        self,
        source: Path | str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> LoadResult:
        path = Path(source)
        raw = self._read(path)
        canonical, parsing_issues = self._normalize(raw, path)
        canonical = self._apply_cutoff(canonical, start=start, end=end)
        accepted, report = validate_market_bars(
            canonical,
            self._sessions,
            frequency_minutes=self._config.data.expected_frequency_minutes,
            timestamp_semantics=self._config.data.timestamp_semantics,
            initial_issues=tuple(parsing_issues),
        )
        return LoadResult(bars=accepted, validation=report, source_path=path.resolve())

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError(f"unsupported market data format: {suffix or '<none>'}")

    def _normalize(
        self, raw: pd.DataFrame, source_path: Path
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        timestamp_column = self._config.data.timestamp_column
        required = {timestamp_column, *self.REQUIRED_PRICE_COLUMNS}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"source data missing required columns: {sorted(missing)}")

        frame = pd.DataFrame(index=raw.index)
        issues: list[ValidationIssue] = []
        parsed: list[pd.Timestamp | pd.NaT] = []
        for index, value in raw[timestamp_column].items():
            timestamp, issue = self._parse_timestamp(value, int(index))
            parsed.append(timestamp)
            if issue is not None:
                issues.append(issue)

        # All valid timestamps are normalized to one timezone, allowing a stable dtype.
        frame["timestamp"] = pd.DatetimeIndex(parsed)
        for column in self.REQUIRED_PRICE_COLUMNS:
            frame[column] = pd.to_numeric(raw[column], errors="coerce")
        frame["volume"] = (
            pd.to_numeric(raw["volume"], errors="coerce")
            if "volume" in raw
            else pd.Series(pd.NA, index=raw.index, dtype="Float64")
        )
        frame["symbol"] = raw["symbol"].astype(str) if "symbol" in raw else "SPX"
        frame["source"] = raw["source"].astype(str) if "source" in raw else source_path.name
        return frame, issues

    def _parse_timestamp(
        self, value: Any, row_number: int
    ) -> tuple[pd.Timestamp | pd.NaT, ValidationIssue | None]:
        if pd.isna(value) or str(value).strip() == "":
            return pd.NaT, ValidationIssue(
                ValidationCode.MISSING_TIMESTAMP,
                Severity.ERROR,
                "timestamp is missing",
                row_number=row_number,
                field="timestamp",
                observed_value=value,
            )
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return pd.NaT, ValidationIssue(
                ValidationCode.INVALID_TIMESTAMP,
                Severity.ERROR,
                "timestamp cannot be parsed",
                row_number=row_number,
                field="timestamp",
                observed_value=value,
            )
        if timestamp.tzinfo is None:
            source_timezone = self._config.data.source_timezone
            if source_timezone is None:
                return pd.NaT, ValidationIssue(
                    ValidationCode.NAIVE_TIMESTAMP,
                    Severity.ERROR,
                    "naive timestamp requires data.source_timezone",
                    row_number=row_number,
                    timestamp=str(timestamp),
                    field="timestamp",
                    observed_value=value,
                )
            try:
                timestamp = timestamp.tz_localize(
                    source_timezone, ambiguous="raise", nonexistent="raise"
                )
            except (TypeError, ValueError) as exc:
                return pd.NaT, ValidationIssue(
                    ValidationCode.INVALID_TIMEZONE,
                    Severity.ERROR,
                    f"timestamp cannot be localized: {exc}",
                    row_number=row_number,
                    timestamp=str(timestamp),
                    field="timestamp",
                    observed_value=value,
                )
        return timestamp.tz_convert(self._config.project.timezone), None

    @staticmethod
    def _apply_cutoff(
        frame: pd.DataFrame,
        *,
        start: datetime | None,
        end: datetime | None,
    ) -> pd.DataFrame:
        filtered = frame
        for label, value in (("start", start), ("end", end)):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{label} cutoff must be timezone-aware")
        if start is not None:
            filtered = filtered[filtered["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            filtered = filtered[filtered["timestamp"] <= pd.Timestamp(end)]
        return filtered.copy()
