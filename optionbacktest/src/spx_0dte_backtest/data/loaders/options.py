"""Provider-neutral CSV and Parquet loader for historical SPX option snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.data.greeks import GreeksFallbackSettings, fill_missing_greeks
from spx_0dte_backtest.data.validators import Severity, ValidationCode, ValidationIssue
from spx_0dte_backtest.data.validators.data_validation import ValidationReport
from spx_0dte_backtest.data.validators.option_validation import validate_option_quotes
from spx_0dte_backtest.market import SessionService


@dataclass(frozen=True, slots=True)
class OptionLoadResult:
    quotes: pd.DataFrame
    validation: ValidationReport
    source_path: Path


class HistoricalOptionLoader(Protocol):
    def load(
        self,
        source: Path | str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OptionLoadResult: ...


class OptionChainLoader:
    """Normalize historical snapshots without deriving or substituting Greeks."""

    REQUIRED_COLUMNS = {
        "timestamp",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "delta",
        "gamma",
        "theta",
        "vega",
        "implied_volatility",
    }

    def __init__(
        self,
        config: AppConfig,
        sessions: SessionService | None = None,
        *,
        greeks_fallback: GreeksFallbackSettings | None = None,
    ):
        self._config = config
        self._sessions = sessions or SessionService(config.market.calendar, config.project.timezone)
        self._greeks_fallback = greeks_fallback

    def load(
        self,
        source: Path | str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OptionLoadResult:
        path = Path(source)
        raw = self._read(path)
        missing = self.REQUIRED_COLUMNS.difference(raw.columns)
        if missing:
            raise ValueError(f"option data missing required columns: {sorted(missing)}")
        frame, parsing_issues = self._normalize(raw, path)
        frame = self._apply_cutoff(frame, start, end)
        if self._greeks_fallback is not None:
            if "underlying_price" not in raw:
                raise ValueError("Greeks fallback requires an underlying_price column")
            frame["underlying_price"] = pd.to_numeric(
                raw.loc[frame.index, "underlying_price"], errors="coerce"
            )
            frame = fill_missing_greeks(frame, self._greeks_fallback)
        accepted, report = validate_option_quotes(
            frame, self._sessions, initial_issues=tuple(parsing_issues)
        )
        return OptionLoadResult(accepted, report, path.resolve())

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError(f"unsupported option data format: {path.suffix or '<none>'}")

    def _normalize(
        self, raw: pd.DataFrame, path: Path
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        frame = raw.copy()
        parsed: list[pd.Timestamp | pd.NaT] = []
        issues: list[ValidationIssue] = []
        for index, value in raw["timestamp"].items():
            timestamp, issue = self._parse_timestamp(value, int(index))
            parsed.append(timestamp)
            if issue:
                issues.append(issue)
        frame["timestamp"] = pd.DatetimeIndex(parsed)
        frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce").dt.date
        numeric = [
            "strike",
            "bid",
            "ask",
            "last",
            "volume",
            "open_interest",
            "delta",
            "gamma",
            "theta",
            "vega",
            "implied_volatility",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["option_type"] = (
            frame["option_type"].astype("string").str.upper().replace({"C": "CALL", "P": "PUT"})
        )
        if "contract_symbol" not in frame:
            frame["contract_symbol"] = pd.NA
        if "source" not in frame:
            frame["source"] = path.name
        columns = [
            "timestamp",
            "expiration",
            "strike",
            "option_type",
            "bid",
            "ask",
            "last",
            "volume",
            "open_interest",
            "delta",
            "gamma",
            "theta",
            "vega",
            "implied_volatility",
            "contract_symbol",
            "source",
        ]
        return frame[columns], issues

    def _parse_timestamp(
        self, value: Any, index: int
    ) -> tuple[pd.Timestamp | pd.NaT, ValidationIssue | None]:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            timestamp = pd.NaT
        if pd.isna(timestamp):
            return pd.NaT, ValidationIssue(
                ValidationCode.MISSING_TIMESTAMP,
                Severity.ERROR,
                "timestamp is missing or invalid",
                row_number=index,
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
                    row_number=index,
                    timestamp=str(timestamp),
                    field="timestamp",
                    observed_value=value,
                )
            timestamp = timestamp.tz_localize(
                source_timezone, ambiguous="raise", nonexistent="raise"
            )
        return timestamp.tz_convert(self._config.project.timezone), None

    @staticmethod
    def _apply_cutoff(
        frame: pd.DataFrame, start: datetime | None, end: datetime | None
    ) -> pd.DataFrame:
        result = frame
        for name, cutoff in (("start", start), ("end", end)):
            if cutoff is not None and (cutoff.tzinfo is None or cutoff.utcoffset() is None):
                raise ValueError(f"{name} cutoff must be timezone-aware")
        if start is not None:
            result = result[result["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            result = result[result["timestamp"] <= pd.Timestamp(end)]
        return result.copy()
