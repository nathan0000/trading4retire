"""Same-snapshot, same-expiration SPX option strike selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite

import numpy as np
import pandas as pd

from spx_0dte_backtest.data import OptionQuote, OptionType


class SelectionFailure(StrEnum):
    NO_OPTION_DATA = "NO_OPTION_DATA"
    NO_VALID_STRIKE = "NO_VALID_STRIKE"
    NO_VALID_WING = "NO_VALID_WING"
    INCORRECT_EXPIRATION = "INCORRECT_EXPIRATION"


@dataclass(frozen=True, slots=True)
class StrikeSelectionSettings:
    """Unoptimized initial iron-condor strike-selection parameters."""

    target_abs_delta: float = 0.10
    min_abs_delta: float = 0.08
    max_abs_delta: float = 0.15
    wing_width: float = 30.0

    def __post_init__(self) -> None:
        values = (self.target_abs_delta, self.min_abs_delta, self.max_abs_delta)
        if not all(isfinite(value) and 0 < value < 1 for value in values):
            raise ValueError("delta settings must be finite and between zero and one")
        if not self.min_abs_delta <= self.target_abs_delta <= self.max_abs_delta:
            raise ValueError("target delta must be inside the configured delta range")
        if not isfinite(self.wing_width) or self.wing_width <= 0:
            raise ValueError("wing width must be finite and positive")


@dataclass(frozen=True, slots=True)
class SelectionResult:
    short_put: OptionQuote | None = None
    long_put: OptionQuote | None = None
    short_call: OptionQuote | None = None
    long_call: OptionQuote | None = None
    failure: SelectionFailure | None = None
    detail: str | None = None

    @property
    def is_success(self) -> bool:
        return self.failure is None and len(self.legs) == 4

    @property
    def legs(self) -> tuple[OptionQuote, ...]:
        values = (self.short_put, self.long_put, self.short_call, self.long_call)
        return tuple(leg for leg in values if leg is not None)


class StrikeSelector:
    """Select short strikes by delta and require exact configured wings.

    Selection never falls back to a previous or later snapshot, a different
    expiration, or an approximately spaced wing.
    """

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
        "contract_symbol",
        "source",
    }

    def __init__(self, settings: StrikeSelectionSettings | None = None):
        self.settings = settings or StrikeSelectionSettings()

    def select(
        self,
        quotes: pd.DataFrame,
        timestamp: datetime | pd.Timestamp,
        expiration: date,
        underlying_price: float,
    ) -> SelectionResult:
        self._validate_request(quotes, timestamp, underlying_price)
        requested = pd.Timestamp(timestamp)
        local_date = requested.tz_convert("America/New_York").date()
        if expiration != local_date:
            return SelectionResult(
                failure=SelectionFailure.INCORRECT_EXPIRATION,
                detail="0DTE expiration must equal the New York snapshot date",
            )

        snapshot = quotes[
            (quotes["timestamp"] == requested) & (quotes["expiration"] == expiration)
        ].copy()
        if snapshot.empty:
            return SelectionResult(
                failure=SelectionFailure.NO_OPTION_DATA,
                detail="no quotes at the exact requested timestamp and expiration",
            )

        puts = self._short_candidates(snapshot, OptionType.PUT, underlying_price)
        calls = self._short_candidates(snapshot, OptionType.CALL, underlying_price)
        if puts.empty or calls.empty:
            return SelectionResult(
                failure=SelectionFailure.NO_VALID_STRIKE,
                detail="both put and call short candidates are required",
            )

        put_pair = self._select_pair(puts, snapshot, OptionType.PUT)
        call_pair = self._select_pair(calls, snapshot, OptionType.CALL)
        if put_pair is None or call_pair is None:
            return SelectionResult(
                failure=SelectionFailure.NO_VALID_WING,
                detail="an exact same-snapshot wing is unavailable",
            )
        return SelectionResult(
            short_put=self._to_quote(put_pair[0]),
            long_put=self._to_quote(put_pair[1]),
            short_call=self._to_quote(call_pair[0]),
            long_call=self._to_quote(call_pair[1]),
        )

    def _short_candidates(
        self, snapshot: pd.DataFrame, option_type: OptionType, underlying: float
    ) -> pd.DataFrame:
        signed = snapshot["delta"] < 0 if option_type is OptionType.PUT else snapshot["delta"] > 0
        otm = (
            snapshot["strike"] < underlying
            if option_type is OptionType.PUT
            else snapshot["strike"] > underlying
        )
        absolute_delta = snapshot["delta"].abs()
        return snapshot[
            (snapshot["option_type"] == option_type.value)
            & signed
            & otm
            & absolute_delta.between(
                self.settings.min_abs_delta, self.settings.max_abs_delta, inclusive="both"
            )
        ].copy()

    def _select_pair(
        self, candidates: pd.DataFrame, snapshot: pd.DataFrame, option_type: OptionType
    ) -> tuple[pd.Series, pd.Series] | None:
        ranked = candidates.copy()
        ranked["_delta_distance"] = (
            (ranked["delta"].abs() - self.settings.target_abs_delta).abs().round(12)
        )
        # With equal delta error, prefer the more OTM short: lower put or higher call.
        ranked["_otm_rank"] = (
            ranked["strike"] if option_type is OptionType.PUT else -ranked["strike"]
        )
        ranked = ranked.sort_values(
            ["_delta_distance", "_otm_rank", "strike", "contract_symbol"], kind="stable"
        )
        for _, short in ranked.iterrows():
            wing_strike = short["strike"] + (
                -self.settings.wing_width
                if option_type is OptionType.PUT
                else self.settings.wing_width
            )
            wing = snapshot[
                (snapshot["option_type"] == option_type.value)
                & np.isclose(snapshot["strike"], wing_strike, rtol=0, atol=1e-9)
            ].sort_values(["contract_symbol"], kind="stable")
            if not wing.empty:
                return short, wing.iloc[0]
        return None

    def _validate_request(
        self, quotes: pd.DataFrame, timestamp: datetime | pd.Timestamp, underlying: float
    ) -> None:
        missing = self.REQUIRED_COLUMNS.difference(quotes.columns)
        if missing:
            raise ValueError(f"option quotes missing required columns: {sorted(missing)}")
        requested = pd.Timestamp(timestamp)
        if requested.tzinfo is None:
            raise ValueError("selection timestamp must be timezone-aware")
        if not isfinite(underlying) or underlying <= 0:
            raise ValueError("underlying price must be finite and positive")
        if not quotes.empty and quotes["timestamp"].dt.tz is None:
            raise ValueError("option quote timestamps must be timezone-aware")

    @staticmethod
    def _to_quote(row: pd.Series) -> OptionQuote:
        return OptionQuote(
            timestamp=pd.Timestamp(row["timestamp"]),
            expiration=row["expiration"],
            strike=float(row["strike"]),
            option_type=OptionType(row["option_type"]),
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            last=None if pd.isna(row["last"]) else float(row["last"]),
            volume=None if pd.isna(row["volume"]) else int(row["volume"]),
            open_interest=(None if pd.isna(row["open_interest"]) else int(row["open_interest"])),
            delta=float(row["delta"]),
            gamma=float(row["gamma"]),
            theta=float(row["theta"]),
            vega=float(row["vega"]),
            implied_volatility=float(row["implied_volatility"]),
            contract_symbol=(
                None if pd.isna(row["contract_symbol"]) else str(row["contract_symbol"])
            ),
            source=None if pd.isna(row["source"]) else str(row["source"]),
        )
