"""Common immutable strategy inputs and candidate outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Protocol

import pandas as pd

from spx_0dte_backtest.data import OptionQuote


class StrategyType(StrEnum):
    IRON_CONDOR = "IRON_CONDOR"


class RejectionReason(StrEnum):
    NO_OPTION_DATA = "NO_OPTION_DATA"
    NO_VALID_STRIKE = "NO_VALID_STRIKE"
    CREDIT_TOO_LOW = "CREDIT_TOO_LOW"
    CREDIT_TOO_HIGH = "CREDIT_TOO_HIGH"
    MISSING_GREEKS = "MISSING_GREEKS"
    OUTSIDE_TRADING_WINDOW = "OUTSIDE_TRADING_WINDOW"
    REGIME_FILTER = "REGIME_FILTER"
    INCORRECT_EXPIRATION = "INCORRECT_EXPIRATION"


class TradeLegRole(StrEnum):
    LONG_PUT = "LONG_PUT"
    SHORT_PUT = "SHORT_PUT"
    SHORT_CALL = "SHORT_CALL"
    LONG_CALL = "LONG_CALL"


@dataclass(frozen=True, slots=True)
class TradeLeg:
    role: TradeLegRole
    quote: OptionQuote
    quantity: int

    def __post_init__(self) -> None:
        expected = -1 if self.role in {TradeLegRole.SHORT_PUT, TradeLegRole.SHORT_CALL} else 1
        if self.quantity != expected:
            raise ValueError(f"{self.role.value} quantity must be {expected}")


@dataclass(frozen=True, slots=True, init=False)
class MarketContext:
    """Information available to a strategy at one timestamp only."""

    timestamp: pd.Timestamp
    expiration: date
    spx_price: float
    features: Mapping[str, Any]
    _option_quotes: pd.DataFrame = field(repr=False, compare=False)

    def __init__(
        self,
        timestamp: datetime | pd.Timestamp,
        expiration: date,
        spx_price: float,
        option_quotes: pd.DataFrame,
        features: Mapping[str, Any] | None = None,
    ) -> None:
        point_in_time = pd.Timestamp(timestamp)
        if point_in_time.tzinfo is None:
            raise ValueError("market context timestamp must be timezone-aware")
        if not isfinite(spx_price) or spx_price <= 0:
            raise ValueError("SPX price must be finite and positive")
        quotes = option_quotes.copy(deep=True)
        if "timestamp" not in quotes:
            raise ValueError("option quotes must include timestamp")
        if not quotes.empty:
            if quotes["timestamp"].dt.tz is None:
                raise ValueError("option quote timestamps must be timezone-aware")
            if (quotes["timestamp"] > point_in_time).any():
                raise ValueError("market context cannot contain future option quotes")
        known_features = dict(features or {})
        cutoff = known_features.get("data_cutoff_timestamp")
        if cutoff is not None and pd.Timestamp(cutoff) > point_in_time:
            raise ValueError("feature cutoff cannot be later than context timestamp")
        object.__setattr__(self, "timestamp", point_in_time)
        object.__setattr__(self, "expiration", expiration)
        object.__setattr__(self, "spx_price", float(spx_price))
        object.__setattr__(self, "features", MappingProxyType(known_features))
        object.__setattr__(self, "_option_quotes", quotes)

    @property
    def option_quotes(self) -> pd.DataFrame:
        """Return a defensive copy so strategies cannot mutate shared context."""

        return self._option_quotes.copy(deep=True)


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    strategy: StrategyType
    timestamp: pd.Timestamp
    expiration: date
    spx_price: float
    legs: tuple[TradeLeg, ...] = ()
    theoretical_mid_credit: float | None = None
    conservative_executable_credit: float | None = None
    put_credit: float | None = None
    call_credit: float | None = None
    max_profit_points: float | None = None
    max_loss_points: float | None = None
    short_put_delta: float | None = None
    short_call_delta: float | None = None
    rejection_reason: RejectionReason | None = None
    rejection_detail: str | None = None
    context_features: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_features", MappingProxyType(dict(self.context_features)))

    @property
    def is_accepted(self) -> bool:
        return self.rejection_reason is None and len(self.legs) == 4


class Strategy(Protocol):
    def generate_trade(self, context: MarketContext) -> TradeCandidate: ...
