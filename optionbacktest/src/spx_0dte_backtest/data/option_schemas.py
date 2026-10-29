"""Canonical historical option quote records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite


class OptionType(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """A historical quote snapshot; values are never substituted from current data."""

    timestamp: datetime
    expiration: date
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    last: float | None
    volume: int | None
    open_interest: int | None
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    contract_symbol: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not isfinite(self.strike) or self.strike <= 0:
            raise ValueError("strike must be finite and positive")
        if not all(isfinite(value) and value >= 0 for value in (self.bid, self.ask)):
            raise ValueError("bid and ask must be finite and non-negative")
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")
        if self.last is not None and (not isfinite(self.last) or self.last < 0):
            raise ValueError("last must be finite and non-negative")
        if self.volume is not None and self.volume < 0:
            raise ValueError("volume must be non-negative")
        if self.open_interest is not None and self.open_interest < 0:
            raise ValueError("open_interest must be non-negative")
        if not isfinite(self.delta) or not -1 <= self.delta <= 1:
            raise ValueError("delta must be between -1 and 1")
        if not all(isfinite(value) for value in (self.gamma, self.theta, self.vega)):
            raise ValueError("Greeks must be finite")
        if not isfinite(self.implied_volatility) or self.implied_volatility < 0:
            raise ValueError("implied_volatility must be finite and non-negative")
