"""Canonical data records used at package boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite


@dataclass(frozen=True, slots=True)
class MarketBar:
    """A canonical, timezone-aware OHLCV market bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        prices = (self.open, self.high, self.low, self.close)
        if not all(isfinite(value) and value > 0 for value in prices):
            raise ValueError("OHLC prices must be finite and positive")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be at least every other OHLC value")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be at most every other OHLC value")
        if self.volume is not None and (not isfinite(self.volume) or self.volume < 0):
            raise ValueError("volume must be finite and non-negative when provided")
