"""Canonical schemas and the abstract ``DataProvider``.

Every timestamp in the canonical format is a UTC "known-at" time: the instant at
which the row's contents were fully observable. A daily bar is therefore stamped
at its session close, never at midnight or at the session open. Downstream layers
may use any row whose timestamp is ``<= current_timestamp`` without lookahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from options_backtester.data.cache import ParquetCache
    from options_backtester.data.greeks_engine import GreeksConfig

MARKET_TZ = "America/New_York"
MARKET_CLOSE = time(16, 0)


class OptionType(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


class DataUnavailableError(RuntimeError):
    """The provider cannot supply the requested data (not a transient failure)."""


BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

GREEK_COLUMNS = ["delta", "gamma", "theta", "vega"]

CHAIN_COLUMNS = [
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "iv_rank",
    "iv_percentile",
    "expiration",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
    *GREEK_COLUMNS,
    "greeks_source",
]

_CHAIN_REQUIRED = [
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
]
_CHAIN_FLOATS = [
    "underlying_price",
    "iv_rank",
    "iv_percentile",
    "strike",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
    *GREEK_COLUMNS,
]


def to_utc(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"naive timestamp {value!r}: canonical timestamps must be tz-aware")
    return stamp.tz_convert("UTC")


def market_date(timestamp: pd.Timestamp) -> date:
    """Exchange-local calendar date of a canonical UTC timestamp."""
    return to_utc(timestamp).tz_convert(MARKET_TZ).date()


def market_close_utc(day: date | pd.Timestamp) -> pd.Timestamp:
    """UTC timestamp of the regular-session close on ``day``."""
    local = pd.Timestamp.combine(pd.Timestamp(day).date(), MARKET_CLOSE).tz_localize(MARKET_TZ)
    return local.tz_convert("UTC")


def _utc_series(values: pd.Series) -> pd.Series:
    series = pd.to_datetime(values)
    if series.dt.tz is None:
        raise ValueError("naive timestamps: canonical timestamps must be tz-aware")
    return series.dt.tz_convert("UTC")


def normalize_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce provider bars to the canonical ``UnderlyingBar`` frame."""
    missing = {"timestamp", "open", "high", "low", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    out = frame.copy()
    out["timestamp"] = _utc_series(out["timestamp"])
    out["symbol"] = symbol
    if "volume" not in out:
        out["volume"] = np.nan
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    out = out.dropna(subset=["close"])
    out = out.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return out[BAR_COLUMNS].reset_index(drop=True)


def normalize_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider option chain to the canonical ``OptionChainSnapshot`` frame."""
    missing = set(_CHAIN_REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"option chain missing columns: {sorted(missing)}")
    out = frame.copy()
    for column in CHAIN_COLUMNS:
        if column not in out:
            out[column] = "PROVIDER" if column == "greeks_source" else np.nan
    out["timestamp"] = _utc_series(out["timestamp"])
    out["expiration"] = pd.to_datetime(out["expiration"]).dt.tz_localize(None).dt.normalize()
    out["option_type"] = (
        out["option_type"].astype(str).str.upper().replace({"C": "CALL", "P": "PUT"})
    )
    bad = ~out["option_type"].isin(["CALL", "PUT"])
    if bad.any():
        raise ValueError(
            f"invalid option_type values: {sorted(out.loc[bad, 'option_type'].unique())}"
        )
    for column in _CHAIN_FLOATS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    out["mid"] = out["mid"].fillna((out["bid"] + out["ask"]) / 2.0)
    local_day = out["timestamp"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None).dt.normalize()
    out["dte"] = (out["expiration"] - local_day).dt.days.astype(int)
    out["greeks_source"] = out["greeks_source"].fillna("PROVIDER").astype(str)
    out = out[(out["strike"] > 0) & (out["dte"] >= 0)]
    out = out.drop_duplicates(["timestamp", "expiration", "strike", "option_type"], keep="last")
    out = out.sort_values(["timestamp", "expiration", "option_type", "strike"])
    return out[CHAIN_COLUMNS].reset_index(drop=True)


ContractKey = tuple[pd.Timestamp, float, str]


@dataclass(frozen=True)
class OptionChainSnapshot:
    """All quotes for one underlying at one known-at timestamp."""

    timestamp: pd.Timestamp
    underlying_symbol: str
    underlying_price: float
    quotes: pd.DataFrame
    iv_rank: float | None = None
    iv_percentile: float | None = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> OptionChainSnapshot:
        if frame.empty:
            raise ValueError("cannot build a snapshot from an empty chain")
        stamps = frame["timestamp"].unique()
        if len(stamps) != 1:
            raise ValueError("a snapshot must contain exactly one timestamp")
        first = frame.iloc[0]

        def optional(name: str) -> float | None:
            value = first[name]
            return None if pd.isna(value) else float(value)

        return cls(
            timestamp=pd.Timestamp(first["timestamp"]),
            underlying_symbol=str(first["underlying_symbol"]),
            underlying_price=float(first["underlying_price"]),
            quotes=frame.reset_index(drop=True),
            iv_rank=optional("iv_rank"),
            iv_percentile=optional("iv_percentile"),
        )

    @property
    def date(self) -> date:
        return market_date(self.timestamp)

    @cached_property
    def _index(self) -> dict[ContractKey, int]:
        q = self.quotes
        return {
            key: i
            for i, key in enumerate(
                zip(q["expiration"], q["strike"], q["option_type"], strict=True)
            )
        }

    def lookup(self, expiration: date | pd.Timestamp, strike: float, option_type: str):
        """Return the quote row for one contract, or ``None`` if it is not quoted."""
        position = self._index.get((pd.Timestamp(expiration), float(strike), str(option_type)))
        return None if position is None else self.quotes.iloc[position]


class DataProvider(ABC):
    """Normalizes one data source into canonical frames, cache-first.

    Subclasses implement only ``_fetch_bars`` and ``_fetch_chain``. Normalization,
    the Greeks/IV fallback and the Parquet cache are applied here so that every
    adapter behaves identically downstream.
    """

    name: str = "abstract"
    #: True when option prices are modelled rather than observed.
    synthetic: bool = False

    def __init__(
        self,
        cache: ParquetCache | None = None,
        greeks: GreeksConfig | None = None,
    ) -> None:
        from options_backtester.data.greeks_engine import GreeksConfig

        self.cache = cache
        self.greeks = greeks or GreeksConfig()
        self._chain_memo: dict[tuple[str, date], OptionChainSnapshot] = {}

    @abstractmethod
    def _fetch_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return raw bars with tz-aware known-at ``timestamp`` and OHLC(V)."""

    @abstractmethod
    def _fetch_chain(self, symbol: str, day: date) -> pd.DataFrame:
        """Return the raw chain for ``day`` or raise ``DataUnavailableError``."""

    def get_underlying_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if start > end:
            raise ValueError("start cannot be after end")
        if self.cache is not None:
            cached = self.cache.read_bars(self.name, symbol, start, end)
            if cached is not None:
                return cached
            covered = self.cache.bars_coverage(self.name, symbol)
            fetch_start, fetch_end = start, end
            if covered is not None:
                fetch_start, fetch_end = min(start, covered[0]), max(end, covered[1])
            bars = normalize_bars(self._fetch_bars(symbol, fetch_start, fetch_end), symbol)
            self.cache.write_bars(self.name, symbol, bars, fetch_start, fetch_end)
        else:
            bars = normalize_bars(self._fetch_bars(symbol, start, end), symbol)
        days = bars["timestamp"].dt.tz_convert(MARKET_TZ).dt.date
        return bars[(days >= start) & (days <= end)].reset_index(drop=True)

    def get_option_chain(self, symbol: str, day: date) -> OptionChainSnapshot:
        """Return the chain snapshot for ``day``; raise ``DataUnavailableError`` if none."""
        from options_backtester.data.greeks_engine import fill_missing_greeks

        key = (symbol, day)
        if key in self._chain_memo:
            return self._chain_memo[key]
        frame = self.cache.read_chain(self.name, symbol, day) if self.cache else None
        if frame is None:
            raw = self._fetch_chain(symbol, day)
            if raw is None or raw.empty:
                raise DataUnavailableError(f"{self.name}: no option chain for {symbol} on {day}")
            frame = fill_missing_greeks(normalize_chain(raw), self.greeks)
            if frame.empty:
                raise DataUnavailableError(f"{self.name}: no usable quotes for {symbol} on {day}")
            if self.cache is not None:
                self.cache.write_chain(self.name, symbol, day, frame)
        snapshot = OptionChainSnapshot.from_frame(frame)
        self._chain_memo[key] = snapshot
        return snapshot
