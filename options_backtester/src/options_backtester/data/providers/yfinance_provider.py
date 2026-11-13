"""yfinance baseline adapter: historical underlying OHLCV and the *current* option chain.

Yahoo does not serve historical option chains or Greeks. A chain can only be
fetched for today; it is then cached, so running a daily snapshot job builds a
real history over time. Greeks and IV are always computed by the fallback engine.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from options_backtester.data.base import (
    MARKET_TZ,
    DataProvider,
    DataUnavailableError,
    market_close_utc,
)
from options_backtester.data.cache import ParquetCache
from options_backtester.data.greeks_engine import GreeksConfig


def _yf() -> Any:
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("install the 'yfinance' extra to use this provider") from exc
    return yfinance


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(
        self,
        cache: ParquetCache | None = None,
        greeks: GreeksConfig | None = None,
        *,
        max_dte: int = 120,
        download: Callable[..., pd.DataFrame] | None = None,
        ticker_factory: Callable[[str], Any] | None = None,
        now: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        super().__init__(cache=cache, greeks=greeks)
        self.max_dte = max_dte
        self._download = download
        self._ticker_factory = ticker_factory
        self._now = now or (lambda: pd.Timestamp.now(tz="UTC"))

    def _fetch_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        download = self._download or _yf().download
        raw = download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise DataUnavailableError(f"yfinance: no bars for {symbol} in {start}..{end}")
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(c).lower() for c in frame.columns]
        # Yahoo stamps daily bars at midnight; the close is only known at 16:00 New York.
        days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame = frame.reset_index(drop=True)
        frame["timestamp"] = [market_close_utc(d) for d in days]
        today = self._now().tz_convert(MARKET_TZ)
        complete = frame["timestamp"] <= today.tz_convert("UTC")
        return frame.loc[complete, ["timestamp", "open", "high", "low", "close", "volume"]]

    def _fetch_chain(self, symbol: str, day: date) -> pd.DataFrame:
        now = self._now()
        today = now.tz_convert(MARKET_TZ).date()
        if day != today:
            raise DataUnavailableError(
                f"yfinance has no historical option chains ({symbol} {day}); only snapshots "
                "taken on the day and cached are available"
            )
        ticker = (self._ticker_factory or _yf().Ticker)(symbol)
        spot = _spot(ticker)
        frames = []
        for text in ticker.options:
            expiration = date.fromisoformat(text)
            if not 0 <= (expiration - day).days <= self.max_dte:
                continue
            chain = ticker.option_chain(text)
            for option_type, table in (("CALL", chain.calls), ("PUT", chain.puts)):
                if table is None or table.empty:
                    continue
                quoted = (table["bid"] > 0) & (table["ask"] > 0)
                frames.append(
                    pd.DataFrame(
                        {
                            "expiration": pd.Timestamp(expiration),
                            "strike": table["strike"],
                            "option_type": option_type,
                            "bid": table["bid"],
                            "ask": table["ask"],
                            "last": table["lastPrice"],
                            "volume": table.get("volume"),
                            "open_interest": table.get("openInterest"),
                            # Yahoo's IV is unreliable; re-solve from the mid when quoted.
                            "implied_volatility": np.where(
                                quoted, np.nan, table["impliedVolatility"]
                            ),
                        }
                    )
                )
        if not frames:
            raise DataUnavailableError(f"yfinance: empty option chain for {symbol}")
        out = pd.concat(frames, ignore_index=True)
        out["timestamp"] = now
        out["underlying_symbol"] = symbol
        out["underlying_price"] = spot
        return out


def _spot(ticker: Any) -> float:
    info = getattr(ticker, "fast_info", None)
    for key in ("last_price", "lastPrice"):
        try:
            value = info[key] if info is not None else None
        except (KeyError, TypeError):
            value = None
        if value is not None and np.isfinite(value) and value > 0:
            return float(value)
    history = ticker.history(period="1d")
    if history is None or history.empty:
        raise DataUnavailableError("yfinance: cannot determine the underlying price")
    return float(history["Close"].iloc[-1])
