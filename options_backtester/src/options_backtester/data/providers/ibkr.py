"""Interactive Brokers adapter (``ib_insync``): historical bars and the current chain.

IBKR does not serve historical chain snapshots with Greeks, so the chain is a
live snapshot that is cached per day. Historical requests are throttled to stay
inside IBKR's pacing limits (60 requests per 10 minutes) and are retried with
backoff when the gateway reports a pacing violation (error 162) or a
max-rate breach (error 100).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
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
from options_backtester.data.providers._http import ProviderError, RateLimiter

PACING_ERROR_CODES = frozenset({100, 162, 366})


class PacingViolationError(ProviderError):
    """IBKR refused a request for exceeding its pacing limits."""


def _ib_insync() -> Any:
    try:
        import ib_insync
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("install the 'ibkr' extra (ib_insync) to use this provider") from exc
    return ib_insync


class IBKRProvider(DataProvider):
    name = "ibkr"

    def __init__(
        self,
        cache: ParquetCache | None = None,
        greeks: GreeksConfig | None = None,
        *,
        ib: Any | None = None,
        contracts: Any | None = None,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 17,
        exchange: str = "SMART",
        currency: str = "USD",
        index_symbols: frozenset[str] = frozenset({"SPX", "NDX", "RUT", "VIX"}),
        max_dte: int = 120,
        strike_range_pct: float = 0.25,
        batch_size: int = 50,
        limiter: RateLimiter | None = None,
        max_attempts: int = 4,
        backoff_seconds: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        super().__init__(cache=cache, greeks=greeks)
        self._ib = ib
        #: Module providing ``Stock``, ``Index`` and ``Option`` constructors (ib_insync).
        self._contracts = contracts
        self.host, self.port, self.client_id = host, port, client_id
        self.exchange, self.currency = exchange, currency
        self.index_symbols = index_symbols
        self.max_dte, self.strike_range_pct, self.batch_size = max_dte, strike_range_pct, batch_size
        self.limiter = limiter or RateLimiter(max_calls=55, period=600.0)
        self.max_attempts, self.backoff_seconds, self._sleep = max_attempts, backoff_seconds, sleep
        self._now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        self._errors: list[tuple[int, str]] = []
        self._subscribed = False

    # -- connection and pacing -------------------------------------------
    @property
    def ib(self) -> Any:
        if self._ib is None:
            self._ib = _ib_insync().IB()
        if not self._ib.isConnected():
            self._ib.connect(self.host, self.port, clientId=self.client_id)
        if not self._subscribed:
            self._ib.errorEvent += self._on_error
            self._subscribed = True
        return self._ib

    @property
    def api(self) -> Any:
        if self._contracts is None:
            self._contracts = _ib_insync()
        return self._contracts

    def _on_error(self, req_id: int, code: int, message: str, *args: Any) -> None:
        self._errors.append((int(code), str(message)))

    def _paced(self, call: Callable[[], Any]) -> Any:
        """Run one throttled request, retrying with backoff on pacing violations."""
        _ = self.ib  # connect and subscribe to error events
        violation = None
        for attempt in range(self.max_attempts):
            self.limiter.acquire()
            self._errors.clear()
            result = call()
            violation = next(
                (
                    message
                    for code, message in self._errors
                    if code in PACING_ERROR_CODES or "pacing" in message.lower()
                ),
                None,
            )
            if violation is None:
                return result
            if attempt < self.max_attempts - 1:
                self._sleep(self.backoff_seconds * 2**attempt)
        raise PacingViolationError(
            f"IBKR pacing violation persisted after {self.max_attempts} attempts: {violation}"
        )

    def _underlying(self, symbol: str) -> Any:
        if symbol in self.index_symbols:
            contract = self.api.Index(symbol, "CBOE", self.currency)
        else:
            contract = self.api.Stock(symbol, self.exchange, self.currency)
        self.ib.qualifyContracts(contract)
        return contract

    # -- underlying ------------------------------------------------------
    def _fetch_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        contract = self._underlying(symbol)
        span = (end - start).days + 1
        duration = f"{span} D" if span <= 365 else f"{-(-span // 365)} Y"
        bars = self._paced(
            lambda: self.ib.reqHistoricalData(
                contract,
                endDateTime=f"{end:%Y%m%d} 23:59:59 US/Eastern",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        )
        if not bars:
            raise DataUnavailableError(f"ibkr: no bars for {symbol} in {start}..{end}")
        frame = pd.DataFrame(
            {
                "timestamp": [market_close_utc(pd.Timestamp(b.date).date()) for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            }
        )
        return frame[frame["timestamp"] <= self._now()]

    # -- chain -----------------------------------------------------------
    def _fetch_chain(self, symbol: str, day: date) -> pd.DataFrame:
        now = self._now()
        if day != now.tz_convert(MARKET_TZ).date():
            raise DataUnavailableError(
                f"ibkr has no historical option-chain snapshots ({symbol} {day}); only "
                "snapshots taken on the day and cached are available"
            )
        ib = self.ib
        underlying = self._underlying(symbol)
        spot = _first_finite(*(_ticker_prices(ib.reqTickers(underlying)[0])))
        if spot is None:
            raise DataUnavailableError(f"ibkr: no underlying price for {symbol}")
        definitions = ib.reqSecDefOptParams(
            underlying.symbol, "", underlying.secType, underlying.conId
        )
        chain = next((d for d in definitions if d.exchange == self.exchange), None)
        if chain is None:
            raise DataUnavailableError(f"ibkr: no option definitions for {symbol}")
        expirations = [
            e for e in sorted(chain.expirations) if 0 <= (_expiry(e) - day).days <= self.max_dte
        ]
        low, high = spot * (1 - self.strike_range_pct), spot * (1 + self.strike_range_pct)
        strikes = [k for k in sorted(chain.strikes) if low <= k <= high]
        wanted = [
            self.api.Option(symbol, e, k, right, self.exchange, tradingClass=chain.tradingClass)
            for e in expirations
            for k in strikes
            for right in ("C", "P")
        ]
        rows = []
        for offset in range(0, len(wanted), self.batch_size):
            batch = ib.qualifyContracts(*wanted[offset : offset + self.batch_size])
            if not batch:
                continue
            tickers = self._paced(lambda batch=batch: ib.reqTickers(*batch))
            for ticker in tickers:
                contract, model = ticker.contract, getattr(ticker, "modelGreeks", None)
                rows.append(
                    {
                        "expiration": pd.Timestamp(_expiry(contract.lastTradeDateOrContractMonth)),
                        "strike": float(contract.strike),
                        "option_type": "CALL" if contract.right == "C" else "PUT",
                        "bid": _positive(ticker.bid),
                        "ask": _positive(ticker.ask),
                        "last": _positive(ticker.last),
                        "volume": _positive(ticker.volume),
                        "implied_volatility": getattr(model, "impliedVol", np.nan),
                        "delta": getattr(model, "delta", np.nan),
                        "gamma": getattr(model, "gamma", np.nan),
                        # IBKR theta is per day and vega per vol point: the canonical units.
                        "theta": getattr(model, "theta", np.nan),
                        "vega": getattr(model, "vega", np.nan),
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise DataUnavailableError(f"ibkr: empty option chain for {symbol}")
        frame = frame.dropna(subset=["bid", "ask"])
        frame["timestamp"] = now
        frame["underlying_symbol"] = symbol
        frame["underlying_price"] = spot
        return frame


def _expiry(text: str) -> date:
    return pd.to_datetime(str(text)[:8], format="%Y%m%d").date()


def _positive(value: Any) -> float:
    """IBKR reports -1 (or NaN) when a field has no value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number >= 0 else np.nan


def _ticker_prices(ticker: Any) -> tuple[float, ...]:
    market = ticker.marketPrice() if hasattr(ticker, "marketPrice") else np.nan
    return (market, getattr(ticker, "last", np.nan), getattr(ticker, "close", np.nan))


def _first_finite(*values: Any) -> float | None:
    for value in values:
        number = _positive(value)
        if np.isfinite(number) and number > 0:
            return number
    return None
