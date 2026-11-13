"""Model-priced option chains for offline development and engine validation.

THIS IS NOT HISTORICAL OPTION DATA. Chains are priced with Black-Scholes from a
trailing realized-volatility estimate plus a fixed skew. Results built on this
provider validate engine mechanics and relative parameter behaviour only; they
are never a historical performance claim. Reports label them as SYNTHETIC.

Underlying bars come from another provider (e.g. real ``yfinance`` closes) when
``bars_source`` is given, otherwise from a seeded stochastic-volatility path.
"""

from __future__ import annotations

import zlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from options_backtester.data.base import (
    MARKET_TZ,
    DataProvider,
    DataUnavailableError,
    market_close_utc,
    normalize_bars,
)
from options_backtester.data.cache import ParquetCache
from options_backtester.data.greeks_engine import (
    GreeksConfig,
    bsm_greeks,
    bsm_price,
    years_to_expiry,
)


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 7
    history_start: date = date(2015, 1, 2)
    history_end: date = date(2026, 12, 31)
    start_price: float = Field(default=400.0, gt=0)
    annual_drift: float = 0.07
    base_volatility: float = Field(default=0.16, gt=0)
    #: Implied vol = trailing realized vol x premium (the volatility risk premium).
    iv_premium: float = Field(default=1.15, gt=0)
    realized_window: int = Field(default=20, ge=5)
    min_iv: float = Field(default=0.08, gt=0)
    #: IV change per standard deviation of moneyness (negative = put skew).
    skew: float = -0.10
    smile: float = 0.03
    strike_range_pct: float = Field(default=0.25, gt=0, lt=1)
    max_dte: int = Field(default=190, ge=7)
    weekly_dte: int = Field(default=60, ge=0)
    spread_pct: float = Field(default=0.03, ge=0)
    min_half_spread: float = Field(default=0.01, gt=0)


def _strike_step(price: float) -> float:
    # Few, widely spaced thresholds: a contract must stay listed as the underlying drifts.
    if price < 50:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


def expirations_for(day: date, max_dte: int, weekly_dte: int) -> list[date]:
    """Fridays out to ``weekly_dte`` and third Fridays (monthlies) out to ``max_dte``."""
    result = []
    current = day + timedelta(days=(4 - day.weekday()) % 7 or 7)
    while (current - day).days <= max_dte:
        monthly = 15 <= current.day <= 21
        if monthly or (current - day).days <= weekly_dte:
            result.append(current)
        current += timedelta(days=7)
    return result


class SyntheticProvider(DataProvider):
    name = "synthetic"
    synthetic = True

    def __init__(
        self,
        bars_source: DataProvider | None = None,
        config: SyntheticConfig | None = None,
        cache: ParquetCache | None = None,
        greeks: GreeksConfig | None = None,
    ) -> None:
        super().__init__(cache=cache, greeks=greeks)
        self.config = config or SyntheticConfig()
        self.bars_source = bars_source
        if bars_source is not None:
            self.name = f"synthetic-{bars_source.name}"
        self._history: dict[str, pd.DataFrame] = {}
        self._window: dict[str, tuple[date, date, pd.Series]] = {}

    # -- underlying -----------------------------------------------------
    def _generate_path(self, symbol: str) -> pd.DataFrame:
        cfg = self.config
        days = pd.bdate_range(cfg.history_start, cfg.history_end)
        rng = np.random.default_rng([cfg.seed, zlib.crc32(symbol.encode())])
        n = len(days)
        z_ret, z_vol = rng.standard_normal(n), rng.standard_normal(n)
        z_vol = -0.6 * z_ret + 0.8 * z_vol  # leverage effect
        log_vol = np.empty(n)
        level = 0.0
        for i in range(n):  # OU process on log-volatility (scalar recursion)
            level = 0.97 * level + 0.07 * z_vol[i]
            log_vol[i] = level
        sigma = cfg.base_volatility * np.exp(log_vol)
        sigma_prev = np.concatenate([[cfg.base_volatility], sigma[:-1]])
        dt = 1.0 / 252.0
        returns = (cfg.annual_drift - 0.5 * sigma_prev**2) * dt + sigma_prev * np.sqrt(dt) * z_ret
        close = cfg.start_price * np.exp(np.cumsum(returns))
        open_ = np.concatenate([[cfg.start_price], close[:-1]])
        wiggle = np.abs(rng.standard_normal(n)) * sigma_prev * np.sqrt(dt) * 0.5
        return pd.DataFrame(
            {
                "timestamp": [market_close_utc(d) for d in days],
                "open": open_,
                "high": np.maximum(open_, close) * (1 + wiggle),
                "low": np.minimum(open_, close) * (1 - wiggle),
                "close": close,
                "volume": np.nan,
            }
        )

    def _fetch_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if self.bars_source is not None:
            return self.bars_source.get_underlying_bars(symbol, start, end)
        if symbol not in self._history:
            self._history[symbol] = self._generate_path(symbol)
        bars = self._history[symbol]
        days = bars["timestamp"].dt.tz_convert(MARKET_TZ).dt.date
        return bars[(days >= start) & (days <= end)]

    def _closes_through(self, symbol: str, day: date) -> pd.Series:
        """Closes with known-at time <= ``day``'s close, indexed by market date."""
        cfg = self.config
        # Long enough for realized vol and for every strike listed while an expiration lived.
        lookback = day - timedelta(days=max(cfg.realized_window * 3, cfg.max_dte + 10))
        memo = self._window.get(symbol)
        if memo is None or lookback < memo[0] or day > memo[1]:
            # Fetch a generous window once so a backtest does not hit the source daily.
            yesterday = pd.Timestamp.now(tz=MARKET_TZ).date() - timedelta(days=1)
            start = lookback - timedelta(days=30)
            end = max(day, min(day + timedelta(days=400), yesterday))
            bars = normalize_bars(self._fetch_bars(symbol, start, end), symbol)
            days = bars["timestamp"].dt.tz_convert(MARKET_TZ).dt.date
            memo = (start, end, pd.Series(bars["close"].to_numpy(), index=pd.Index(days)))
            self._window[symbol] = memo
        closes = memo[2]
        return closes[(closes.index >= lookback) & (closes.index <= day)]

    # -- chain ----------------------------------------------------------
    def _fetch_chain(self, symbol: str, day: date) -> pd.DataFrame:
        cfg = self.config
        closes = self._closes_through(symbol, day)
        if closes.empty or closes.index[-1] != day:
            raise DataUnavailableError(f"synthetic: no underlying bar for {symbol} on {day}")
        spot = float(closes.iloc[-1])
        returns = np.log(closes).diff().dropna().tail(cfg.realized_window)
        realized = (
            float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) >= 5 else cfg.base_volatility
        )
        atm_iv = max(cfg.min_iv, realized * cfg.iv_premium)

        step = _strike_step(spot)
        # Strikes stay listed once opened: span the trailing price range, not just today's
        # spot, so a held contract is still quoted after a large move.
        low = np.floor(float(closes.min()) * (1 - cfg.strike_range_pct) / step) * step
        high = np.ceil(float(closes.max()) * (1 + cfg.strike_range_pct) / step) * step
        strikes = np.arange(max(low, step), high + step / 2, step)
        expirations = expirations_for(day, cfg.max_dte, cfg.weekly_dte)
        grid = pd.MultiIndex.from_product(
            [pd.to_datetime(expirations), strikes, ["CALL", "PUT"]],
            names=["expiration", "strike", "option_type"],
        ).to_frame(index=False)
        grid["timestamp"] = market_close_utc(day)
        t = years_to_expiry(grid["timestamp"], grid["expiration"])
        k = grid["strike"].to_numpy()
        is_call = (grid["option_type"] == "CALL").to_numpy()
        x = np.clip(np.log(k / spot) / (atm_iv * np.sqrt(t)), -4.0, 4.0)
        iv = np.maximum(cfg.min_iv, atm_iv * (1.0 + cfg.skew * x + cfg.smile * x * x))
        r, q = self.greeks.risk_free_rate, self.greeks.dividend_yield
        theo = bsm_price(spot, k, t, r, q, iv, is_call)
        half = np.maximum(cfg.min_half_spread, theo * cfg.spread_pct / 2.0)
        bid = np.maximum(0.0, np.round(theo - half, 2))
        ask = np.maximum(bid + 0.01, np.round(theo + half, 2))
        interest = np.round(5000.0 * np.exp(-0.5 * x * x))
        grid = grid.assign(
            underlying_symbol=symbol,
            underlying_price=spot,
            bid=bid,
            ask=ask,
            last=np.round(theo, 2),
            volume=np.round(interest * 0.2),
            open_interest=interest,
            implied_volatility=iv,
            greeks_source="SYNTHETIC_BSM",
            **bsm_greeks(spot, k, t, r, q, iv, is_call),
        )
        return grid
