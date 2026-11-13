"""Shared fixtures: a scripted provider serving hand-built bars and chains."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_backtester.data.base import DataProvider, DataUnavailableError, market_close_utc

EXPIRY = date(2024, 2, 16)


def chain_frame(day, spot, rows, expiration=EXPIRY, symbol="TEST"):
    """rows: (strike, option_type, bid, ask, delta[, expiration])."""
    records = []
    for row in rows:
        strike, option_type, bid, ask, delta = row[:5]
        records.append(
            {
                "timestamp": market_close_utc(day),
                "underlying_symbol": symbol,
                "underlying_price": spot,
                "expiration": pd.Timestamp(row[5] if len(row) > 5 else expiration),
                "strike": strike,
                "option_type": option_type,
                "bid": bid,
                "ask": ask,
                "volume": 1000,
                "open_interest": 5000,
                "implied_volatility": 0.2,
                "delta": delta,
                "gamma": 0.01,
                "theta": -0.02,
                "vega": 0.1,
            }
        )
    return pd.DataFrame(records)


class ScriptedProvider(DataProvider):
    name = "scripted"

    def __init__(self, closes: dict[date, float], chains: dict[date, pd.DataFrame], **kwargs):
        super().__init__(**kwargs)
        self.closes, self.chains = closes, chains
        self.bar_calls = self.chain_calls = 0

    def _fetch_bars(self, symbol, start, end):
        self.bar_calls += 1
        days = [d for d in sorted(self.closes) if start <= d <= end]
        closes = np.array([self.closes[d] for d in days], dtype=float)
        return pd.DataFrame(
            {
                "timestamp": [market_close_utc(d) for d in days],
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": 1e6,
            }
        )

    def _fetch_chain(self, symbol, day):
        self.chain_calls += 1
        if day not in self.chains:
            raise DataUnavailableError(f"no chain on {day}")
        return self.chains[day]


@pytest.fixture
def put_spread_days():
    """Three sessions of a 95/90 put spread on a 100 underlying, 39 DTE at entry."""
    d1, d2, d3 = date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)
    return d1, d2, d3
