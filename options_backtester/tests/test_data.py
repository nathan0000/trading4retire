from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from options_backtester.data.base import (
    DataUnavailableError,
    market_close_utc,
    normalize_chain,
)
from options_backtester.data.cache import ParquetCache
from options_backtester.data.providers.synthetic import SyntheticProvider, expirations_for
from options_backtester.data.providers.yfinance_provider import YFinanceProvider
from tests.conftest import ScriptedProvider, chain_frame

DAY = date(2024, 1, 8)


def test_normalize_chain_derives_mid_and_dte_and_rejects_naive_timestamps():
    frame = chain_frame(DAY, 100.0, [(95, "p", 1.0, 1.2, -0.2)])
    frame["option_type"] = "P"
    out = normalize_chain(frame)
    assert out.loc[0, "option_type"] == "PUT"
    assert out.loc[0, "mid"] == pytest.approx(1.1)
    assert out.loc[0, "dte"] == 39
    assert str(out["timestamp"].dt.tz) == "UTC"
    frame["timestamp"] = pd.Timestamp("2024-01-08 16:00")
    with pytest.raises(ValueError, match="naive"):
        normalize_chain(frame)


def test_daily_bar_is_stamped_at_the_close_not_midnight():
    assert market_close_utc(date(2024, 1, 8)) == pd.Timestamp("2024-01-08 21:00", tz="UTC")
    assert market_close_utc(date(2024, 7, 8)) == pd.Timestamp("2024-07-08 20:00", tz="UTC")  # DST


def test_provider_is_cache_first(tmp_path):
    closes = {DAY + timedelta(days=i): 100.0 + i for i in range(3)}
    chains = {DAY: chain_frame(DAY, 100.0, [(95, "PUT", 1.0, 1.1, -0.2)])}
    cache = ParquetCache(tmp_path)
    first = ScriptedProvider(closes, chains, cache=cache)
    first.get_underlying_bars("TEST", DAY, DAY + timedelta(days=2))
    snapshot = first.get_option_chain("TEST", DAY)
    assert cache.chain_path("scripted", "TEST", DAY).exists()
    assert cache.chain_path("scripted", "TEST", DAY).name == "2024-01-08.parquet"

    second = ScriptedProvider({}, {}, cache=cache)  # would fail if it had to fetch
    bars = second.get_underlying_bars("TEST", DAY, DAY + timedelta(days=1))
    again = second.get_option_chain("TEST", DAY)
    assert (second.bar_calls, second.chain_calls) == (0, 0)
    assert list(bars["close"]) == [100.0, 101.0]
    pd.testing.assert_frame_equal(again.quotes, snapshot.quotes)
    with pytest.raises(DataUnavailableError):
        second.get_option_chain("TEST", DAY + timedelta(days=1))


def test_snapshot_lookup_finds_contracts():
    provider = ScriptedProvider(
        {DAY: 100.0}, {DAY: chain_frame(DAY, 100.0, [(95, "PUT", 1.0, 1.1, -0.2)])}
    )
    snapshot = provider.get_option_chain("TEST", DAY)
    assert snapshot.lookup(date(2024, 2, 16), 95, "PUT")["bid"] == 1.0
    assert snapshot.lookup(date(2024, 2, 16), 96, "PUT") is None


def test_synthetic_chain_is_consistent_and_labelled():
    provider = SyntheticProvider()
    snapshot = provider.get_option_chain("SPY", date(2023, 6, 1))
    q = snapshot.quotes
    assert provider.synthetic and set(q["greeks_source"]) == {"SYNTHETIC_BSM"}
    assert (q["ask"] > q["bid"]).all() and (q["bid"] >= 0).all()
    assert q["dte"].min() >= 1
    puts = q[(q["option_type"] == "PUT") & (q["expiration"] == q["expiration"].min())]
    assert puts["delta"].between(-1, 0).all()
    assert puts.sort_values("strike")["delta"].is_monotonic_decreasing
    assert all(d.weekday() == 4 for d in expirations_for(date(2023, 6, 1), 190, 60))


def test_synthetic_chain_does_not_depend_on_future_bars():
    days = pd.bdate_range("2023-01-02", "2023-06-30").date
    rng = np.random.default_rng(1)
    closes = dict(zip(days, 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), strict=True))
    target = date(2023, 4, 3)
    full = SyntheticProvider(ScriptedProvider(closes, {}))
    truncated = SyntheticProvider(
        ScriptedProvider({d: c for d, c in closes.items() if d <= target}, {})
    )
    pd.testing.assert_frame_equal(
        full.get_option_chain("X", target).quotes, truncated.get_option_chain("X", target).quotes
    )


class _FakeTicker:
    options = ("2024-02-16", "2030-01-18")
    fast_info = {"last_price": 100.0}

    def option_chain(self, expiration):
        table = pd.DataFrame(
            {
                "strike": [95.0, 100.0, 105.0],
                "bid": [6.0, 2.9, 1.0],
                "ask": [6.2, 3.1, 1.1],
                "lastPrice": [6.1, 3.0, 1.05],
                "volume": [10, 20, 30],
                "openInterest": [100, 200, 300],
                "impliedVolatility": [1e-5, 1e-5, 1e-5],
            }
        )
        return type("Chain", (), {"calls": table, "puts": table})()


def _yf(now):
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Volume": [5, 6],
        },
        index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
    )
    return YFinanceProvider(
        download=lambda *a, **k: raw, ticker_factory=lambda s: _FakeTicker(), now=lambda: now
    )


def test_yfinance_bars_use_close_timestamps_and_drop_incomplete_sessions():
    before_close = pd.Timestamp("2024-01-08 15:00", tz="America/New_York").tz_convert("UTC")
    bars = _yf(before_close).get_underlying_bars("SPY", date(2024, 1, 5), date(2024, 1, 8))
    assert list(bars["timestamp"]) == [pd.Timestamp("2024-01-05 21:00", tz="UTC")]


def test_yfinance_chain_is_current_only_and_uses_the_greeks_fallback():
    now = pd.Timestamp("2024-01-08 15:00", tz="America/New_York").tz_convert("UTC")
    provider = _yf(now)
    with pytest.raises(DataUnavailableError, match="no historical option chains"):
        provider.get_option_chain("SPY", date(2024, 1, 5))
    snapshot = provider.get_option_chain("SPY", date(2024, 1, 8))
    q = snapshot.quotes
    assert snapshot.timestamp == now and set(q["dte"]) == {39}  # 2030 expiry is beyond max_dte
    calls = q[q["option_type"] == "CALL"].set_index("strike")
    assert set(q["greeks_source"]) <= {"ESTIMATED_BSM", "UNAVAILABLE"}
    assert calls.loc[100.0, "greeks_source"] == "ESTIMATED_BSM"
    assert calls.loc[100.0, "implied_volatility"] > 0.05  # Yahoo's 1e-5 was discarded
    assert 0.4 < calls.loc[100.0, "delta"] < 0.6


def test_synthetic_chain_keeps_strikes_listed_after_a_crash():
    days = pd.bdate_range("2023-01-02", "2023-03-31").date
    closes = {d: 400.0 if d < date(2023, 3, 1) else 280.0 for d in days}  # -30% gap
    provider = SyntheticProvider(ScriptedProvider(closes, {}))
    before = provider.get_option_chain("X", date(2023, 2, 28)).quotes
    after = provider.get_option_chain("X", date(2023, 3, 10)).quotes
    held = before[(before["dte"] > 30) & (before["strike"].between(360, 440))]
    keys = ["expiration", "strike", "option_type"]
    merged = held[keys].merge(after[keys], on=keys, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
