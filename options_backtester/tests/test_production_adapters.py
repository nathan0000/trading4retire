"""ThetaData, Schwab and IBKR adapters against fake transports (no network)."""

import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from options_backtester.data.base import DataUnavailableError
from options_backtester.data.providers import build_provider
from options_backtester.data.providers._http import (
    HttpClient,
    HttpResponse,
    ProviderError,
    RateLimiter,
)
from options_backtester.data.providers.ibkr import IBKRProvider, PacingViolationError
from options_backtester.data.providers.schwab import SchwabAuth, SchwabProvider
from options_backtester.data.providers.thetadata import ThetaDataProvider

DAY = date(2024, 1, 8)
NOW = pd.Timestamp("2024-01-08 15:30", tz="America/New_York").tz_convert("UTC")


class FakeTransport:
    """Replays queued responses and records every request."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.requests = []

    def request(self, method, url, *, headers=None, data=None, timeout=30.0):
        self.requests.append(
            SimpleNamespace(method=method, url=url, headers=headers or {}, data=data)
        )
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(payload, status=200, headers=None):
    return HttpResponse(status, headers or {}, json.dumps(payload).encode())


def client(transport, **kwargs):
    sleeps = []
    return HttpClient(transport=transport, sleep=sleeps.append, **kwargs), sleeps


# -- shared HTTP plumbing ------------------------------------------------


def test_retries_transient_failures_with_backoff_and_retry_after():
    transport = FakeTransport(
        reply({}, 429, {"Retry-After": "7"}), ConnectionError("reset"), reply({"ok": True})
    )
    http, sleeps = client(transport, backoff_seconds=1.0)
    payload, _ = http.get_json("http://x/y", {"a": 1})
    assert payload == {"ok": True} and sleeps == [7.0, 2.0]
    assert transport.requests[0].url == "http://x/y?a=1"


def test_gives_up_after_max_attempts_and_does_not_retry_client_errors():
    http, sleeps = client(FakeTransport(*[reply({}, 503)] * 3), max_attempts=3)
    with pytest.raises(ProviderError, match="failed after 3 attempts: HTTP 503"):
        http.get_json("http://x/y")
    assert len(sleeps) == 2
    transport = FakeTransport(reply({}, 404))
    http, sleeps = client(transport)
    with pytest.raises(ProviderError, match="HTTP 404"):
        http.get_json("http://x/y")
    assert len(transport.requests) == 1 and sleeps == []


def test_rate_limiter_waits_for_the_window_to_clear():
    clock = SimpleNamespace(now=0.0)
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock.now += seconds

    limiter = RateLimiter(2, 10.0, clock=lambda: clock.now, sleep=sleep)
    limiter.acquire()
    clock.now = 4.0
    limiter.acquire()
    limiter.acquire()  # third call inside 10s must wait until t=10
    assert waits == [6.0] and clock.now == 10.0


# -- ThetaData -----------------------------------------------------------

GREEKS_FORMAT = [
    "ms_of_day",
    "bid",
    "ask",
    "close",
    "volume",
    "delta",
    "gamma",
    "theta",
    "vega",
    "implied_vol",
    "underlying_price",
    "date",
]


def theta_contract(strike, right, bid, ask, delta, iv=0.2):
    return {
        "contract": {
            "root": "SPY",
            "expiration": 20240216,
            "strike": strike * 1000,
            "right": right,
        },
        "ticks": [
            [57600000, bid, ask, (bid + ask) / 2, 50, delta, 0.01, -0.02, 0.1, iv, 470.0, 20240108]
        ],
    }


def test_thetadata_chain_parses_pages_and_falls_back_for_missing_greeks():
    transport = FakeTransport(
        reply({"header": {"format": ["date"]}, "response": [20240105, 20240216, 20260116]}),
        reply(
            {
                "header": {"format": GREEKS_FORMAT},
                "response": [theta_contract(460, "P", 2.0, 2.1, -0.2)],
            },
            headers={"Next-Page": "http://127.0.0.1:25510/v2/page/2"},
        ),
        reply(
            {
                "header": {"format": GREEKS_FORMAT},
                "response": [theta_contract(480, "C", 1.9, 2.0, 0.0, iv=0.0)],
            }
        ),
    )
    http, _ = client(transport)
    provider = ThetaDataProvider(client=http)
    snapshot = provider.get_option_chain("SPY", DAY)
    urls = [r.url for r in transport.requests]
    assert "/list/expirations?root=SPY" in urls[0]
    # Only the expiration inside the DTE window is requested; the second page is followed.
    assert "exp=20240216" in urls[1] and "start_date=20240108" in urls[1]
    assert urls[2] == "http://127.0.0.1:25510/v2/page/2" and len(urls) == 3
    q = snapshot.quotes.set_index("option_type")
    assert snapshot.underlying_price == 470.0 and snapshot.timestamp == pd.Timestamp(
        "2024-01-08 21:00", tz="UTC"
    )
    assert q.loc["PUT", "strike"] == 460.0 and q.loc["PUT", "delta"] == -0.2
    assert q.loc["PUT", "greeks_source"] == "PROVIDER"
    # ThetaData's zero IV means "not computed": the fallback solves it from the mid.
    assert q.loc["CALL", "greeks_source"] == "ESTIMATED_BSM" and 0 < q.loc["CALL", "delta"] < 0.5


def test_thetadata_bars_and_errors():
    fields = ["ms_of_day", "open", "high", "low", "close", "volume", "date"]
    transport = FakeTransport(
        reply(
            {
                "header": {"format": fields},
                "response": [
                    [0, 470, 472, 469, 471, 1000, 20240105],
                    [0, 471, 475, 470, 474, 900, 20240108],
                ],
            }
        ),
    )
    http, _ = client(transport)
    bars = ThetaDataProvider(client=http).get_underlying_bars("SPY", date(2024, 1, 5), DAY)
    assert (
        "/hist/stock/eod?root=SPY&start_date=20240105&end_date=20240108"
        in transport.requests[0].url
    )
    assert list(bars["close"]) == [471.0, 474.0]
    assert bars["timestamp"].iloc[-1] == pd.Timestamp("2024-01-08 21:00", tz="UTC")

    no_data = reply({"header": {"error_type": "NO_DATA", "error_msg": "none"}, "response": []})
    http, _ = client(
        FakeTransport(reply({"header": {"format": ["date"]}, "response": [20240216]}), no_data)
    )
    with pytest.raises(DataUnavailableError):
        ThetaDataProvider(client=http).get_option_chain("SPY", DAY)
    denied = reply({"header": {"error_type": "PERMISSION", "error_msg": "upgrade"}, "response": []})
    http, _ = client(FakeTransport(denied))
    with pytest.raises(ProviderError, match="PERMISSION: upgrade"):
        ThetaDataProvider(client=http).get_underlying_bars("SPY", DAY, DAY)


# -- Schwab --------------------------------------------------------------


def schwab_auth(tmp_path, transport, tokens, clock):
    path = tmp_path / "token.json"
    path.write_text(json.dumps(tokens))
    http, _ = client(transport)
    return SchwabAuth("id", "secret", path, client=http, clock=lambda: clock.now), path


def test_schwab_refreshes_expired_token_and_persists_rotation(tmp_path):
    clock = SimpleNamespace(now=1000.0)
    transport = FakeTransport(
        reply({"access_token": "new", "refresh_token": "r2", "expires_in": 1800})
    )
    auth, path = schwab_auth(
        tmp_path,
        transport,
        {"access_token": "old", "refresh_token": "r1", "expires_at": 1030.0},
        clock,
    )
    assert auth.access_token() == "new"  # within the 60s refresh margin
    sent = transport.requests[0]
    assert sent.method == "POST" and sent.headers["Authorization"].startswith("Basic ")
    assert sent.data == b"grant_type=refresh_token&refresh_token=r1"
    saved = json.loads(path.read_text())
    assert saved["refresh_token"] == "r2" and saved["expires_at"] == 2800.0
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert auth.access_token() == "new" and len(transport.requests) == 1  # still valid: no refresh


def test_schwab_refresh_failure_explains_reauthorization(tmp_path):
    clock = SimpleNamespace(now=1000.0)
    auth, _ = schwab_auth(tmp_path, FakeTransport(reply({}, 400)), {"refresh_token": "r1"}, clock)
    with pytest.raises(ProviderError, match="browser authorization"):
        auth.access_token()


CHAIN = {
    "underlyingPrice": 470.0,
    "callExpDateMap": {
        "2024-02-16:39": {
            "480.0": [
                {
                    "bid": 1.9,
                    "ask": 2.0,
                    "last": 1.95,
                    "totalVolume": 10,
                    "openInterest": 100,
                    "volatility": 15.0,
                    "delta": 0.2,
                    "gamma": 0.01,
                    "theta": -0.05,
                    "vega": 0.3,
                }
            ]
        }
    },
    "putExpDateMap": {
        "2024-02-16:39": {
            "460.0": [
                {
                    "bid": 2.0,
                    "ask": 2.1,
                    "last": 2.05,
                    "totalVolume": 5,
                    "openInterest": 50,
                    "volatility": -999.0,
                    "delta": -999.0,
                    "gamma": -999.0,
                    "theta": -999.0,
                    "vega": -999.0,
                }
            ]
        }
    },
}


def test_schwab_chain_retries_once_on_401_and_normalizes_units(tmp_path):
    clock = SimpleNamespace(now=1000.0)
    token_transport = FakeTransport(reply({"access_token": "fresh", "expires_in": 1800}))
    auth, _ = schwab_auth(
        tmp_path,
        token_transport,
        {"access_token": "stale", "refresh_token": "r1", "expires_at": 9e9},
        clock,
    )
    api = FakeTransport(reply({}, 401), reply(CHAIN))
    http, _ = client(api)
    provider = SchwabProvider(auth, client=http, now=lambda: NOW)
    snapshot = provider.get_option_chain("SPY", DAY)
    assert [r.headers["Authorization"] for r in api.requests] == ["Bearer stale", "Bearer fresh"]
    assert "/marketdata/v1/chains?symbol=SPY" in api.requests[0].url
    q = snapshot.quotes.set_index("option_type")
    assert q.loc["CALL", "implied_volatility"] == pytest.approx(0.15)  # percent -> decimal
    assert q.loc["CALL", "greeks_source"] == "PROVIDER"
    assert (
        q.loc["PUT", "greeks_source"] == "ESTIMATED_BSM" and q.loc["PUT", "delta"] < 0
    )  # -999 sentinel
    with pytest.raises(DataUnavailableError, match="no historical option chains"):
        provider.get_option_chain("SPY", date(2024, 1, 5))


def test_schwab_bars(tmp_path):
    auth, _ = schwab_auth(
        tmp_path,
        FakeTransport(),
        {"access_token": "t", "refresh_token": "r", "expires_at": 9e9},
        SimpleNamespace(now=0.0),
    )
    candles = [
        {
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 2,
            "volume": 9,
            "datetime": int(pd.Timestamp("2024-01-05 06:00", tz="UTC").timestamp() * 1000),
        }
    ]
    http, _ = client(FakeTransport(reply({"candles": candles})))
    bars = SchwabProvider(auth, client=http, now=lambda: NOW).get_underlying_bars(
        "SPY", date(2024, 1, 5), DAY
    )
    assert list(bars["timestamp"]) == [pd.Timestamp("2024-01-05 21:00", tz="UTC")]


# -- IBKR ----------------------------------------------------------------


class Event(list):
    def __iadd__(self, handler):
        self.append(handler)
        return self


class FakeIB:
    def __init__(self, pacing_failures=0):
        self.errorEvent = Event()
        self.pacing_failures = pacing_failures
        self.historical_calls = 0

    def isConnected(self):
        return True

    def qualifyContracts(self, *contracts):
        return list(contracts)

    def reqHistoricalData(self, contract, **kwargs):
        self.historical_calls += 1
        if self.historical_calls <= self.pacing_failures:
            for handler in self.errorEvent:
                handler(1, 162, "Historical Market Data Service error message:Pacing violation")
            return []
        return [
            SimpleNamespace(date=date(2024, 1, 5), open=1.0, high=2.0, low=1.0, close=2.0, volume=9)
        ]

    def reqSecDefOptParams(self, symbol, exchange, sec_type, con_id):
        return [
            SimpleNamespace(
                exchange="SMART",
                tradingClass="SPY",
                expirations=["20240216", "20260116"],
                strikes=[300.0, 460.0, 480.0, 700.0],
            )
        ]

    def reqTickers(self, *contracts):
        tickers = []
        for c in contracts:
            if getattr(c, "secType", "OPT") == "STK":
                tickers.append(
                    SimpleNamespace(contract=c, marketPrice=lambda: 470.0, last=470.0, close=469.0)
                )
                continue
            greeks = SimpleNamespace(
                impliedVol=0.2,
                delta=0.2 if c.right == "C" else -0.2,
                gamma=0.01,
                theta=-0.05,
                vega=0.3,
            )
            tickers.append(
                SimpleNamespace(
                    contract=c, bid=2.0, ask=2.1, last=-1.0, volume=7, modelGreeks=greeks
                )
            )
        return tickers


CONTRACTS = SimpleNamespace(
    Stock=lambda symbol, exchange, currency: SimpleNamespace(symbol=symbol, secType="STK", conId=1),
    Index=lambda symbol, exchange, currency: SimpleNamespace(symbol=symbol, secType="IND", conId=2),
    Option=lambda symbol, expiry, strike, right, exchange, tradingClass=None: SimpleNamespace(
        symbol=symbol, lastTradeDateOrContractMonth=expiry, strike=strike, right=right
    ),
)


def ibkr(fake, **kwargs):
    sleeps = []
    provider = IBKRProvider(
        ib=fake,
        contracts=CONTRACTS,
        sleep=sleeps.append,
        now=lambda: NOW,
        limiter=RateLimiter(1000, 1.0),
        **kwargs,
    )
    return provider, sleeps


def test_ibkr_retries_pacing_violations_with_backoff():
    fake = FakeIB(pacing_failures=2)
    provider, sleeps = ibkr(fake, backoff_seconds=15.0)
    bars = provider.get_underlying_bars("SPY", date(2024, 1, 5), date(2024, 1, 5))
    assert fake.historical_calls == 3 and sleeps == [15.0, 30.0]
    assert list(bars["timestamp"]) == [pd.Timestamp("2024-01-05 21:00", tz="UTC")]
    stuck, _ = ibkr(FakeIB(pacing_failures=99), max_attempts=2)
    with pytest.raises(PacingViolationError, match="persisted after 2 attempts"):
        stuck.get_underlying_bars("SPY", date(2024, 1, 5), date(2024, 1, 5))


def test_ibkr_chain_snapshot_is_current_only_and_filtered():
    provider, _ = ibkr(FakeIB())
    snapshot = provider.get_option_chain("SPY", DAY)
    q = snapshot.quotes
    assert snapshot.underlying_price == 470.0 and snapshot.timestamp == NOW
    # Strikes outside +/-25% and expirations beyond max_dte are never requested.
    assert sorted(q["strike"].unique()) == [460.0, 480.0] and set(q["dte"]) == {39}
    assert len(q) == 4 and q["last"].isna().all()  # IBKR's -1 means "no value"
    assert set(q["greeks_source"]) == {"PROVIDER"}
    with pytest.raises(DataUnavailableError, match="no historical option-chain snapshots"):
        provider.get_option_chain("SPY", date(2024, 1, 5))


def test_provider_factory():
    assert build_provider("synthetic").name == "synthetic"
    assert build_provider("thetadata", options={"security_type": "index"}).security_type == "index"
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("bloomberg")
