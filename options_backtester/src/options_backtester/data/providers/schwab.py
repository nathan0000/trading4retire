"""Charles Schwab Trader API adapter: OAuth2 refresh, price history and the current chain.

Schwab serves no historical option chains. As with ``yfinance``, the chain can
only be fetched for today and is then cached, so a daily job builds real history.

Credentials are read from the environment (``SCHWAB_CLIENT_ID`` /
``SCHWAB_CLIENT_SECRET``); tokens live in a JSON file that already contains a
``refresh_token`` obtained through Schwab's one-time browser authorization.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

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
from options_backtester.data.providers._http import HttpClient, ProviderError, RateLimiter

TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
MARKET_DATA_URL = "https://api.schwabapi.com/marketdata/v1"


class SchwabAuth:
    """Holds the OAuth2 token pair and refreshes the access token before it expires."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_path: str | Path,
        *,
        client: HttpClient | None = None,
        clock: Callable[[], float] = time.time,
        refresh_margin_seconds: float = 60.0,
    ) -> None:
        if not client_id or not client_secret:
            raise ValueError("Schwab client id and secret are required")
        self._basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        self.token_path = Path(token_path)
        self.client = client or HttpClient()
        self._clock = clock
        self._margin = refresh_margin_seconds
        self._tokens: dict[str, Any] | None = None

    @classmethod
    def from_env(cls, token_path: str | Path, **kwargs: Any) -> SchwabAuth:
        return cls(
            os.environ.get("SCHWAB_CLIENT_ID", ""),
            os.environ.get("SCHWAB_CLIENT_SECRET", ""),
            token_path,
            **kwargs,
        )

    def _load(self) -> dict[str, Any]:
        if self._tokens is None:
            if not self.token_path.exists():
                raise ProviderError(
                    f"Schwab token file {self.token_path} not found; complete the one-time "
                    "browser authorization and save its JSON response there"
                )
            self._tokens = json.loads(self.token_path.read_text())
        return self._tokens

    def access_token(self, force_refresh: bool = False) -> str:
        tokens = self._load()
        expires_at = float(tokens.get("expires_at", 0.0))
        if (
            force_refresh
            or not tokens.get("access_token")
            or self._clock() >= expires_at - self._margin
        ):
            tokens = self._refresh(tokens)
        return str(tokens["access_token"])

    def _refresh(self, tokens: dict[str, Any]) -> dict[str, Any]:
        if not tokens.get("refresh_token"):
            raise ProviderError("Schwab token file has no refresh_token; re-authorize")
        body = urlencode({"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]})
        response = self.client.request(
            "POST",
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {self._basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=body.encode(),
        )
        if response.status >= 400:
            raise ProviderError(
                f"Schwab token refresh failed with HTTP {response.status}; the refresh token "
                "expires after 7 days and then needs a new browser authorization",
                response.status,
            )
        fresh = response.json()
        updated = {
            **tokens,
            **fresh,
            "expires_at": self._clock() + float(fresh.get("expires_in", 1800)),
        }
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(updated, indent=2))
        os.chmod(self.token_path, 0o600)
        self._tokens = updated
        return updated


class SchwabProvider(DataProvider):
    name = "schwab"

    def __init__(
        self,
        auth: SchwabAuth,
        cache: ParquetCache | None = None,
        greeks: GreeksConfig | None = None,
        *,
        max_dte: int = 120,
        client: HttpClient | None = None,
        now: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        super().__init__(cache=cache, greeks=greeks)
        self.auth = auth
        self.max_dte = max_dte
        # Schwab allows 120 market-data requests per minute.
        self.client = client or HttpClient(limiter=RateLimiter(max_calls=110, period=60.0))
        self._now = now or (lambda: pd.Timestamp.now(tz="UTC"))

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        for attempt in range(2):
            headers = {
                "Authorization": f"Bearer {self.auth.access_token(force_refresh=attempt == 1)}",
                "Accept": "application/json",
            }
            response = self.client.request(
                "GET", f"{MARKET_DATA_URL}{path}", params=params, headers=headers
            )
            if response.status != 401:
                break
        if response.status >= 400:
            raise ProviderError(
                f"Schwab GET {path} returned HTTP {response.status}", response.status
            )
        return response.json()

    def _fetch_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        zone = ZoneInfo(MARKET_TZ)

        def millis(day: date, at: clock_time) -> int:
            return int(datetime.combine(day, at, tzinfo=zone).timestamp() * 1000)

        payload = self._get(
            "/pricehistory",
            {
                "symbol": symbol,
                "periodType": "year",
                "frequencyType": "daily",
                "frequency": 1,
                "startDate": millis(start, clock_time(0, 0)),
                "endDate": millis(end, clock_time(23, 59)),
                "needExtendedHoursData": "false",
            },
        )
        candles = payload.get("candles") or []
        if not candles:
            raise DataUnavailableError(f"schwab: no bars for {symbol} in {start}..{end}")
        frame = pd.DataFrame(candles)
        days = pd.to_datetime(frame["datetime"], unit="ms", utc=True).dt.tz_convert(MARKET_TZ)
        frame["timestamp"] = [market_close_utc(d.date()) for d in days]
        complete = frame["timestamp"] <= self._now()
        return frame.loc[complete, ["timestamp", "open", "high", "low", "close", "volume"]]

    def _fetch_chain(self, symbol: str, day: date) -> pd.DataFrame:
        now = self._now()
        today = now.tz_convert(MARKET_TZ).date()
        if day != today:
            raise DataUnavailableError(
                f"schwab has no historical option chains ({symbol} {day}); only snapshots "
                "taken on the day and cached are available"
            )
        payload = self._get(
            "/chains",
            {
                "symbol": symbol,
                "contractType": "ALL",
                "includeUnderlyingQuote": "true",
                "fromDate": day.isoformat(),
                "toDate": (day + timedelta(days=self.max_dte)).isoformat(),
            },
        )
        spot = payload.get("underlyingPrice") or (payload.get("underlying") or {}).get("last")
        rows = []
        for option_type, key in (("CALL", "callExpDateMap"), ("PUT", "putExpDateMap")):
            for expiry_key, strikes in (payload.get(key) or {}).items():
                expiration = date.fromisoformat(expiry_key.split(":")[0])
                for strike, contracts in strikes.items():
                    for c in contracts:
                        rows.append(
                            {
                                "expiration": pd.Timestamp(expiration),
                                "strike": float(strike),
                                "option_type": option_type,
                                "bid": c.get("bid"),
                                "ask": c.get("ask"),
                                "last": c.get("last"),
                                "volume": c.get("totalVolume"),
                                "open_interest": c.get("openInterest"),
                                # Schwab quotes IV in percent and uses -999 for "not available".
                                "implied_volatility": _valid(c.get("volatility"), scale=0.01),
                                "delta": _valid(c.get("delta")),
                                "gamma": _valid(c.get("gamma")),
                                "theta": _valid(c.get("theta")),
                                "vega": _valid(c.get("vega")),
                            }
                        )
        if not rows or not spot:
            raise DataUnavailableError(f"schwab: empty option chain for {symbol}")
        frame = pd.DataFrame(rows)
        frame["timestamp"] = now
        frame["underlying_symbol"] = symbol
        frame["underlying_price"] = float(spot)
        return frame


def _valid(value: Any, scale: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if not np.isfinite(number) or number <= -999 else number * scale
