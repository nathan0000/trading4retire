"""ThetaData adapter (Theta Terminal v2 REST): EOD underlying bars and EOD chains with Greeks.

Endpoints used (all served by the local terminal, default ``http://127.0.0.1:25510/v2``):

* ``/hist/stock/eod`` or ``/hist/index/eod``     underlying daily bars
* ``/list/expirations``                          expirations listed for a root
* ``/bulk_hist/option/eod_greeks``               one expiration's EOD quotes + Greeks
* ``/bulk_hist/option/open_interest``            optional open interest

Responses carry ``header.format`` (column names) and page through the ``Next-Page``
header. Field names are read from ``header.format`` rather than assumed by position.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from options_backtester.data.base import DataProvider, DataUnavailableError, market_close_utc
from options_backtester.data.cache import ParquetCache
from options_backtester.data.greeks_engine import GreeksConfig
from options_backtester.data.providers._http import HttpClient, ProviderError, RateLimiter


def _compact(day: date) -> str:
    return day.strftime("%Y%m%d")


def _day(value: Any) -> date:
    return pd.to_datetime(str(int(value)), format="%Y%m%d").date()


class ThetaDataProvider(DataProvider):
    name = "thetadata"

    def __init__(
        self,
        cache: ParquetCache | None = None,
        greeks: GreeksConfig | None = None,
        *,
        base_url: str = "http://127.0.0.1:25510/v2",
        security_type: Literal["stock", "index"] = "stock",
        max_dte: int = 120,
        include_open_interest: bool = False,
        client: HttpClient | None = None,
    ) -> None:
        super().__init__(cache=cache, greeks=greeks)
        self.base_url = base_url.rstrip("/")
        self.security_type = security_type
        self.max_dte = max_dte
        self.include_open_interest = include_open_interest
        self.client = client or HttpClient(
            limiter=RateLimiter(max_calls=20, period=1.0), timeout=120
        )
        self._expirations: dict[str, list[date]] = {}

    # -- transport ------------------------------------------------------
    def _pages(self, path: str, params: dict[str, Any]) -> list[tuple[list[str], list[Any]]]:
        """Follow ``Next-Page`` and return ``(format, rows)`` for every page."""
        url: str | None = f"{self.base_url}{path}"
        current: dict[str, Any] | None = {**params, "use_csv": "false"}
        pages = []
        while url:
            payload, response = self.client.get_json(url, current)
            header = payload.get("header", {}) if isinstance(payload, dict) else {}
            error = header.get("error_type")
            if error not in (None, "null", ""):
                if str(error).upper() == "NO_DATA":
                    return pages
                raise ProviderError(f"ThetaData {error}: {header.get('error_msg') or ''}".strip())
            fields = header.get("format")
            pages.append((list(fields) if fields else [], payload.get("response", [])))
            following = response.header("Next-Page")
            url = following if following and following != "null" else None
            current = None
        return pages

    # -- underlying -----------------------------------------------------
    def _fetch_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        pages = self._pages(
            f"/hist/{self.security_type}/eod",
            {"root": symbol, "start_date": _compact(start), "end_date": _compact(end)},
        )
        rows = [dict(zip(fields, tick, strict=False)) for fields, ticks in pages for tick in ticks]
        if not rows:
            raise DataUnavailableError(f"thetadata: no bars for {symbol} in {start}..{end}")
        frame = pd.DataFrame(rows)
        frame["timestamp"] = [market_close_utc(_day(d)) for d in frame["date"]]
        if "volume" not in frame:
            frame["volume"] = np.nan
        return frame[["timestamp", "open", "high", "low", "close", "volume"]]

    # -- chain ----------------------------------------------------------
    def expirations(self, symbol: str) -> list[date]:
        if symbol not in self._expirations:
            pages = self._pages("/list/expirations", {"root": symbol})
            self._expirations[symbol] = sorted(_day(v) for _, values in pages for v in values)
        return self._expirations[symbol]

    def _bulk(self, endpoint: str, symbol: str, expiration: date, day: date) -> list[dict]:
        pages = self._pages(
            f"/bulk_hist/option/{endpoint}",
            {
                "root": symbol,
                "exp": _compact(expiration),
                "start_date": _compact(day),
                "end_date": _compact(day),
            },
        )
        rows = []
        for fields, contracts in pages:
            for item in contracts:
                contract = item["contract"]
                for tick in item.get("ticks", []):
                    rows.append(
                        {
                            "expiration": _day(contract["expiration"]),
                            "strike": float(contract["strike"]) / 1000.0,
                            "option_type": {"C": "CALL", "P": "PUT"}[
                                str(contract["right"]).upper()
                            ],
                            **dict(zip(fields, tick, strict=False)),
                        }
                    )
        return rows

    def _fetch_chain(self, symbol: str, day: date) -> pd.DataFrame:
        wanted = [e for e in self.expirations(symbol) if 0 <= (e - day).days <= self.max_dte]
        rows = [row for e in wanted for row in self._bulk("eod_greeks", symbol, e, day)]
        if not rows:
            raise DataUnavailableError(f"thetadata: no option chain for {symbol} on {day}")
        raw = pd.DataFrame(rows)

        def column(*names: str) -> pd.Series:
            for name in names:
                if name in raw:
                    return pd.to_numeric(raw[name], errors="coerce")
            return pd.Series(np.nan, index=raw.index)

        frame = pd.DataFrame(
            {
                "timestamp": market_close_utc(day),
                "underlying_symbol": symbol,
                "underlying_price": column("underlying_price"),
                "expiration": pd.to_datetime(raw["expiration"]),
                "strike": raw["strike"],
                "option_type": raw["option_type"],
                "bid": column("bid"),
                "ask": column("ask"),
                "last": column("close"),
                "volume": column("volume"),
                "implied_volatility": column("implied_vol", "iv"),
                "delta": column("delta"),
                "gamma": column("gamma"),
                "theta": column("theta"),
                "vega": column("vega"),
            }
        )
        # ThetaData reports 0 for IV/Greeks it could not compute; let the fallback fill them.
        unsolved = ~(frame["implied_volatility"] > 0)
        frame.loc[unsolved, ["implied_volatility", "delta", "gamma", "theta", "vega"]] = np.nan
        if frame["underlying_price"].isna().any():
            spot = float(self.get_underlying_bars(symbol, day, day)["close"].iloc[-1])
            frame["underlying_price"] = frame["underlying_price"].fillna(spot)
        if self.include_open_interest:
            interest = pd.DataFrame(
                [r for e in wanted for r in self._bulk("open_interest", symbol, e, day)]
            )
            if not interest.empty and "open_interest" in interest:
                keys = ["expiration", "strike", "option_type"]
                interest["expiration"] = pd.to_datetime(interest["expiration"])
                interest = interest.drop_duplicates(keys, keep="last")[[*keys, "open_interest"]]
                frame = frame.merge(interest, on=keys, how="left")
        return frame
