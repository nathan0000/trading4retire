"""ThetaData v2 local-terminal historical SPX adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from typing import Any

import pandas as pd

from spx_0dte_backtest.providers.base import HistoricalDataRequest, HistoricalDataset
from spx_0dte_backtest.providers.http import (
    JsonTransport,
    ProviderConnectionError,
    ProviderResponse,
    UrllibJsonTransport,
)


@dataclass(frozen=True, slots=True)
class ThetaDataSettings:
    base_url: str = "http://127.0.0.1:25510/v2"
    timeout_seconds: float = 60.0
    timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("ThetaData base URL must use HTTP or HTTPS")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("ThetaData timeout must be finite and positive")


class ThetaDataAdapter:
    """Download canonical SPX bars and same-day chains from Theta Terminal."""

    name = "thetadata"

    def __init__(
        self,
        settings: ThetaDataSettings | None = None,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.settings = settings or ThetaDataSettings()
        self.transport = transport or UrllibJsonTransport()

    def download(self, request: HistoricalDataRequest) -> HistoricalDataset:
        spx = self._index_bars(request.spx_root, request)
        vix = self._index_bars(request.vix_root, request) if request.include_vix else None
        if vix is not None:
            vix = vix[["timestamp", "close"]].rename(columns={"close": "vix"})
        option_frames = [self._zero_dte_chain(day, request) for day in self._weekdays(request)]
        options = (
            pd.concat(option_frames, ignore_index=True)
            if option_frames
            else _empty_options(self.settings.timezone)
        )
        if not options.empty:
            options = options.sort_values(
                ["timestamp", "expiration", "option_type", "strike"], kind="stable"
            ).reset_index(drop=True)
        return HistoricalDataset(
            provider=self.name,
            spx_bars=spx,
            option_quotes=options,
            vix=vix,
            metadata={
                "api_version": "v2",
                "base_url": self.settings.base_url,
                "option_endpoint": "bulk_hist/option/all_greeks",
                "interval_milliseconds": request.interval_minutes * 60_000,
                "greeks_source": "THETADATA_HISTORICAL_CALCULATION",
            },
        )

    def _index_bars(self, root: str, request: HistoricalDataRequest) -> pd.DataFrame:
        payloads = self._pages(
            "/hist/index/ohlc",
            {
                "root": root,
                "start_date": _compact(request.start),
                "end_date": _compact(request.end),
                "ivl": request.interval_minutes * 60_000,
                "rth": "true",
                "use_csv": "false",
            },
        )
        rows: list[dict[str, Any]] = []
        for response in payloads:
            fields = self._format(response)
            for tick in response.payload.get("response", []):
                values = dict(zip(fields, tick, strict=True))
                rows.append(
                    {
                        "timestamp": self._timestamp(values["date"], values["ms_of_day"]),
                        "open": values["open"],
                        "high": values["high"],
                        "low": values["low"],
                        "close": values["close"],
                        "volume": values.get("volume"),
                        "symbol": root,
                        "source": "THETADATA_INDEX_OHLC_V2",
                    }
                )
        columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "source"]
        if not rows:
            return pd.DataFrame(
                {
                    "timestamp": pd.DatetimeIndex([], tz=self.settings.timezone),
                    **{column: [] for column in columns[1:]},
                }
            )
        return pd.DataFrame(rows, columns=columns).sort_values("timestamp").reset_index(drop=True)

    def _zero_dte_chain(self, day: date, request: HistoricalDataRequest) -> pd.DataFrame:
        responses = self._pages(
            "/bulk_hist/option/all_greeks",
            {
                "root": request.spx_root,
                "exp": _compact(day),
                "start_date": _compact(day),
                "end_date": _compact(day),
                "ivl": request.interval_minutes * 60_000,
                "rth": "true",
                "use_csv": "false",
            },
        )
        rows: list[dict[str, Any]] = []
        for response in responses:
            fields = self._format(response)
            contracts = response.payload.get("response", [])
            for item in contracts:
                contract = item["contract"]
                expiration = pd.to_datetime(str(contract["expiration"]), format="%Y%m%d").date()
                right = str(contract["right"]).upper()
                for tick in item.get("ticks", []):
                    values = dict(zip(fields, tick, strict=True))
                    rows.append(
                        {
                            "timestamp": self._timestamp(values["date"], values["ms_of_day"]),
                            "expiration": expiration,
                            "strike": float(contract["strike"]) / 1000.0,
                            "option_type": {"C": "CALL", "P": "PUT"}[right],
                            "bid": values["bid"],
                            "ask": values["ask"],
                            "last": pd.NA,
                            "volume": pd.NA,
                            "open_interest": pd.NA,
                            "delta": values["delta"],
                            "gamma": values["gamma"],
                            "theta": values["theta"],
                            "vega": values["vega"],
                            "implied_volatility": values["implied_vol"],
                            "contract_symbol": (
                                f"THETADATA:{contract['root']}:{contract['expiration']}:"
                                f"{right}:{contract['strike']}"
                            ),
                            "source": "THETADATA_BULK_ALL_GREEKS_V2",
                        }
                    )
        return pd.DataFrame(rows) if rows else _empty_options(self.settings.timezone)

    def _pages(self, path: str, params: dict[str, object]) -> tuple[ProviderResponse, ...]:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        current_params: dict[str, object] | None = params
        pages: list[ProviderResponse] = []
        while url:
            response = self.transport.get_json(
                url, current_params, timeout=self.settings.timeout_seconds
            )
            self._raise_provider_error(response)
            pages.append(response)
            next_page = _header(response.headers, "Next-Page")
            url = "" if not next_page or next_page == "null" else next_page
            current_params = None
        return tuple(pages)

    @staticmethod
    def _format(response: ProviderResponse) -> tuple[str, ...]:
        header = response.payload.get("header", {})
        fields = header.get("format") if isinstance(header, dict) else None
        if not isinstance(fields, list) or not all(isinstance(value, str) for value in fields):
            raise ProviderConnectionError("ThetaData response is missing header.format")
        return tuple(fields)

    @staticmethod
    def _raise_provider_error(response: ProviderResponse) -> None:
        header = response.payload.get("header", {})
        if not isinstance(header, dict):
            raise ProviderConnectionError("ThetaData response header is invalid")
        error = header.get("error_type")
        if error not in {None, "null", ""}:
            message = header.get("error_msg") or error
            raise ProviderConnectionError(f"ThetaData error {error}: {message}")

    def _timestamp(self, day: object, milliseconds: object) -> pd.Timestamp:
        midnight = pd.to_datetime(str(int(day)), format="%Y%m%d")
        return (midnight + pd.Timedelta(milliseconds=int(milliseconds))).tz_localize(
            self.settings.timezone, ambiguous="raise", nonexistent="raise"
        )

    @staticmethod
    def _weekdays(request: HistoricalDataRequest) -> tuple[date, ...]:
        days: list[date] = []
        current = request.start
        while current <= request.end:
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)
        return tuple(days)


def _compact(value: date) -> str:
    return value.strftime("%Y%m%d")


def _header(headers: Any, name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return None


def _empty_options(timezone: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex([], tz=timezone),
            "expiration": [],
            "strike": [],
            "option_type": [],
            "bid": [],
            "ask": [],
            "last": [],
            "volume": [],
            "open_interest": [],
            "delta": [],
            "gamma": [],
            "theta": [],
            "vega": [],
            "implied_volatility": [],
            "contract_symbol": [],
            "source": [],
        }
    )
