from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from spx_0dte_backtest.providers import (
    HistoricalDataRequest,
    ProviderConnectionError,
    ProviderResponse,
    ThetaDataAdapter,
    ThetaDataSettings,
)


class FakeTransport:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def get_json(
        self, url: str, params: dict[str, object] | None, *, timeout: float
    ) -> ProviderResponse:
        self.calls.append((url, params))
        return self.responses.pop(0)


def index_response(root: str = "SPX") -> ProviderResponse:
    return ProviderResponse(
        payload={
            "header": {
                "format": [
                    "ms_of_day",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "count",
                    "date",
                ],
                "error_type": None,
            },
            "response": [[34200000, 6000, 6002, 5999, 6001, 0, 0, 20260812]],
            "root": root,
        },
        headers={},
    )


def option_response(*, next_page: str | None = None) -> ProviderResponse:
    fields = [
        "ms_of_day",
        "bid",
        "ask",
        "delta",
        "theta",
        "vega",
        "rho",
        "epsilon",
        "lambda",
        "gamma",
        "vanna",
        "charm",
        "vomma",
        "veta",
        "vera",
        "speed",
        "zomma",
        "color",
        "ultima",
        "d1",
        "d2",
        "dual_delta",
        "dual_gamma",
        "implied_vol",
        "iv_error",
        "ms_of_day2",
        "underlying_price",
        "date",
    ]
    tick: list[Any] = [
        38700000,
        1.20,
        1.40,
        -0.10,
        -0.50,
        0.10,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.22,
        0,
        38700000,
        6000,
        20260812,
    ]
    headers = {} if next_page is None else {"Next-Page": next_page}
    return ProviderResponse(
        payload={
            "header": {"format": fields, "error_type": None},
            "response": [
                {
                    "contract": {
                        "root": "SPX",
                        "expiration": 20260812,
                        "strike": 5970000,
                        "right": "P",
                    },
                    "ticks": [tick],
                }
            ],
        },
        headers=headers,
    )


def test_download_normalizes_spx_and_0dte_chain() -> None:
    transport = FakeTransport([index_response(), option_response()])
    adapter = ThetaDataAdapter(ThetaDataSettings(), transport=transport)

    result = adapter.download(HistoricalDataRequest(start=date(2026, 8, 12), end=date(2026, 8, 12)))

    bar = result.spx_bars.iloc[0]
    assert bar["timestamp"] == pd.Timestamp("2026-08-12 09:30", tz="America/New_York")
    assert bar["close"] == 6001
    quote = result.option_quotes.iloc[0]
    assert quote["timestamp"] == pd.Timestamp("2026-08-12 10:45", tz="America/New_York")
    assert quote["expiration"] == date(2026, 8, 12)
    assert quote["strike"] == 5970.0
    assert quote["option_type"] == "PUT"
    assert quote["delta"] == pytest.approx(-0.10)
    assert quote["gamma"] == pytest.approx(0.01)
    assert quote["implied_volatility"] == pytest.approx(0.22)
    assert pd.isna(quote["last"])
    assert quote["source"] == "THETADATA_BULK_ALL_GREEKS_V2"
    assert transport.calls[0][1]["ivl"] == 300000
    assert transport.calls[1][1]["exp"] == "20260812"


def test_pagination_follows_next_page_without_reusing_query_parameters() -> None:
    page = "http://127.0.0.1:25510/v2/page/2"
    transport = FakeTransport(
        [index_response(), option_response(next_page=page), option_response()]
    )
    result = ThetaDataAdapter(ThetaDataSettings(), transport=transport).download(
        HistoricalDataRequest(start="2026-08-12", end="2026-08-12")
    )
    assert len(result.option_quotes) == 2
    assert transport.calls[2] == (page, None)


def test_provider_error_is_not_treated_as_empty_data() -> None:
    response = ProviderResponse(
        payload={
            "header": {"format": [], "error_type": "NO_PERMISSION", "error_msg": "Pro required"},
            "response": [],
        },
        headers={},
    )
    with pytest.raises(ProviderConnectionError, match="Pro required"):
        ThetaDataAdapter(ThetaDataSettings(), transport=FakeTransport([response])).download(
            HistoricalDataRequest(start="2026-08-12", end="2026-08-12")
        )


def test_request_rejects_invalid_dates_and_interval() -> None:
    with pytest.raises(ValueError):
        HistoricalDataRequest(start="2026-08-13", end="2026-08-12")
    with pytest.raises(ValueError):
        HistoricalDataRequest(start="2026-08-12", end="2026-08-12", interval_minutes=0)
