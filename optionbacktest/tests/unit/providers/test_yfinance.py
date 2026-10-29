from __future__ import annotations

import pandas as pd
import pytest

from spx_0dte_backtest.providers import HistoricalDataRequest, YFinanceAdapter


def test_yfinance_emits_thetadata_compatible_spx_columns() -> None:
    index = pd.DatetimeIndex(["2026-08-12 13:29Z", "2026-08-12 13:30Z", "2026-08-12 20:00Z"])
    raw = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0, 1, 2],
            "Close": [1, 2, 3],
            "Volume": [0, 0, 0],
        },
        index=index,
    )
    calls: list[dict[str, object]] = []

    def fake_download(_ticker: str, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return raw

    dataset = YFinanceAdapter(download=fake_download).download(
        HistoricalDataRequest(start="2026-08-12", end="2026-08-12", interval_minutes=1)
    )
    assert list(dataset.spx_bars.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "source",
    ]
    assert len(dataset.spx_bars) == 2
    assert str(dataset.spx_bars["timestamp"].dt.tz) == "America/New_York"
    assert dataset.option_quotes.empty
    assert calls[0]["interval"] == "1m"
    assert calls[0]["end"] == "2026-08-13"


def test_yfinance_rejects_non_one_minute_request() -> None:
    with pytest.raises(ValueError, match="one-minute"):
        YFinanceAdapter(download=lambda *_args, **_kwargs: pd.DataFrame()).download(
            HistoricalDataRequest(start="2026-08-12", end="2026-08-12")
        )
