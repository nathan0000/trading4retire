from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from spx_0dte_backtest.data import MarketBar


def test_valid_market_bar() -> None:
    bar = MarketBar(
        "SPX",
        datetime(2026, 8, 12, 9, 30, tzinfo=ZoneInfo("America/New_York")),
        6000,
        6002,
        5999,
        6001,
        None,
        "fixture",
    )
    assert bar.close == 6001


@pytest.mark.parametrize(
    "kwargs",
    [
        {"open": -1},
        {"high": 5998},
        {"low": 6003},
        {"volume": -1},
    ],
)
def test_invalid_market_bar(kwargs: dict[str, float]) -> None:
    values = dict(open=6000, high=6002, low=5999, close=6001, volume=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        MarketBar(
            symbol="SPX",
            timestamp=datetime(2026, 8, 12, 9, 30, tzinfo=ZoneInfo("America/New_York")),
            source="fixture",
            **values,
        )


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketBar("SPX", datetime(2026, 8, 12, 9, 30), 6000, 6002, 5999, 6001)
