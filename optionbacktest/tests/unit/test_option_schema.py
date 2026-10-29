from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from spx_0dte_backtest.data import OptionQuote, OptionType


def valid_quote(**overrides: object) -> OptionQuote:
    values: dict[str, object] = {
        "timestamp": datetime(2026, 8, 12, 10, 45, tzinfo=ZoneInfo("America/New_York")),
        "expiration": date(2026, 8, 12),
        "strike": 6000.0,
        "option_type": OptionType.CALL,
        "bid": 1.0,
        "ask": 1.2,
        "last": 1.1,
        "volume": 10,
        "open_interest": 100,
        "delta": 0.1,
        "gamma": 0.01,
        "theta": -0.5,
        "vega": 0.1,
        "implied_volatility": 0.2,
        "contract_symbol": "SPXW260812C06000000",
        "source": "fixture",
    }
    values.update(overrides)
    return OptionQuote(**values)  # type: ignore[arg-type]


def test_valid_option_quote_is_immutable() -> None:
    quote = valid_quote()
    assert quote.option_type is OptionType.CALL
    with pytest.raises(AttributeError):
        quote.bid = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"strike": 0},
        {"bid": -1},
        {"ask": 0.5, "bid": 1.0},
        {"delta": 1.1},
        {"implied_volatility": -0.1},
        {"volume": -1},
    ],
)
def test_invalid_option_quote_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        valid_quote(**overrides)
