from __future__ import annotations

import pandas as pd
import pytest

from spx_0dte_backtest.data.greeks import GreeksFallbackSettings, fill_missing_greeks
from spx_0dte_backtest.data.loaders import OptionChainLoader


def incomplete_quote(**overrides):
    values = {
        "timestamp": pd.Timestamp("2026-08-12 10:45", tz="America/New_York"),
        "expiration": pd.Timestamp("2026-08-12"),
        "strike": 6000.0,
        "option_type": "CALL",
        "bid": 10.0,
        "ask": 10.2,
        "underlying_price": 6000.0,
        "delta": None,
        "gamma": None,
        "theta": None,
        "vega": None,
        "implied_volatility": None,
        "source": "THETADATA_SAMPLE",
    }
    values.update(overrides)
    return values


def test_fallback_solves_iv_and_greeks_without_mutating_input():
    frame = pd.DataFrame([incomplete_quote()])
    result = fill_missing_greeks(frame)
    assert frame["delta"].isna().all()
    assert 0 < result.loc[0, "implied_volatility"] < 5
    assert 0 < result.loc[0, "delta"] < 1
    assert result.loc[0, "gamma"] > 0
    assert result.loc[0, "theta"] < 0
    assert result.loc[0, "vega"] > 0
    assert result.loc[0, "source"].endswith("ESTIMATED_BLACK_SCHOLES_GREEKS")


def test_fallback_preserves_vendor_values():
    supplied = incomplete_quote(delta=0.1, gamma=0.2, theta=-0.3, vega=0.4, implied_volatility=0.25)
    result = fill_missing_greeks(pd.DataFrame([supplied]))
    assert result.loc[0, "delta"] == pytest.approx(0.1)
    assert result.loc[0, "source"] == "THETADATA_SAMPLE"


def test_unsolvable_quote_remains_missing_for_validation():
    result = fill_missing_greeks(pd.DataFrame([incomplete_quote(bid=0.0, ask=0.0)]))
    assert result.loc[0, ["delta", "gamma", "theta", "vega"]].isna().all()


def test_loader_fallback_is_opt_in(config, write_csv):
    frame = pd.DataFrame(
        [
            {
                **incomplete_quote(),
                "timestamp": "2026-08-12T10:45:00-04:00",
                "expiration": "2026-08-12",
                "last": 10.1,
                "volume": 1,
                "open_interest": 1,
                "contract_symbol": "SPXW_TEST",
            }
        ]
    )
    path = write_csv(frame.to_csv(index=False))
    default = OptionChainLoader(config).load(path)
    assert default.validation.counts_by_code["MISSING_GREEKS"] == 1
    estimated = OptionChainLoader(config, greeks_fallback=GreeksFallbackSettings()).load(path)
    assert estimated.validation.is_valid
    assert estimated.validation.accepted_rows == 1


def test_loader_fallback_requires_point_in_time_underlying(config, write_csv):
    frame = pd.DataFrame(
        [{**incomplete_quote(), "last": 10.1, "volume": 1, "open_interest": 1}]
    ).drop(columns="underlying_price")
    path = write_csv(frame.to_csv(index=False))
    with pytest.raises(ValueError, match="underlying_price"):
        OptionChainLoader(config, greeks_fallback=GreeksFallbackSettings()).load(path)
