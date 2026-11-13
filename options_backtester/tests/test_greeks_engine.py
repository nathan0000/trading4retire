from datetime import date

import numpy as np
import pytest

from options_backtester.data.base import normalize_chain
from options_backtester.data.greeks_engine import (
    GreeksConfig,
    baw_price,
    bsm_greeks,
    bsm_price,
    fill_missing_greeks,
    greeks,
    implied_volatility,
)
from tests.conftest import chain_frame


def crr_american(s, k, t, r, q, sigma, is_call, steps=800):
    dt = t / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    prices = s * u**j * d ** (steps - j)
    sign = 1 if is_call else -1
    values = np.maximum(sign * (prices - k), 0)
    for n in range(steps - 1, -1, -1):
        j = np.arange(n + 1)
        prices = s * u**j * d ** (n - j)
        values = np.maximum(disc * (p * values[1:] + (1 - p) * values[:-1]), sign * (prices - k))
    return float(values[0])


def test_bsm_matches_textbook_values_and_parity():
    call = bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, True)
    put = bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, False)
    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)
    assert call - put == pytest.approx(100 - 100 * np.exp(-0.05), abs=1e-10)


def test_bsm_greeks_match_finite_differences():
    s, k, t, r, q, v = 100.0, 105.0, 0.25, 0.04, 0.01, 0.3
    for is_call in (True, False):
        g = bsm_greeks(s, k, t, r, q, v, is_call)
        h = 1e-3
        delta = (
            bsm_price(s + h, k, t, r, q, v, is_call) - bsm_price(s - h, k, t, r, q, v, is_call)
        ) / (2 * h)
        vega = (
            bsm_price(s, k, t, r, q, v + 1e-4, is_call)
            - bsm_price(s, k, t, r, q, v - 1e-4, is_call)
        ) / 2e-4
        theta = (
            bsm_price(s, k, t - 1e-5, r, q, v, is_call) - bsm_price(s, k, t, r, q, v, is_call)
        ) / 1e-5
        assert g["delta"] == pytest.approx(delta, abs=1e-6)
        assert g["vega"] == pytest.approx(vega / 100, abs=1e-6)
        assert g["theta"] == pytest.approx(theta / 365, abs=1e-5)
    assert bsm_greeks(s, k, t, r, q, v, False)["delta"] < 0


def test_implied_volatility_round_trips_and_rejects_arbitrage():
    strikes = np.array([80.0, 100.0, 120.0, 100.0])
    vols = np.array([0.35, 0.2, 0.5, 0.15])
    is_call = np.array([False, True, True, False])
    prices = bsm_price(100.0, strikes, 0.2, 0.03, 0.0, vols, is_call)
    solved = implied_volatility(prices, 100.0, strikes, 0.2, 0.03, 0.0, is_call)
    np.testing.assert_allclose(solved, vols, atol=1e-6)
    # A call priced below intrinsic has no implied volatility.
    assert np.isnan(implied_volatility(5.0, 120.0, 100.0, 0.2, 0.03, 0.0, True))


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "q", "v", "is_call"),
    [
        (100, 100, 0.5, 0.05, 0.0, 0.25, False),
        (90, 100, 0.25, 0.05, 0.0, 0.3, False),
        (100, 100, 0.5, 0.05, 0.08, 0.25, True),
        (110, 100, 0.25, 0.03, 0.06, 0.2, True),
    ],
)
def test_baw_tracks_binomial_american_price(s, k, t, r, q, v, is_call):
    tree = crr_american(s, k, t, r, q, v, is_call)
    approx = float(baw_price(s, k, t, r, q, v, is_call))
    assert approx == pytest.approx(tree, rel=0.02)
    assert approx >= float(bsm_price(s, k, t, r, q, v, is_call)) - 1e-9


def test_baw_call_without_dividend_equals_european():
    assert float(baw_price(100, 95, 0.5, 0.05, 0.0, 0.2, True)) == pytest.approx(
        float(bsm_price(100, 95, 0.5, 0.05, 0.0, 0.2, True))
    )


def test_baw_greeks_have_sane_signs():
    g = greeks(100.0, 100.0, 0.5, 0.05, 0.0, 0.25, False, model="baw")
    assert -1 < g["delta"] < 0 and g["gamma"] > 0 and g["vega"] > 0 and g["theta"] < 0


def test_fill_missing_greeks_fills_only_missing_rows_and_labels_them():
    day = date(2024, 1, 8)
    frame = chain_frame(day, 100.0, [(95, "PUT", 1.0, 1.1, -0.2), (105, "CALL", 0.9, 1.0, 0.2)])
    frame.loc[1, ["delta", "gamma", "theta", "vega", "implied_volatility"]] = np.nan
    out = fill_missing_greeks(normalize_chain(frame), GreeksConfig())
    by_type = out.set_index("option_type")
    assert by_type.loc["PUT", "delta"] == -0.2  # provider value untouched
    assert by_type.loc["PUT", "greeks_source"] == "PROVIDER"
    assert 0 < by_type.loc["CALL", "delta"] < 0.5
    assert by_type.loc["CALL", "implied_volatility"] > 0
    assert by_type.loc["CALL", "greeks_source"] == "ESTIMATED_BSM"
