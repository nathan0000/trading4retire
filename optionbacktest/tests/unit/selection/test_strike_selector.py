from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from spx_0dte_backtest.selection import (
    SelectionFailure,
    StrikeSelectionSettings,
    StrikeSelector,
)

TIMESTAMP = pd.Timestamp("2026-08-12 10:45", tz="America/New_York")
EXPIRATION = date(2026, 8, 12)


def quote(
    strike: float, option_type: str, delta: float, *, timestamp=TIMESTAMP
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "expiration": EXPIRATION,
        "strike": strike,
        "option_type": option_type,
        "bid": 1.0,
        "ask": 1.2,
        "last": 1.1,
        "volume": 10,
        "open_interest": 100,
        "delta": delta,
        "gamma": 0.01,
        "theta": -0.5,
        "vega": 0.1,
        "implied_volatility": 0.2,
        "contract_symbol": f"{option_type}_{strike}_{timestamp.hour}{timestamp.minute}",
        "source": "fixture",
    }


def valid_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            quote(5940, "PUT", -0.04),
            quote(5970, "PUT", -0.10),
            quote(5980, "PUT", -0.12),
            quote(6000, "PUT", -0.25),
            quote(6030, "CALL", 0.10),
            quote(6040, "CALL", 0.12),
            quote(6060, "CALL", 0.04),
        ]
    )


def test_selects_closest_target_delta_and_exact_thirty_point_wings() -> None:
    result = StrikeSelector().select(valid_chain(), TIMESTAMP, EXPIRATION, underlying_price=6000)
    assert result.is_success
    assert result.failure is None
    assert result.short_put is not None and result.short_put.strike == 5970
    assert result.long_put is not None and result.long_put.strike == 5940
    assert result.short_call is not None and result.short_call.strike == 6030
    assert result.long_call is not None and result.long_call.strike == 6060
    assert result.short_put.delta == pytest.approx(-0.10)
    assert result.short_call.delta == pytest.approx(0.10)


def test_uses_absolute_delta_for_puts_but_preserves_signed_delta() -> None:
    chain = valid_chain()
    result = StrikeSelector().select(chain, TIMESTAMP, EXPIRATION, underlying_price=6000)
    assert result.short_put is not None
    assert result.short_put.delta == -0.10


def test_filters_to_exact_snapshot_and_expiration() -> None:
    future = TIMESTAMP + pd.Timedelta("5min")
    chain = pd.concat(
        [
            valid_chain(),
            pd.DataFrame([quote(5990, "PUT", -0.10, timestamp=future)]),
        ],
        ignore_index=True,
    )
    result = StrikeSelector().select(chain, TIMESTAMP, EXPIRATION, underlying_price=6000)
    assert result.is_success
    assert result.short_put is not None and result.short_put.timestamp == TIMESTAMP
    assert all(leg.expiration == EXPIRATION for leg in result.legs)


def test_does_not_use_a_wing_from_a_future_snapshot() -> None:
    chain = valid_chain()
    chain = chain[chain["strike"] != 5940]
    chain = pd.concat(
        [
            chain,
            pd.DataFrame([quote(5940, "PUT", -0.04, timestamp=TIMESTAMP + pd.Timedelta("5min"))]),
        ],
        ignore_index=True,
    )
    result = StrikeSelector().select(chain, TIMESTAMP, EXPIRATION, underlying_price=6000)
    assert not result.is_success
    assert result.failure == SelectionFailure.NO_VALID_WING


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda frame: frame.iloc[0:0], SelectionFailure.NO_OPTION_DATA),
        (
            lambda frame: frame.assign(delta=0.30),
            SelectionFailure.NO_VALID_STRIKE,
        ),
        (
            lambda frame: frame[frame["strike"] != 6060],
            SelectionFailure.NO_VALID_WING,
        ),
    ],
)
def test_returns_structured_failure(mutation, failure) -> None:
    result = StrikeSelector().select(
        mutation(valid_chain()), TIMESTAMP, EXPIRATION, underlying_price=6000
    )
    assert not result.is_success
    assert result.failure == failure
    assert result.legs == ()


def test_rejects_non_0dte_request() -> None:
    result = StrikeSelector().select(
        valid_chain(), TIMESTAMP, date(2026, 8, 13), underlying_price=6000
    )
    assert result.failure == SelectionFailure.INCORRECT_EXPIRATION


def test_requires_otm_short_strikes() -> None:
    chain = valid_chain()
    chain.loc[chain["option_type"] == "PUT", "strike"] += 100
    result = StrikeSelector().select(chain, TIMESTAMP, EXPIRATION, underlying_price=6000)
    assert result.failure == SelectionFailure.NO_VALID_STRIKE


def test_tie_breaking_is_deterministic_and_prefers_more_otm_short() -> None:
    chain = valid_chain()
    chain.loc[chain["strike"] == 5970, "delta"] = -0.09
    chain.loc[chain["strike"] == 5980, "delta"] = -0.11
    chain = pd.concat([chain, pd.DataFrame([quote(5950, "PUT", -0.04)])], ignore_index=True)
    result = StrikeSelector().select(
        chain.sample(frac=1, random_state=7), TIMESTAMP, EXPIRATION, 6000
    )
    assert result.is_success
    assert result.short_put is not None and result.short_put.strike == 5970


def test_settings_validate_delta_range_and_wing_width() -> None:
    with pytest.raises(ValueError):
        StrikeSelectionSettings(min_abs_delta=0.15, max_abs_delta=0.08)
    with pytest.raises(ValueError):
        StrikeSelectionSettings(wing_width=0)
