from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from spx_0dte_backtest.strategies import (
    IronCondorSettings,
    IronCondorStrategy,
    MarketContext,
    RejectionReason,
    StrategyType,
    TradeLegRole,
)

ENTRY = pd.Timestamp("2026-08-12 10:45", tz="America/New_York")
EXPIRATION = date(2026, 8, 12)


def quote(
    strike: float,
    option_type: str,
    delta: float,
    bid: float,
    ask: float,
    *,
    timestamp: pd.Timestamp = ENTRY,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "expiration": EXPIRATION,
        "strike": strike,
        "option_type": option_type,
        "bid": bid,
        "ask": ask,
        "last": (bid + ask) / 2,
        "volume": 10,
        "open_interest": 100,
        "delta": delta,
        "gamma": 0.01,
        "theta": -0.5,
        "vega": 0.1,
        "implied_volatility": 0.2,
        "contract_symbol": f"{option_type}_{strike}",
        "source": "fixture",
    }


def chain(*, credit_shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            quote(5940, "PUT", -0.04, 0.30, 0.40),
            quote(5970, "PUT", -0.10, 1.20 + credit_shift, 1.40 + credit_shift),
            quote(6030, "CALL", 0.10, 1.30 + credit_shift, 1.50 + credit_shift),
            quote(6060, "CALL", 0.04, 0.30, 0.40),
        ]
    )


def context(timestamp: pd.Timestamp = ENTRY, quotes: pd.DataFrame | None = None) -> MarketContext:
    available_quotes = chain() if quotes is None else quotes
    if quotes is None:
        available_quotes["timestamp"] = timestamp
    return MarketContext(
        timestamp=timestamp,
        expiration=timestamp.tz_convert("America/New_York").date(),
        spx_price=6000.0,
        option_quotes=available_quotes,
        features={"regime": "RANGE", "data_cutoff_timestamp": timestamp},
    )


def test_constructs_candidate_and_calculates_both_quoted_credits() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    assert candidate.is_accepted
    assert candidate.strategy is StrategyType.IRON_CONDOR
    assert candidate.conservative_executable_credit == pytest.approx(1.70)
    assert candidate.theoretical_mid_credit == pytest.approx(2.00)
    assert candidate.put_credit == pytest.approx(0.80)
    assert candidate.call_credit == pytest.approx(0.90)
    assert candidate.max_profit_points == pytest.approx(1.70)
    assert candidate.max_loss_points == pytest.approx(28.30)
    assert [leg.role for leg in candidate.legs] == [
        TradeLegRole.LONG_PUT,
        TradeLegRole.SHORT_PUT,
        TradeLegRole.SHORT_CALL,
        TradeLegRole.LONG_CALL,
    ]
    assert candidate.short_put_delta == pytest.approx(-0.10)
    assert candidate.short_call_delta == pytest.approx(0.10)


@pytest.mark.parametrize(
    "timestamp",
    [
        pd.Timestamp("2026-08-12 10:45", tz="America/New_York"),
        pd.Timestamp("2026-08-12 15:15", tz="America/New_York"),
        pd.Timestamp("2026-08-12 14:45", tz="UTC"),  # 10:45 ET
    ],
)
def test_accepts_eligible_entry_times(timestamp: pd.Timestamp) -> None:
    assert IronCondorStrategy().generate_trade(context(timestamp)).is_accepted


@pytest.mark.parametrize(
    "timestamp",
    [
        pd.Timestamp("2026-08-12 10:30", tz="America/New_York"),
        pd.Timestamp("2026-08-12 10:50", tz="America/New_York"),
        pd.Timestamp("2026-08-12 15:30", tz="America/New_York"),
    ],
)
def test_rejects_times_outside_entry_schedule(timestamp: pd.Timestamp) -> None:
    candidate = IronCondorStrategy().generate_trade(context(timestamp))
    assert candidate.rejection_reason is RejectionReason.OUTSIDE_TRADING_WINDOW
    assert candidate.legs == ()


def test_rejects_credit_below_minimum() -> None:
    low = chain(credit_shift=-0.40)
    candidate = IronCondorStrategy().generate_trade(context(quotes=low))
    assert candidate.rejection_reason is RejectionReason.CREDIT_TOO_LOW
    assert candidate.conservative_executable_credit == pytest.approx(0.90)
    assert candidate.legs == ()


def test_rejects_credit_above_maximum() -> None:
    high = chain(credit_shift=0.50)
    candidate = IronCondorStrategy().generate_trade(context(quotes=high))
    assert candidate.rejection_reason is RejectionReason.CREDIT_TOO_HIGH
    assert candidate.conservative_executable_credit == pytest.approx(2.70)


def test_credit_boundaries_are_inclusive() -> None:
    settings = IronCondorSettings(min_credit=1.70, max_credit=1.70)
    assert IronCondorStrategy(settings).generate_trade(context()).is_accepted


def test_maps_strike_selection_failure_to_rejection() -> None:
    missing_wing = chain()[lambda frame: frame["strike"] != 5940]
    candidate = IronCondorStrategy().generate_trade(context(quotes=missing_wing))
    assert candidate.rejection_reason is RejectionReason.NO_VALID_STRIKE
    assert candidate.legs == ()


def test_no_snapshot_maps_to_no_option_data() -> None:
    earlier = chain().copy()
    earlier["timestamp"] = ENTRY - pd.Timedelta("5min")
    candidate = IronCondorStrategy().generate_trade(context(quotes=earlier))
    assert candidate.rejection_reason is RejectionReason.NO_OPTION_DATA


def test_context_rejects_future_feature_cutoff() -> None:
    with pytest.raises(ValueError, match="feature cutoff"):
        MarketContext(
            timestamp=ENTRY,
            expiration=EXPIRATION,
            spx_price=6000,
            option_quotes=chain(),
            features={"data_cutoff_timestamp": ENTRY + pd.Timedelta("5min")},
        )


def test_context_and_candidate_are_immutable_at_boundaries() -> None:
    source = chain()
    market = context(quotes=source)
    source.loc[:, "bid"] = 99.0
    assert market.option_quotes.iloc[0]["bid"] != 99.0
    with pytest.raises(TypeError):
        market.features["regime"] = "TREND_UP"  # type: ignore[index]


def test_settings_validate_credit_and_schedule() -> None:
    with pytest.raises(ValueError):
        IronCondorSettings(min_credit=2.5, max_credit=1.0)
    with pytest.raises(ValueError):
        IronCondorSettings(entry_interval_minutes=0)
