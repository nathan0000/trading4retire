from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from options_backtester.data.base import OptionChainSnapshot, normalize_chain
from options_backtester.regimes.classifier import RegimeState
from options_backtester.strategies import build_strategy
from options_backtester.strategies.base import (
    EntryRules,
    ExitRules,
    LegBuilder,
    MarketContext,
    Rejection,
    StrategyConfig,
    StrikeRule,
    WingRule,
)
from options_backtester.strategies.conditions import (
    ExitReason,
    PositionState,
    check_fill,
    evaluate_exit,
    triggered_exits,
)
from tests.conftest import chain_frame

DAY = date(2024, 1, 8)
BACK = date(2024, 3, 15)

ROWS = [
    # strike, type, bid, ask, delta
    (85, "PUT", 0.10, 0.20, -0.04),
    (90, "PUT", 0.40, 0.50, -0.08),
    (93, "PUT", 0.70, 0.80, -0.14),
    (95, "PUT", 1.00, 1.10, -0.18),
    (100, "PUT", 2.90, 3.10, -0.48),
    (100, "CALL", 3.00, 3.20, 0.52),
    (105, "CALL", 1.00, 1.10, 0.18),
    (107, "CALL", 0.70, 0.80, 0.14),
    (110, "CALL", 0.40, 0.50, 0.08),
    (115, "CALL", 0.10, 0.20, 0.04),
    (100, "PUT", 4.00, 4.30, -0.47, BACK),
    (100, "CALL", 4.20, 4.50, 0.53, BACK),
]


def snapshot(rows=ROWS, spot=100.0):
    return OptionChainSnapshot.from_frame(normalize_chain(chain_frame(DAY, spot, rows)))


def side(option_type, rows=ROWS):
    chain = snapshot(rows)
    return LegBuilder.side_frame(chain.quotes, pd.Timestamp("2024-02-16"), option_type)


def context(regime=None, rows=ROWS):
    chain = snapshot(rows)
    return MarketContext(chain.timestamp, chain, regime)


def regime(trend="SIDEWAYS", volatility="HIGH_IV", iv_rank=70.0):
    return RegimeState(
        pd.Timestamp("2024-01-08 21:00", tz="UTC"), trend, volatility, "STABLE", "UNKNOWN", iv_rank
    )


# -- leg selection -------------------------------------------------------


def test_delta_selection_picks_nearest_absolute_delta():
    assert LegBuilder.pick_strike(side("PUT"), StrikeRule(value=0.17), 100)["strike"] == 95
    assert LegBuilder.pick_strike(side("CALL"), StrikeRule(value=0.09), 100)["strike"] == 110


def test_delta_tie_resolves_to_the_further_otm_strike():
    # 0.16 is equidistant from 0.15 and 0.17 on both sides.
    rows = [
        (93, "PUT", 0.7, 0.8, -0.15),
        (95, "PUT", 1.0, 1.1, -0.17),
        (105, "CALL", 1.0, 1.1, 0.17),
        (107, "CALL", 0.7, 0.8, 0.15),
    ]
    assert LegBuilder.pick_strike(side("PUT", rows), StrikeRule(value=0.16), 100)["strike"] == 93
    assert LegBuilder.pick_strike(side("CALL", rows), StrikeRule(value=0.16), 100)["strike"] == 107


def test_delta_selection_ignores_missing_deltas_and_enforces_tolerance():
    rows = [(95, "PUT", 1.0, 1.1, np.nan), (90, "PUT", 0.4, 0.5, -0.08)]
    frame = side("PUT", rows)
    frame.loc[frame["strike"] == 95, "delta"] = np.nan  # provider gave no Greek here
    assert LegBuilder.pick_strike(frame, StrikeRule(value=0.10), 100)["strike"] == 90
    assert LegBuilder.pick_strike(frame, StrikeRule(value=0.30), 100) is None  # 0.22 away
    frame["delta"] = np.nan
    assert LegBuilder.pick_strike(frame, StrikeRule(value=0.10), 100) is None


def test_pct_otm_and_stdev_selection():
    assert (
        LegBuilder.pick_strike(side("PUT"), StrikeRule(method="pct_otm", value=0.05), 100)["strike"]
        == 95
    )
    assert (
        LegBuilder.pick_strike(side("CALL"), StrikeRule(method="pct_otm", value=0.10), 100)[
            "strike"
        ]
        == 110
    )
    # 1 sd = 100 * 0.2 * sqrt(39/365) = 6.5 points
    assert (
        LegBuilder.pick_strike(side("PUT"), StrikeRule(method="stdev", value=1.0), 100)["strike"]
        == 93
    )
    assert (
        LegBuilder.pick_strike(side("CALL"), StrikeRule(method="stdev", value=1.0), 100)["strike"]
        == 107
    )


def test_wing_selection_by_dollars_delta_and_percent():
    puts, calls = side("PUT"), side("CALL")
    short_put = puts[puts["strike"] == 95].iloc[0]
    short_call = calls[calls["strike"] == 105].iloc[0]
    assert LegBuilder.pick_wing(puts, short_put, WingRule(value=5), 100)["strike"] == 90
    assert LegBuilder.pick_wing(calls, short_call, WingRule(value=5), 100)["strike"] == 110
    assert (
        LegBuilder.pick_wing(puts, short_put, WingRule(method="delta", value=0.10), 100)["strike"]
        == 90
    )
    assert (
        LegBuilder.pick_wing(calls, short_call, WingRule(method="pct_underlying", value=0.10), 100)[
            "strike"
        ]
        == 115
    )
    # The wing must be strictly further OTM even when the requested width is tiny.
    assert LegBuilder.pick_wing(puts, short_put, WingRule(value=0.5), 100)["strike"] == 93
    lowest = puts[puts["strike"] == 85].iloc[0]
    assert LegBuilder.pick_wing(puts, lowest, WingRule(value=5), 100) is None


def test_expiration_selection_respects_the_dte_window():
    quotes = snapshot().quotes
    assert LegBuilder.pick_expiration(quotes, 30, 45) == pd.Timestamp("2024-02-16")
    assert LegBuilder.pick_expiration(quotes, 60, 90) == pd.Timestamp("2024-03-15")
    assert LegBuilder.pick_expiration(quotes, 30, 90, target_dte=70) == pd.Timestamp("2024-03-15")
    assert LegBuilder.pick_expiration(quotes, 5, 20) is None


# -- templates -----------------------------------------------------------

BASE = {"min_dte": 30, "max_dte": 45, "strike": {"method": "delta", "value": 0.18}}


def legs_of(config, **kwargs):
    candidate = build_strategy(config).generate(context(**kwargs))
    assert not isinstance(candidate, Rejection), candidate
    return [(leg.strike, leg.option_type, leg.side) for leg in candidate.legs], candidate


def test_iron_condor_structure_and_credit():
    legs, candidate = legs_of({**BASE, "template": "iron_condor", "wing": {"value": 5}})
    assert legs == [(95, "PUT", -1), (90, "PUT", 1), (105, "CALL", -1), (110, "CALL", 1)]
    assert candidate.mid_value == pytest.approx(-(1.05 - 0.45 + 1.05 - 0.45))


def test_vertical_credit_and_debit():
    credit, c = legs_of(
        {**BASE, "template": "vertical", "wing": {"value": 5}, "option_side": "put"}
    )
    assert credit == [(95, "PUT", -1), (90, "PUT", 1)] and c.mid_value < 0
    debit, d = legs_of(
        {
            **BASE,
            "template": "vertical",
            "wing": {"value": 5},
            "option_side": "call",
            "direction": "debit",
        }
    )
    assert debit == [(105, "CALL", 1), (110, "CALL", -1)] and d.mid_value > 0


def test_strangle_and_calendar():
    strangle, _ = legs_of({**BASE, "template": "strangle"})
    assert strangle == [(95, "PUT", -1), (105, "CALL", -1)]
    config = {
        "template": "calendar",
        "min_dte": 30,
        "max_dte": 45,
        "back_min_dte": 60,
        "back_max_dte": 75,
        "strike": {"method": "delta", "value": 0.5},
        "option_side": "call",
    }
    candidate = build_strategy(config).generate(context())
    assert [(leg.expiration, leg.strike, leg.side) for leg in candidate.legs] == [
        (date(2024, 2, 16), 100, -1),
        (BACK, 100, 1),
    ]
    assert candidate.mid_value > 0  # long calendar is a debit


def test_config_validation():
    with pytest.raises(ValidationError):
        StrategyConfig(template="iron_condor")  # wing required
    with pytest.raises(ValidationError):
        StrategyConfig(template="strangle", min_dte=50, max_dte=40)
    with pytest.raises(ValidationError):
        StrategyConfig(template="strangle", unknown_field=1)


# -- entry conditions ----------------------------------------------------


def reason(entry, **kwargs):
    config = {**BASE, "template": "iron_condor", "wing": {"value": 5}, "entry": entry}
    result = build_strategy(config).generate(context(**kwargs))
    return result.reason if isinstance(result, Rejection) else None


def test_regime_and_iv_rank_filters():
    allowed = {"allowed_trend": ["SIDEWAYS"], "allowed_volatility": ["HIGH_IV"]}
    assert reason(allowed, regime=regime()) is None
    assert reason(allowed, regime=regime(trend="BEARISH")) == "REGIME_FILTER"
    assert reason(allowed, regime=None) == "REGIME_FILTER"
    assert reason({"min_iv_rank": 50}, regime=regime(iv_rank=30)) == "IV_RANK_TOO_LOW"
    assert reason({"min_iv_rank": 50}, regime=None) == "IV_RANK_UNAVAILABLE"
    assert reason({"min_iv_rank": 50}, regime=regime(iv_rank=70)) is None


def test_liquidity_filters():
    assert reason({"max_spread_pct": 0.05}) == "SPREAD_TOO_WIDE"
    assert reason({"min_open_interest": 10_000}) == "OPEN_INTEREST_TOO_LOW"
    zero_bid = [r if r[0] != 95 else (95, "PUT", 0.0, 0.10, -0.18) for r in ROWS]
    assert reason({}, rows=zero_bid) == "ZERO_BID"
    assert reason({"entry_time": "10:30"}) == "OUTSIDE_ENTRY_TIME"
    assert reason({"entry_time": "16:00"}) is None


def test_credit_filter_uses_signed_fill_value():
    rules = EntryRules(min_credit=1.0, max_credit=2.0)
    assert check_fill(rules, -1.5) is None
    assert check_fill(rules, -0.9).reason == "CREDIT_TOO_LOW"
    assert check_fill(rules, -2.5).reason == "CREDIT_TOO_HIGH"
    assert check_fill(rules, 0.3).reason == "CREDIT_TOO_LOW"  # a debit is not a credit
    assert check_fill(EntryRules(max_debit=1.0), 1.5).reason == "DEBIT_TOO_HIGH"


# -- exit conditions -----------------------------------------------------


def state(**overrides):
    values = dict(basis=1.0, pnl=0.0, peak_pnl=0.0, min_dte=30, days_held=5, underlying_price=100.0)
    return PositionState(**{**values, **overrides})


def test_stop_loss_outranks_dte_exit_on_the_same_bar():
    rules = ExitRules(profit_target_pct=0.5, stop_loss_multiple=2.0, exit_dte=21)
    both = state(pnl=-2.0, min_dte=21)
    assert triggered_exits(rules, both) == [ExitReason.STOP_LOSS, ExitReason.DTE_EXIT]
    assert evaluate_exit(rules, both) is ExitReason.STOP_LOSS
    assert evaluate_exit(rules, state(pnl=0.5, min_dte=21)) is ExitReason.PROFIT_TARGET
    assert evaluate_exit(rules, state(pnl=-1.99, min_dte=22)) is None
    assert evaluate_exit(rules, state(pnl=-5.0, min_dte=0)) is ExitReason.EXPIRATION


def test_stop_multiple_is_a_loss_not_a_cost_to_close():
    # 2x stop on a 1.50 credit triggers at a 3.00 loss, i.e. a 4.50 cost to close.
    rules = ExitRules(stop_loss_multiple=2.0, profit_target_pct=None, exit_dte=None)
    assert evaluate_exit(rules, state(basis=1.5, pnl=-(4.49 - 1.5))) is None
    assert evaluate_exit(rules, state(basis=1.5, pnl=-(4.50 - 1.5))) is ExitReason.STOP_LOSS


def test_breach_delta_trailing_hold_and_regime_exits():
    off = dict(profit_target_pct=None, stop_loss_multiple=None, exit_dte=None)
    breach = ExitRules(**off, underlying_breach=True)
    assert (
        evaluate_exit(breach, state(short_put_strikes=(95.0,), underlying_price=94.0))
        is ExitReason.UNDERLYING_BREACH
    )
    assert evaluate_exit(breach, state(short_call_strikes=(105.0,), underlying_price=104.0)) is None
    delta = ExitRules(**off, max_position_delta=0.3)
    assert evaluate_exit(delta, state(position_delta=-0.31)) is ExitReason.DELTA_BREACH
    assert evaluate_exit(delta, state(position_delta=float("nan"))) is None
    trail = ExitRules(**off, trailing_stop_pct=0.2, trailing_activation_pct=0.3)
    assert evaluate_exit(trail, state(peak_pnl=0.4, pnl=0.2)) is ExitReason.TRAILING_STOP
    assert evaluate_exit(trail, state(peak_pnl=0.25, pnl=0.0)) is None  # never armed
    assert (
        evaluate_exit(ExitRules(**off, max_hold_days=5), state(days_held=5)) is ExitReason.MAX_HOLD
    )
    shift = ExitRules(**off, exit_on_regime_shift=True)
    held = state(entry_regime=regime(), regime=regime(volatility="LOW_IV"))
    assert evaluate_exit(shift, held) is ExitReason.REGIME_SHIFT
    assert evaluate_exit(shift, state(entry_regime=regime(), regime=regime())) is None
    allowed = EntryRules(allowed_volatility=["HIGH_IV", "LOW_IV"])
    assert evaluate_exit(shift, held, allowed) is None  # still inside the allowed set
