"""Execution, portfolio accounting and the backtest loop on hand-built chains."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from options_backtester.data.base import DataUnavailableError
from options_backtester.engine.backtester import BacktestConfig, Backtester
from options_backtester.engine.execution import ExecutionConfig, ExecutionModel, FillModel
from options_backtester.engine.portfolio import margin_per_unit, max_loss_per_unit
from options_backtester.regimes.classifier import RegimeTimeline
from options_backtester.strategies import build_strategy
from options_backtester.strategies.base import Leg, LegQuote
from tests.conftest import EXPIRY, ScriptedProvider, chain_frame

D1, D2, D3 = date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)
FEE = 0.65 + 0.05

PUT_CREDIT = {
    "name": "pcs",
    "template": "vertical",
    "min_dte": 30,
    "max_dte": 45,
    "strike": {"method": "delta", "value": 0.20},
    "wing": {"method": "dollar", "value": 5},
    "exit": {"profit_target_pct": 0.5, "stop_loss_multiple": 2.0, "exit_dte": 21},
}
CALL_DEBIT = {
    **PUT_CREDIT,
    "name": "cds",
    "option_side": "call",
    "direction": "debit",
    "strike": {"method": "delta", "value": 0.50},
}


def puts(short, long_):
    """95/90 put quotes as (bid, ask) pairs, plus an ATM call spread for debit tests."""
    return [(95, "PUT", *short, -0.20), (90, "PUT", *long_, -0.08)]


def calls(long_, short):
    return [(100, "CALL", *long_, 0.50), (105, "CALL", *short, 0.20)]


def run(strategy, chains, closes=None, fill_model="MID", **config):
    closes = closes or {day: 100.0 for day in chains}
    provider = ScriptedProvider(closes, {d: c for d, c in chains.items() if c is not None})
    backtest = BacktestConfig(
        symbol="TEST",
        start=min(closes),
        end=max(closes),
        execution=ExecutionConfig(fill_model=fill_model),
        **config,
    )
    return Backtester(provider, build_strategy(strategy), backtest).run()


# -- execution -----------------------------------------------------------

QUOTE = LegQuote(bid=1.00, ask=1.20, mid=1.10, volume=1000)


def test_fill_models_price_each_side():
    assert ExecutionModel(ExecutionConfig(fill_model="MID")).leg_price(1, QUOTE) == pytest.approx(
        1.10
    )
    natural = ExecutionModel(ExecutionConfig(fill_model="NATURAL"))
    assert natural.leg_price(1, QUOTE) == pytest.approx(1.20)  # buy at the ask
    assert natural.leg_price(-1, QUOTE) == pytest.approx(1.00)  # sell at the bid
    slip = ExecutionModel(ExecutionConfig(fill_model="SLIPPAGE", spread_fraction=0.25))
    assert slip.leg_price(1, QUOTE) == pytest.approx(1.15)
    assert slip.leg_price(-1, QUOTE) == pytest.approx(1.05)
    thin = LegQuote(bid=1.00, ask=1.20, mid=1.10, volume=0)
    assert slip.leg_price(1, thin) == pytest.approx(1.20)  # no volume: pay the full spread


def test_combo_fill_is_atomic_and_signed():
    legs = (Leg(EXPIRY, 95, "PUT", -1), Leg(EXPIRY, 90, "PUT", 1))
    quotes = (LegQuote(1.00, 1.10, 1.05, 1000), LegQuote(0.40, 0.50, 0.45, 1000))
    model = ExecutionModel(ExecutionConfig(fill_model=FillModel.NATURAL))
    opening = model.fill(legs, quotes)
    assert opening.value == pytest.approx(-(1.00 - 0.50))  # credit of 0.50
    assert opening.slippage == pytest.approx(0.10)
    closing = model.fill(legs, quotes, closing=True)
    assert closing.value == pytest.approx(1.10 - 0.40)  # pay 0.70 to close
    broken = (quotes[0], LegQuote(float("nan"), 0.50, float("nan")))
    assert model.fill(legs, broken) is None
    assert model.costs(legs, 3) == (pytest.approx(6 * 0.65), pytest.approx(6 * 0.05))


def test_max_loss_and_margin():
    condor = (
        Leg(EXPIRY, 95, "PUT", -1),
        Leg(EXPIRY, 90, "PUT", 1),
        Leg(EXPIRY, 105, "CALL", -1),
        Leg(EXPIRY, 110, "CALL", 1),
    )
    assert max_loss_per_unit(condor, -1.2) == pytest.approx(3.8)
    debit = (Leg(EXPIRY, 100, "CALL", 1), Leg(EXPIRY, 105, "CALL", -1))
    assert max_loss_per_unit(debit, 2.0) == pytest.approx(2.0)
    strangle = (Leg(EXPIRY, 95, "PUT", -1), Leg(EXPIRY, 105, "CALL", -1))
    assert max_loss_per_unit(strangle, -2.0) == float("inf")
    quotes = (LegQuote(1.0, 1.1, 1.05), LegQuote(1.0, 1.1, 1.05))
    # Reg-T style: 20% of 100 less 5 OTM, plus premium, plus the other side's premium.
    assert margin_per_unit(strangle, quotes, -2.0, 100.0) == pytest.approx(15 + 1.05 + 1.05)
    calendar = (Leg(EXPIRY, 100, "CALL", -1), Leg(date(2024, 3, 15), 100, "CALL", 1))
    assert max_loss_per_unit(calendar, 1.2) == pytest.approx(1.2)


# -- multi-leg P&L -------------------------------------------------------


def test_credit_spread_profit_target_pnl():
    chains = {
        D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50))),
        D2: chain_frame(D2, 101, puts((0.45, 0.55), (0.15, 0.25))),
        D3: chain_frame(D3, 101, puts((0.45, 0.55), (0.15, 0.25))),
    }
    strategy = {**PUT_CREDIT, "entry": {"min_days_between_entries": 30}}
    result = run(strategy, chains)
    trade = result.trades.iloc[0]
    assert len(result.trades) == 1
    assert trade["entry_value"] == pytest.approx(-0.60)  # 1.05 - 0.45 credit
    assert trade["exit_value"] == pytest.approx(-0.30)
    assert trade["exit_reason"] == "PROFIT_TARGET"
    assert trade["gross_pnl"] == pytest.approx(30.0)
    assert trade["commission"] + trade["fees"] == pytest.approx(4 * FEE)
    assert trade["net_pnl"] == pytest.approx(30.0 - 4 * FEE)
    assert trade["max_loss"] == pytest.approx(440.0)
    legs = result.legs.set_index("strike")
    assert legs.loc[95, "pnl"] == pytest.approx(55.0) and legs.loc[90, "pnl"] == pytest.approx(
        -25.0
    )
    assert legs.loc[95, "side"] == "SHORT" and legs.loc[95, "entry_delta"] == -0.20
    # Equity reconciles with the trade log.
    assert result.equity["equity"].iloc[-1] == pytest.approx(100_000 + trade["net_pnl"])


def test_debit_spread_profit_and_stop():
    entry = chain_frame(D1, 100, calls((3.00, 3.20), (1.00, 1.10)))  # debit 3.10 - 1.05 = 2.05
    winner = chain_frame(D2, 103, calls((4.90, 5.10), (1.90, 2.00)))  # value 3.05 -> +1.00
    once = {"entry": {"min_days_between_entries": 30}}
    result = run(
        {**CALL_DEBIT, **once, "exit": {"profit_target_pct": 0.4, "stop_loss_multiple": 0.5}},
        {D1: entry, D2: winner},
    )
    trade = result.trades.iloc[0]
    assert trade["entry_value"] == pytest.approx(2.05)
    assert trade["exit_reason"] == "PROFIT_TARGET"
    assert trade["gross_pnl"] == pytest.approx(100.0)
    loser = chain_frame(D2, 97, calls((1.40, 1.60), (0.40, 0.50)))  # value 1.05 -> -1.00
    result = run(
        {**CALL_DEBIT, **once, "exit": {"profit_target_pct": 0.4, "stop_loss_multiple": 0.4}},
        {D1: entry, D2: loser},
    )
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["gross_pnl"] == pytest.approx(-100.0)


def test_slippage_is_reported_and_reduces_pnl():
    chains = {
        D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50))),
        D2: chain_frame(D2, 101, puts((0.45, 0.55), (0.15, 0.25))),
    }
    mid = run(PUT_CREDIT, chains).trades.iloc[0]
    natural = run(PUT_CREDIT, chains, fill_model="NATURAL").trades.iloc[0]
    assert mid["slippage_paid"] == pytest.approx(0.0)
    assert natural["entry_value"] == pytest.approx(-0.50)  # 1.00 bid - 0.50 ask
    assert natural["exit_value"] == pytest.approx(-0.40)  # buy 0.55, sell 0.15
    assert natural["slippage_paid"] == pytest.approx(20.0)
    assert natural["gross_pnl"] == pytest.approx(mid["gross_pnl"] - 20.0)


def test_stop_triggers_on_mid_mark_but_fills_at_executable_price():
    # Entry at natural prices: credit 0.50, so a 2x stop needs a mark-to-mid loss of 1.00.
    # Day 2: mid loss 0.90 although closing at the ask would lose 1.50 -> must not stop.
    wide = chain_frame(D2, 96, puts((1.20, 2.20), (0.20, 0.40)))
    hit = chain_frame(D3, 95, puts((1.80, 2.00), (0.20, 0.40)))  # mid loss 1.10
    chains = {D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50))), D2: wide, D3: hit}
    strategy = {**PUT_CREDIT, "entry": {"min_days_between_entries": 30}}
    result = run(strategy, chains, fill_model="NATURAL")
    trade = result.trades.iloc[0]
    assert trade["exit_time"] == pd.Timestamp("2024-01-10 21:00", tz="UTC")
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_trigger_pnl"] == pytest.approx(-1.10)
    assert trade["exit_mid_value"] == pytest.approx(-1.60)
    assert trade["exit_value"] == pytest.approx(-(2.00 - 0.20))  # filled at ask / bid


def test_stop_and_dte_exit_on_same_bar_records_stop_loss():
    late = date(2024, 1, 26)  # 21 DTE
    chains = {
        D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50))),
        late: chain_frame(late, 94, puts((2.40, 2.60), (0.50, 0.60))),
    }
    trade = run(PUT_CREDIT, chains).trades.iloc[0]
    assert trade["exit_reason"] == "STOP_LOSS"


def test_expiration_settles_at_intrinsic_with_no_exit_commission():
    hold = {
        **PUT_CREDIT,
        "exit": {"profit_target_pct": None, "stop_loss_multiple": None, "exit_dte": None},
    }
    chains = {D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50)))}
    closes = {D1: 100.0, EXPIRY - timedelta(days=1): 96.0, EXPIRY: 93.0}
    result = run(hold, chains, closes)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "EXPIRATION"
    assert trade["exit_value"] == pytest.approx(-2.0)  # short 95 put worth 2, long 90 put 0
    assert trade["gross_pnl"] == pytest.approx(-140.0)
    assert trade["commission"] + trade["fees"] == pytest.approx(2 * FEE)
    assert result.legs["settled_at_expiration"].all()
    physical = run(hold, chains, closes, settlement="physical", assignment_fee=5.0).trades.iloc[0]
    assert physical["assigned_contracts"] == 1
    assert physical["net_pnl"] == pytest.approx(trade["net_pnl"] - 5.0)


def test_settlement_never_uses_a_close_after_expiration():
    hold = {
        **PUT_CREDIT,
        "exit": {"profit_target_pct": None, "stop_loss_multiple": None, "exit_dte": None},
    }
    chains = {D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50)))}
    # No session on expiration day: settle on the prior close (96), not the later 80.
    closes = {D1: 100.0, EXPIRY - timedelta(days=1): 96.0, EXPIRY + timedelta(days=3): 80.0}
    trade = run(hold, chains, closes).trades.iloc[0]
    assert trade["exit_value"] == pytest.approx(0.0)


def test_missing_quote_skips_the_mark_instead_of_aborting():
    gap = chain_frame(D2, 101, [(95, "PUT", 0.45, 0.55, -0.2)])  # long leg not quoted
    chains = {
        D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50))),
        D2: gap,
        D3: chain_frame(D3, 101, puts((0.45, 0.55), (0.15, 0.25))),
    }
    strategy = {**PUT_CREDIT, "entry": {"min_days_between_entries": 30}}
    result = run(strategy, chains)
    trade = result.trades.iloc[0]
    assert trade["skipped_marks"] == 1
    assert trade["exit_time"] == pd.Timestamp("2024-01-10 21:00", tz="UTC")
    assert result.diagnostics["skipped_marks"] == 1
    # A day with no chain at all is also survivable.
    result = run(strategy, {**chains, D2: None}, closes={D1: 100.0, D2: 100.0, D3: 100.0})
    assert result.diagnostics["missing_chain_days"] == 1
    assert result.trades.iloc[0]["exit_reason"] == "PROFIT_TARGET"


def test_end_of_data_closes_open_positions_and_flags_estimates():
    chains = {D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50)))}
    trade = run(PUT_CREDIT, chains, closes={D1: 100.0, D2: 100.0}).trades.iloc[0]
    assert trade["exit_reason"] == "END_OF_DATA" and trade["exit_estimated"]
    assert trade["gross_pnl"] == pytest.approx(0.0)  # closed at the last mark


def test_credit_filter_applies_to_executable_fill():
    chains = {D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50)))}
    strategy = {**PUT_CREDIT, "entry": {"min_credit": 0.55}}
    closes = {D1: 100.0, D2: 100.0}
    assert len(run(strategy, chains, closes).trades) == 1  # mid credit 0.60 passes
    rejected = run(strategy, chains, closes, fill_model="NATURAL")  # executable credit 0.50
    assert rejected.trades.empty and rejected.rejections == {"CREDIT_TOO_LOW": 1}


def test_position_and_margin_limits():
    same = puts((1.00, 1.10), (0.40, 0.50))
    d4 = date(2024, 1, 11)  # the last session's row shows the end-of-data close
    chains = {d: chain_frame(d, 100, same) for d in (D1, D2, D3, d4)}
    off = {"profit_target_pct": None, "stop_loss_multiple": None, "exit_dte": None}
    hold = {**PUT_CREDIT, "exit": off}
    one = run({**hold, "entry": {"max_open_positions": 1}}, chains)
    assert one.equity["open_positions"].tolist() == [1, 1, 1, 0]
    three = run({**hold, "entry": {"max_open_positions": 3}}, chains)
    assert three.equity["open_positions"].tolist() == [1, 2, 3, 0]
    spaced = run(
        {**hold, "entry": {"max_open_positions": 3, "min_days_between_entries": 2}}, chains
    )
    assert spaced.equity["open_positions"].tolist() == [1, 1, 2, 0]
    # 440 margin per spread against 1,000 of capital at 50% utilisation: only one fits.
    tight = run({**hold, "entry": {"max_open_positions": 3}}, chains, initial_capital=1000.0)
    assert tight.equity["open_positions"].max() == 1
    assert tight.rejections["INSUFFICIENT_BUYING_POWER"] == 2


def test_portfolio_greeks_are_aggregated():
    chains = {D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50)))}
    row = run({**PUT_CREDIT, "quantity": 2}, chains, closes={D1: 100.0, D2: 100.0}).equity.iloc[0]
    assert row["delta"] == pytest.approx((0.20 - 0.08) * 100 * 2)  # short put is long delta
    assert row["theta"] == pytest.approx(0.0)  # equal thetas, one long one short
    assert row["margin_used"] == pytest.approx(880.0)


def test_engine_only_sees_regimes_known_at_the_step(put_spread_days):
    stamps = pd.to_datetime(["2024-01-08 21:00", "2024-01-09 21:00"], utc=True)
    frame = pd.DataFrame(
        {
            "trend": ["BEARISH", "SIDEWAYS"],
            "volatility": ["LOW_IV", "HIGH_IV"],
            "vol_dynamics": "STABLE",
            "term_structure": "UNKNOWN",
            "iv_rank": [10.0, 80.0],
            "iv_percentile": np.nan,
            "vol_level": np.nan,
            "adx": np.nan,
        },
        index=stamps,
    )
    chains = {
        D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50))),
        D2: chain_frame(D2, 100, puts((1.00, 1.10), (0.40, 0.50))),
    }
    provider = ScriptedProvider({D1: 100.0, D2: 100.0, D3: 100.0}, chains)
    strategy = build_strategy({**PUT_CREDIT, "entry": {"allowed_volatility": ["HIGH_IV"]}})
    config = BacktestConfig(
        symbol="TEST", start=D1, end=D3, execution=ExecutionConfig(fill_model="MID")
    )
    result = Backtester(provider, strategy, config, RegimeTimeline(frame)).run()
    # Day 1 is LOW_IV; the HIGH_IV tag only exists from day 2's close.
    assert result.rejections == {"REGIME_FILTER": 1}
    assert result.trades.iloc[0]["entry_time"] == stamps[1]
    assert result.trades.iloc[0]["entry_volatility"] == "HIGH_IV"


def test_no_chains_at_all_is_an_error():
    with pytest.raises(DataUnavailableError, match="no option chain"):
        run(PUT_CREDIT, {D1: None}, closes={D1: 100.0})


def test_no_entry_on_the_final_session():
    chains = {D1: chain_frame(D1, 100, puts((1.00, 1.10), (0.40, 0.50)))}
    assert run(PUT_CREDIT, chains).trades.empty
