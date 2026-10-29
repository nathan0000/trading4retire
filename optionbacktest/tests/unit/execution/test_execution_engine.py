from __future__ import annotations

from dataclasses import replace

import pytest

from spx_0dte_backtest.data import OptionQuote
from spx_0dte_backtest.execution import (
    CostModel,
    ExecutionEngine,
    ExecutionMode,
    ExecutionPurpose,
    ExecutionSettings,
    OrderSide,
)
from spx_0dte_backtest.strategies import IronCondorStrategy, TradeCandidate
from tests.unit.strategies.test_iron_condor import context


@pytest.fixture
def candidate() -> TradeCandidate:
    result = IronCondorStrategy().generate_trade(context())
    assert result.is_accepted
    return result


def test_mid_entry_prices_every_leg_at_its_midpoint(candidate: TradeCandidate) -> None:
    result = ExecutionEngine(ExecutionSettings(mode=ExecutionMode.MID)).execute_entry(candidate)
    assert result.purpose is ExecutionPurpose.ENTRY
    assert result.gross_cashflow_points == pytest.approx(2.00)
    assert result.entry_credit_points == pytest.approx(2.00)
    assert [fill.fill_price for fill in result.fills] == pytest.approx([0.35, 1.30, 1.40, 0.35])


def test_bid_ask_entry_matches_conservative_candidate_credit(candidate: TradeCandidate) -> None:
    settings = ExecutionSettings(mode=ExecutionMode.BID_ASK)
    result = ExecutionEngine(settings).execute_entry(candidate)
    assert result.gross_cashflow_points == pytest.approx(1.70)
    assert result.gross_cashflow_points == candidate.conservative_executable_credit
    assert [fill.side for fill in result.fills] == [
        OrderSide.BUY,
        OrderSide.SELL,
        OrderSide.SELL,
        OrderSide.BUY,
    ]
    assert [fill.fill_price for fill in result.fills] == pytest.approx([0.40, 1.20, 1.30, 0.40])


def test_slippage_is_applied_against_each_individual_leg(candidate: TradeCandidate) -> None:
    settings = ExecutionSettings(
        mode=ExecutionMode.BID_ASK_PLUS_SLIPPAGE,
        slippage_points=0.05,
    )
    result = ExecutionEngine(settings).execute_entry(candidate)
    assert [fill.fill_price for fill in result.fills] == pytest.approx([0.45, 1.15, 1.25, 0.45])
    assert result.gross_cashflow_points == pytest.approx(1.50)


def test_default_execution_is_conservative(candidate: TradeCandidate) -> None:
    result = ExecutionEngine().execute_entry(candidate)
    assert result.mode is ExecutionMode.BID_ASK_PLUS_SLIPPAGE
    assert result.gross_cashflow_points < candidate.conservative_executable_credit


def test_cost_model_is_per_contract_per_leg(candidate: TradeCandidate) -> None:
    settings = ExecutionSettings(
        mode=ExecutionMode.BID_ASK,
        cost_model=CostModel(commission_per_contract=0.65, exchange_fee_per_contract=0.10),
    )
    result = ExecutionEngine(settings).execute_entry(candidate)
    assert result.gross_cashflow_dollars == pytest.approx(170.0)
    assert result.commission == pytest.approx(2.60)
    assert result.fees == pytest.approx(0.40)
    assert result.net_cashflow_dollars == pytest.approx(167.0)


def exit_quotes(candidate: TradeCandidate) -> tuple[OptionQuote, ...]:
    prices = {
        "LONG_PUT": (0.20, 0.30),
        "SHORT_PUT": (1.50, 1.70),
        "SHORT_CALL": (1.60, 1.80),
        "LONG_CALL": (0.20, 0.30),
    }
    timestamp = candidate.timestamp + __import__("pandas").Timedelta("30min")
    return tuple(
        replace(
            leg.quote,
            timestamp=timestamp,
            bid=prices[leg.role.value][0],
            ask=prices[leg.role.value][1],
            last=sum(prices[leg.role.value]) / 2,
        )
        for leg in candidate.legs
    )


def test_exit_reverses_each_leg_and_reports_debit(candidate: TradeCandidate) -> None:
    result = ExecutionEngine(ExecutionSettings(mode=ExecutionMode.BID_ASK)).execute_exit(
        candidate.legs, exit_quotes(candidate)
    )
    assert result.purpose is ExecutionPurpose.EXIT
    assert [fill.side for fill in result.fills] == [
        OrderSide.SELL,
        OrderSide.BUY,
        OrderSide.BUY,
        OrderSide.SELL,
    ]
    assert result.gross_cashflow_points == pytest.approx(-3.10)
    assert result.exit_debit_points == pytest.approx(3.10)


def test_exit_requires_one_exact_quote_for_every_contract(candidate: TradeCandidate) -> None:
    quotes = exit_quotes(candidate)
    with pytest.raises(ValueError, match="missing exit quote"):
        ExecutionEngine().execute_exit(candidate.legs, quotes[:-1])
    with pytest.raises(ValueError, match="duplicate exit quote"):
        ExecutionEngine().execute_exit(candidate.legs, quotes + (quotes[0],))


def test_rejected_candidate_cannot_be_executed() -> None:
    rejected = IronCondorStrategy().generate_trade(
        context(timestamp=__import__("pandas").Timestamp("2026-08-12 10:50", tz="America/New_York"))
    )
    with pytest.raises(ValueError, match="accepted"):
        ExecutionEngine().execute_entry(rejected)


@pytest.mark.parametrize(
    "settings",
    [
        lambda: ExecutionSettings(slippage_points=-0.01),
        lambda: ExecutionSettings(contract_multiplier=0),
        lambda: ExecutionSettings(cost_model=CostModel(commission_per_contract=-1)),
    ],
)
def test_execution_settings_reject_invalid_values(settings) -> None:
    with pytest.raises(ValueError):
        settings()
