from __future__ import annotations

import pytest

from spx_0dte_backtest.backtest import ExitReason, Position, PositionState
from spx_0dte_backtest.execution import ExecutionEngine, ExecutionMode, ExecutionSettings
from spx_0dte_backtest.risk import RiskSettings
from spx_0dte_backtest.strategies import IronCondorStrategy
from tests.unit.strategies.test_iron_condor import context


def make_position(*, stop_multiplier: float = 0.90, profit_target_fraction=None) -> Position:
    candidate = IronCondorStrategy().generate_trade(context())
    execution = ExecutionEngine(ExecutionSettings(mode=ExecutionMode.BID_ASK))
    entry = execution.execute_entry(candidate)
    return Position.enter(
        candidate,
        entry,
        RiskSettings(
            stop_multiplier=stop_multiplier,
            profit_target_fraction=profit_target_fraction,
        ),
        contract_multiplier=execution.settings.contract_multiplier,
    )


def test_entered_position_has_distinct_credit_stop_and_mark_fields() -> None:
    position = make_position()
    assert position.state is PositionState.MONITORING
    assert position.initial_credit == pytest.approx(1.70)
    assert position.current_mark_to_close is None
    assert position.stop_loss_amount == pytest.approx(1.53)
    assert position.stop_loss_price == pytest.approx(3.23)
    assert position.profit_loss_points is None


def test_mark_updates_pnl_without_closing() -> None:
    position = make_position()
    position.mark(2.00)
    assert position.state is PositionState.MONITORING
    assert position.current_mark_to_close == pytest.approx(2.00)
    assert position.profit_loss_points == pytest.approx(-0.30)


def test_stop_is_triggered_at_cost_to_close_threshold() -> None:
    position = make_position()
    position.mark(3.22)
    assert position.state is PositionState.MONITORING
    position.mark(3.23)
    assert position.state is PositionState.STOPPED
    assert position.exit_reason is ExitReason.STOP_LOSS


def test_optional_profit_target_state() -> None:
    position = make_position(profit_target_fraction=0.50)
    position.mark(0.85)
    assert position.state is PositionState.PROFIT_TARGET
    assert position.exit_reason is ExitReason.PROFIT_TARGET


def test_terminal_position_cannot_be_remarked() -> None:
    position = make_position()
    position.mark(4.0)
    with pytest.raises(ValueError, match="terminal"):
        position.mark(1.0)
