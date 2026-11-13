"""Explicit position lifecycle and trade-record construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

import pandas as pd

from spx_0dte_backtest.execution import ExecutionResult
from spx_0dte_backtest.risk import RiskSettings
from spx_0dte_backtest.strategies import TradeCandidate


class PositionState(StrEnum):
    CREATED = "CREATED"
    ENTERED = "ENTERED"
    MONITORING = "MONITORING"
    STOPPED = "STOPPED"
    PROFIT_TARGET = "PROFIT_TARGET"
    FORCED_EXIT = "FORCED_EXIT"
    EXPIRED = "EXPIRED"


class ExitReason(StrEnum):
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    FORCED_EXIT = "FORCED_EXIT"
    EXPIRED = "EXPIRED"


TERMINAL_STATES = {
    PositionState.STOPPED,
    PositionState.PROFIT_TARGET,
    PositionState.FORCED_EXIT,
    PositionState.EXPIRED,
}


@dataclass(frozen=True, slots=True)
class TradeRecord:
    candidate: TradeCandidate
    state: PositionState
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    initial_credit: float
    exit_debit: float
    gross_pnl: float
    commission: float
    fees: float
    net_pnl: float
    exit_reason: ExitReason
    entry_execution: ExecutionResult
    exit_execution: ExecutionResult
    state_history: tuple[PositionState, ...]
    exit_trigger_mark: float | None = None
    skipped_marks: int = 0


@dataclass(slots=True)
class Position:
    candidate: TradeCandidate
    entry_execution: ExecutionResult
    risk_settings: RiskSettings
    contract_multiplier: float
    state: PositionState = PositionState.CREATED
    current_mark_to_close: float | None = None
    profit_loss_points: float | None = None
    exit_reason: ExitReason | None = None
    _state_history: list[PositionState] = field(default_factory=list, repr=False)

    @classmethod
    def enter(
        cls,
        candidate: TradeCandidate,
        entry_execution: ExecutionResult,
        risk_settings: RiskSettings,
        *,
        contract_multiplier: float,
    ) -> Position:
        if not candidate.is_accepted:
            raise ValueError("position requires an accepted candidate")
        initial_credit = entry_execution.entry_credit_points
        if initial_credit is None or initial_credit <= 0:
            raise ValueError("position requires a positive entry credit")
        position = cls(
            candidate=candidate,
            entry_execution=entry_execution,
            risk_settings=risk_settings,
            contract_multiplier=contract_multiplier,
            _state_history=[PositionState.CREATED],
        )
        position._transition(PositionState.ENTERED)
        position._transition(PositionState.MONITORING)
        return position

    @property
    def initial_credit(self) -> float:
        value = self.entry_execution.entry_credit_points
        assert value is not None
        return value

    @property
    def stop_loss_amount(self) -> float:
        return self.risk_settings.stop_loss_amount(self.initial_credit)

    @property
    def stop_loss_price(self) -> float:
        return self.risk_settings.stop_loss_price(self.initial_credit)

    @property
    def profit_target_price(self) -> float | None:
        return self.risk_settings.profit_target_price(self.initial_credit)

    @property
    def state_history(self) -> tuple[PositionState, ...]:
        return tuple(self._state_history)

    def mark(self, cost_to_close: float) -> None:
        if self.state in TERMINAL_STATES:
            raise ValueError("cannot mark a terminal position")
        if not isfinite(cost_to_close) or cost_to_close < 0:
            raise ValueError("cost to close must be finite and non-negative")
        self.current_mark_to_close = cost_to_close
        self.profit_loss_points = self.initial_credit - cost_to_close
        if cost_to_close >= self.stop_loss_price:
            self.exit_reason = ExitReason.STOP_LOSS
            self._transition(PositionState.STOPPED)
        elif self.profit_target_price is not None and cost_to_close <= self.profit_target_price:
            self.exit_reason = ExitReason.PROFIT_TARGET
            self._transition(PositionState.PROFIT_TARGET)

    def force_exit(self, cost_to_close: float) -> None:
        self.mark(cost_to_close)
        if self.state is PositionState.MONITORING:
            self.exit_reason = ExitReason.FORCED_EXIT
            self._transition(PositionState.FORCED_EXIT)

    def to_record(self, exit_execution: ExecutionResult, *, skipped_marks: int = 0) -> TradeRecord:
        if self.state not in TERMINAL_STATES or self.exit_reason is None:
            raise ValueError("only a terminal position can produce a trade record")
        debit = exit_execution.exit_debit_points
        if debit is None:
            raise ValueError("exit execution is required")
        exit_timestamp = pd.Timestamp(exit_execution.fills[0].quote.timestamp)
        gross_pnl = (self.initial_credit - debit) * self.contract_multiplier
        commission = self.entry_execution.commission + exit_execution.commission
        fees = self.entry_execution.fees + exit_execution.fees
        return TradeRecord(
            candidate=self.candidate,
            state=self.state,
            entry_timestamp=self.candidate.timestamp,
            exit_timestamp=exit_timestamp,
            initial_credit=self.initial_credit,
            exit_debit=debit,
            gross_pnl=gross_pnl,
            commission=commission,
            fees=fees,
            net_pnl=gross_pnl - commission - fees,
            exit_reason=self.exit_reason,
            entry_execution=self.entry_execution,
            exit_execution=exit_execution,
            state_history=self.state_history,
            exit_trigger_mark=self.current_mark_to_close,
            skipped_marks=skipped_marks,
        )

    def _transition(self, state: PositionState) -> None:
        self.state = state
        self._state_history.append(state)
