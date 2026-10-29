"""Phase 7 single-candidate backtest coordinator."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time
from enum import StrEnum

import pandas as pd

from spx_0dte_backtest.backtest.position import Position, PositionState, TradeRecord
from spx_0dte_backtest.data import OptionQuote
from spx_0dte_backtest.execution import ExecutionEngine, MissingExitQuote
from spx_0dte_backtest.risk import RiskSettings
from spx_0dte_backtest.strategies import TradeCandidate


class MissingForcedExitQuotes(RuntimeError):
    """Raised when an executable quote set is unavailable at forced exit."""


class StopTrigger(StrEnum):
    """Price that stop and profit-target decisions are evaluated on.

    Fills always use the execution engine's executable price; only the
    trigger differs. ``MID`` mirrors a broker mark-based stop, so a momentary
    bid/ask widening does not stop a position out. ``EXECUTABLE`` triggers on
    the worst-case fill price.
    """

    MID = "MID"
    EXECUTABLE = "EXECUTABLE"


@dataclass(frozen=True, slots=True)
class BacktestSettings:
    forced_exit_time: time | str = time(15, 55)
    timezone: str = "America/New_York"
    stop_trigger: StopTrigger | str = StopTrigger.MID

    def __post_init__(self) -> None:
        value = self.forced_exit_time
        if isinstance(value, str):
            try:
                parsed = time.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("forced exit time must be HH:MM") from exc
            object.__setattr__(self, "forced_exit_time", parsed)
        elif not isinstance(value, time):
            raise ValueError("forced exit time must be a time or HH:MM string")
        object.__setattr__(self, "stop_trigger", StopTrigger(self.stop_trigger))


class BacktestEngine:
    """Enter, monitor, and close one candidate using exact historical snapshots."""

    def __init__(
        self,
        execution_engine: ExecutionEngine | None = None,
        risk_settings: RiskSettings | None = None,
        settings: BacktestSettings | None = None,
    ) -> None:
        self.execution_engine = execution_engine or ExecutionEngine()
        self.risk_settings = risk_settings or RiskSettings()
        self.settings = settings or BacktestSettings()

    def run_candidate(
        self,
        candidate: TradeCandidate,
        quote_snapshots: Iterable[Iterable[OptionQuote]],
    ) -> TradeRecord:
        entry = self.execution_engine.execute_entry(candidate)
        position = Position.enter(
            candidate,
            entry,
            self.risk_settings,
            contract_multiplier=self.execution_engine.settings.contract_multiplier,
        )
        snapshots = self._validate_snapshots(candidate, quote_snapshots)
        forced_snapshot: tuple[OptionQuote, ...] | None = None
        forced_time = self.settings.forced_exit_time
        assert isinstance(forced_time, time)
        skipped_marks = 0

        for snapshot in snapshots:
            timestamp = pd.Timestamp(snapshot[0].timestamp)
            local_time = timestamp.tz_convert(self.settings.timezone).time().replace(tzinfo=None)
            forced = local_time == forced_time
            if forced:
                forced_snapshot = snapshot
            if local_time > forced_time:
                continue
            try:
                exit_execution = self.execution_engine.execute_exit(candidate.legs, snapshot)
            except MissingExitQuote as exc:
                if forced:
                    raise MissingForcedExitQuotes(
                        f"incomplete snapshot at {forced_time.isoformat(timespec='minutes')}: {exc}"
                    ) from exc
                # A leg without a quote cannot be marked; keep monitoring.
                skipped_marks += 1
                continue
            debit = exit_execution.exit_debit_points
            assert debit is not None
            if self.settings.stop_trigger is StopTrigger.MID:
                trigger = self.execution_engine.mark_to_close(candidate.legs, snapshot)
            else:
                trigger = debit
            if forced:
                position.force_exit(trigger)
            else:
                position.mark(trigger)
            if position.state is not PositionState.MONITORING:
                return position.to_record(exit_execution, skipped_marks=skipped_marks)

        if forced_snapshot is None:
            raise MissingForcedExitQuotes(
                f"no executable snapshot at {forced_time.isoformat(timespec='minutes')}"
            )
        # The loop necessarily terminates at the exact forced snapshot.
        raise RuntimeError("forced-exit snapshot did not terminate the position")

    def _validate_snapshots(
        self,
        candidate: TradeCandidate,
        quote_snapshots: Iterable[Iterable[OptionQuote]],
    ) -> tuple[tuple[OptionQuote, ...], ...]:
        result: list[tuple[OptionQuote, ...]] = []
        previous: pd.Timestamp | None = None
        entry_local_date = candidate.timestamp.tz_convert(self.settings.timezone).date()
        for values in quote_snapshots:
            snapshot = tuple(values)
            if not snapshot:
                raise ValueError("quote snapshots cannot be empty")
            timestamps = {pd.Timestamp(quote.timestamp) for quote in snapshot}
            if len(timestamps) != 1:
                raise ValueError("each quote snapshot must have one timestamp")
            timestamp = next(iter(timestamps))
            if timestamp.tzinfo is None:
                raise ValueError("snapshot timestamps must be timezone-aware")
            if timestamp <= candidate.timestamp:
                raise ValueError("monitoring snapshots must be later than entry")
            if timestamp.tz_convert(self.settings.timezone).date() != entry_local_date:
                raise ValueError("monitoring quotes must remain in the entry session")
            if previous is not None and timestamp <= previous:
                raise ValueError("quote snapshots must be strictly chronological")
            previous = timestamp
            result.append(snapshot)
        return tuple(result)
