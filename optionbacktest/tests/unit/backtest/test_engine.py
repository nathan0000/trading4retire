from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from spx_0dte_backtest.backtest import (
    BacktestEngine,
    BacktestSettings,
    ExitReason,
    MissingForcedExitQuotes,
    PositionState,
    StopTrigger,
)
from spx_0dte_backtest.execution import ExecutionEngine, ExecutionMode, ExecutionSettings
from spx_0dte_backtest.risk import RiskSettings
from spx_0dte_backtest.strategies import IronCondorStrategy
from tests.unit.strategies.test_iron_condor import context


def quotes_at(candidate, timestamp: pd.Timestamp, debit: float):
    # The two short-leg asks split the debit; long-leg bids are zero.
    per_leg = debit / 2
    quotes = []
    for leg in candidate.legs:
        if leg.quantity < 0:  # buy short leg back at ask
            bid, ask = max(0.0, per_leg - 0.10), per_leg
        else:  # sell long leg at bid
            bid, ask = 0.0, 0.10
        quotes.append(
            replace(leg.quote, timestamp=timestamp, bid=bid, ask=ask, last=(bid + ask) / 2)
        )
    return tuple(quotes)


def engine(
    *,
    stop_multiplier=0.90,
    profit_target_fraction=None,
    stop_trigger=StopTrigger.MID,
) -> BacktestEngine:
    return BacktestEngine(
        execution_engine=ExecutionEngine(ExecutionSettings(mode=ExecutionMode.BID_ASK)),
        risk_settings=RiskSettings(
            stop_multiplier=stop_multiplier,
            profit_target_fraction=profit_target_fraction,
        ),
        settings=BacktestSettings(stop_trigger=stop_trigger),
    )


def test_stops_at_first_snapshot_meeting_threshold() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    snapshots = [
        quotes_at(candidate, candidate.timestamp + pd.Timedelta("5min"), 3.20),
        quotes_at(candidate, candidate.timestamp + pd.Timedelta("10min"), 3.30),
        quotes_at(candidate, candidate.timestamp + pd.Timedelta("15min"), 4.00),
    ]
    record = engine(stop_trigger=StopTrigger.EXECUTABLE).run_candidate(candidate, snapshots)
    assert record.state is PositionState.STOPPED
    assert record.exit_reason is ExitReason.STOP_LOSS
    assert record.exit_timestamp == candidate.timestamp + pd.Timedelta("10min")
    assert record.exit_debit == pytest.approx(3.30)
    assert record.gross_pnl == pytest.approx((1.70 - 3.30) * 100)
    assert record.net_pnl == pytest.approx(record.gross_pnl - record.commission - record.fees)


def test_mid_trigger_ignores_spread_but_fills_at_executable_price() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    snapshots = [
        # Executable 3.30 is past the 3.23 stop, but the mid (3.10) is not.
        quotes_at(candidate, candidate.timestamp + pd.Timedelta("10min"), 3.30),
        quotes_at(candidate, candidate.timestamp + pd.Timedelta("15min"), 4.00),
    ]
    record = engine().run_candidate(candidate, snapshots)
    assert record.exit_reason is ExitReason.STOP_LOSS
    assert record.exit_timestamp == candidate.timestamp + pd.Timedelta("15min")
    assert record.exit_trigger_mark == pytest.approx(3.80)
    assert record.exit_debit == pytest.approx(4.00)
    assert record.gross_pnl == pytest.approx((1.70 - 4.00) * 100)


def test_incomplete_monitoring_snapshot_is_skipped_not_fatal() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    gap = quotes_at(candidate, candidate.timestamp + pd.Timedelta("5min"), 9.00)[:-1]
    forced = pd.Timestamp("2026-08-12 15:55", tz="America/New_York")
    record = engine().run_candidate(candidate, [gap, quotes_at(candidate, forced, 0.80)])
    assert record.exit_reason is ExitReason.FORCED_EXIT
    assert record.skipped_marks == 1


def test_incomplete_forced_exit_snapshot_is_fatal() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    forced = pd.Timestamp("2026-08-12 15:55", tz="America/New_York")
    with pytest.raises(MissingForcedExitQuotes, match="incomplete snapshot"):
        engine().run_candidate(candidate, [quotes_at(candidate, forced, 0.80)[:-1]])


def test_forced_exit_requires_exact_1555_new_york_snapshot() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    forced = pd.Timestamp("2026-08-12 15:55", tz="America/New_York")
    record = engine().run_candidate(candidate, [quotes_at(candidate, forced, 0.80)])
    assert record.state is PositionState.FORCED_EXIT
    assert record.exit_reason is ExitReason.FORCED_EXIT
    assert record.exit_timestamp == forced


def test_does_not_assume_last_snapshot_is_forced_exit() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    last = pd.Timestamp("2026-08-12 15:50", tz="America/New_York")
    with pytest.raises(MissingForcedExitQuotes):
        engine().run_candidate(candidate, [quotes_at(candidate, last, 0.80)])


def test_later_day_quotes_are_rejected_not_held_overnight() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    tomorrow = pd.Timestamp("2026-08-13 09:30", tz="America/New_York")
    with pytest.raises(ValueError, match="entry session"):
        engine().run_candidate(candidate, [quotes_at(candidate, tomorrow, 0.80)])


def test_snapshots_must_be_strictly_chronological() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    first = quotes_at(candidate, candidate.timestamp + pd.Timedelta("10min"), 1.0)
    second = quotes_at(candidate, candidate.timestamp + pd.Timedelta("5min"), 1.0)
    with pytest.raises(ValueError, match="chronological"):
        engine().run_candidate(candidate, [first, second])


def test_settings_validate_forced_exit_time() -> None:
    with pytest.raises(ValueError):
        BacktestSettings(forced_exit_time="bad")


def test_settings_parse_and_validate_stop_trigger() -> None:
    assert BacktestSettings(stop_trigger="EXECUTABLE").stop_trigger is StopTrigger.EXECUTABLE
    with pytest.raises(ValueError):
        BacktestSettings(stop_trigger="LAST")
