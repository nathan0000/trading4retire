import pandas as pd
import pytest

from spx_0dte_backtest.backtest import BacktestEngine, Portfolio, calculate_metrics
from spx_0dte_backtest.execution import ExecutionEngine, ExecutionMode, ExecutionSettings
from spx_0dte_backtest.strategies import IronCondorStrategy
from tests.unit.backtest.test_engine import quotes_at
from tests.unit.strategies.test_iron_condor import context


def test_portfolio_collects_completed_records_without_hiding_trade_detail() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    forced = pd.Timestamp("2026-08-12 15:55", tz="America/New_York")
    engine = BacktestEngine(
        execution_engine=ExecutionEngine(ExecutionSettings(mode=ExecutionMode.BID_ASK))
    )
    record = engine.run_candidate(candidate, [quotes_at(candidate, forced, 0.80)])
    portfolio = Portfolio()
    portfolio.add(record)
    assert portfolio.records == (record,)
    assert portfolio.net_pnl == pytest.approx(record.net_pnl)
    metrics = calculate_metrics(portfolio.records)
    assert metrics.total_trades == 1
    assert metrics.net_pnl == pytest.approx(record.net_pnl)
