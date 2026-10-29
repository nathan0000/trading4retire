from __future__ import annotations

import pytest

from spx_0dte_backtest.backtest import BacktestEngine, MissingForcedExitQuotes
from spx_0dte_backtest.strategies import IronCondorStrategy
from tests.unit.strategies.test_iron_condor import context


def test_empty_monitoring_data_reports_missing_forced_exit() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    with pytest.raises(MissingForcedExitQuotes, match="15:55"):
        BacktestEngine().run_candidate(candidate, [])
