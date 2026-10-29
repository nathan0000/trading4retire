from spx_0dte_backtest.strategies import StrategyEngine, StrategyType

from .test_iron_condor import context


def test_engine_dispatches_iron_condor_through_common_interface() -> None:
    engine = StrategyEngine()
    assert engine.supported_strategies == (StrategyType.IRON_CONDOR,)
    candidate = engine.generate_trade(StrategyType.IRON_CONDOR, context())
    assert candidate.is_accepted
    assert candidate.strategy is StrategyType.IRON_CONDOR
