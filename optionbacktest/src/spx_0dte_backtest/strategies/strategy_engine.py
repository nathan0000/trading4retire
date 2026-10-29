"""Strategy dispatcher with a stable interface for future strategy types."""

from __future__ import annotations

from collections.abc import Mapping

from spx_0dte_backtest.strategies.base import (
    MarketContext,
    Strategy,
    StrategyType,
    TradeCandidate,
)
from spx_0dte_backtest.strategies.iron_condor import IronCondorStrategy


class StrategyEngine:
    """Dispatch candidate generation without embedding strategy-specific rules."""

    def __init__(self, strategies: Mapping[StrategyType, Strategy] | None = None):
        self._strategies: dict[StrategyType, Strategy] = dict(
            strategies or {StrategyType.IRON_CONDOR: IronCondorStrategy()}
        )

    @property
    def supported_strategies(self) -> tuple[StrategyType, ...]:
        return tuple(self._strategies)

    def generate_trade(self, strategy: StrategyType, context: MarketContext) -> TradeCandidate:
        try:
            implementation = self._strategies[strategy]
        except KeyError as exc:
            raise ValueError(f"unsupported strategy: {strategy}") from exc
        return implementation.generate_trade(context)
