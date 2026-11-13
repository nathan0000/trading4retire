"""Vertical spread: credit (sell the anchor) or debit (buy the anchor), put or call side."""

from __future__ import annotations

from options_backtester.data.base import OptionChainSnapshot
from options_backtester.strategies.base import Leg, LegQuote, OptionStrategy, Rejection


class VerticalSpread(OptionStrategy):
    def select_legs(self, chain: OptionChainSnapshot) -> list[tuple[Leg, LegQuote]] | Rejection:
        expiration = self._front_expiration(chain)
        if expiration is None:
            return Rejection("NO_EXPIRATION_IN_DTE_WINDOW")
        anchor_side = -1 if self.config.direction == "credit" else 1
        return self._vertical(chain, expiration, self.config.option_side.upper(), anchor_side)
