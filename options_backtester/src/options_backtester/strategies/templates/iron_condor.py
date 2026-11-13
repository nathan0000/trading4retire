"""Short iron condor: short put + short call at the anchor rule, long wings beyond."""

from __future__ import annotations

from options_backtester.data.base import OptionChainSnapshot
from options_backtester.strategies.base import Leg, LegQuote, OptionStrategy, Rejection


class IronCondor(OptionStrategy):
    def select_legs(self, chain: OptionChainSnapshot) -> list[tuple[Leg, LegQuote]] | Rejection:
        expiration = self._front_expiration(chain)
        if expiration is None:
            return Rejection("NO_EXPIRATION_IN_DTE_WINDOW")
        legs: list[tuple[Leg, LegQuote]] = []
        for option_type in ("PUT", "CALL"):
            spread = self._vertical(chain, expiration, option_type, anchor_side=-1)
            if isinstance(spread, Rejection):
                return spread
            legs.extend(spread)
        short_put, short_call = legs[0][0], legs[2][0]
        if short_put.strike >= short_call.strike:
            return Rejection("NO_VALID_STRIKE", "short strikes overlap")
        return legs
