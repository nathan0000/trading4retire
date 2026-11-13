"""Short strangle: short put and short call at the anchor rule, no wings (undefined risk)."""

from __future__ import annotations

from options_backtester.data.base import OptionChainSnapshot
from options_backtester.strategies.base import Leg, LegQuote, OptionStrategy, Rejection, make_leg


class Strangle(OptionStrategy):
    def select_legs(self, chain: OptionChainSnapshot) -> list[tuple[Leg, LegQuote]] | Rejection:
        expiration = self._front_expiration(chain)
        if expiration is None:
            return Rejection("NO_EXPIRATION_IN_DTE_WINDOW")
        legs = []
        for option_type in ("PUT", "CALL"):
            side = self.builder.side_frame(chain.quotes, expiration, option_type)
            anchor = self.builder.pick_strike(side, self.config.strike, chain.underlying_price)
            if anchor is None:
                return Rejection("NO_VALID_STRIKE", f"{option_type} anchor")
            legs.append(make_leg(anchor, -1))
        if legs[0][0].strike >= legs[1][0].strike:
            return Rejection("NO_VALID_STRIKE", "short strikes overlap")
        return legs
