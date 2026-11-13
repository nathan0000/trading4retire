"""Long calendar: short the front expiration, long a later one at the same strike."""

from __future__ import annotations

from options_backtester.data.base import OptionChainSnapshot
from options_backtester.strategies.base import Leg, LegQuote, OptionStrategy, Rejection, make_leg


class Calendar(OptionStrategy):
    def select_legs(self, chain: OptionChainSnapshot) -> list[tuple[Leg, LegQuote]] | Rejection:
        cfg = self.config
        front = self._front_expiration(chain)
        assert cfg.back_min_dte is not None and cfg.back_max_dte is not None
        back = self.builder.pick_expiration(chain.quotes, cfg.back_min_dte, cfg.back_max_dte)
        if front is None or back is None:
            return Rejection("NO_EXPIRATION_IN_DTE_WINDOW")
        option_type = cfg.option_side.upper()
        front_side = self.builder.side_frame(chain.quotes, front, option_type)
        anchor = self.builder.pick_strike(front_side, cfg.strike, chain.underlying_price)
        if anchor is None:
            return Rejection("NO_VALID_STRIKE", f"{option_type} anchor")
        far = chain.lookup(back, float(anchor["strike"]), option_type)
        if far is None:
            return Rejection("NO_VALID_STRIKE", "strike not listed in back expiration")
        return [make_leg(anchor, -1), make_leg(far, 1)]
