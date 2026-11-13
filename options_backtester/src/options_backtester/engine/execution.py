"""Fill simulation, slippage and transaction costs.

Each leg is filled separately against its own bid/ask. A combo is atomic: if any
leg lacks a tradable quote the whole order is not filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from options_backtester.strategies.base import Leg, LegQuote


class FillModel(StrEnum):
    MID = "MID"
    #: Buy at the ask, sell at the bid.
    NATURAL = "NATURAL"
    #: Mid plus a fraction of the spread that grows as volume thins.
    SLIPPAGE = "SLIPPAGE"


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fill_model: FillModel = FillModel.SLIPPAGE
    #: Fraction of the full bid-ask spread paid away from mid (0 = mid, 0.5 = natural).
    spread_fraction: float = Field(default=0.25, ge=0, le=0.5)
    #: Volume at or above which a contract fills at ``spread_fraction``; thinner
    #: contracts slide linearly toward the natural price. None disables the adjustment.
    liquid_volume: float | None = Field(default=100.0, gt=0)
    commission_per_contract: float = Field(default=0.65, ge=0)
    fee_per_contract: float = Field(default=0.05, ge=0)
    multiplier: int = Field(default=100, ge=1)


@dataclass(frozen=True, slots=True)
class ComboFill:
    """Per-unit package prices in option points. ``value`` is signed: negative = credit."""

    value: float
    mid_value: float
    leg_prices: tuple[float, ...]
    leg_mids: tuple[float, ...]

    @property
    def slippage(self) -> float:
        """Points paid away from mid (always >= 0)."""
        return self.value - self.mid_value


class ExecutionModel:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def leg_price(self, side: int, quote: LegQuote) -> float:
        """Fill price for buying (``side`` = +1) or selling (-1) one contract."""
        cfg = self.config
        mid, spread = (quote.bid + quote.ask) / 2.0, quote.ask - quote.bid
        if cfg.fill_model is FillModel.MID:
            fraction = 0.0
        elif cfg.fill_model is FillModel.NATURAL:
            fraction = 0.5
        else:
            fraction = cfg.spread_fraction
            if cfg.liquid_volume is not None and np.isfinite(quote.volume):
                thin = 1.0 - min(max(quote.volume, 0.0) / cfg.liquid_volume, 1.0)
                fraction += (0.5 - fraction) * thin
        return mid + side * fraction * spread

    def fill(
        self, legs: tuple[Leg, ...], quotes: tuple[LegQuote, ...], *, closing: bool = False
    ) -> ComboFill | None:
        """Price a combo order atomically. ``closing`` reverses every leg's side."""
        if len(legs) != len(quotes) or not all(q.tradable for q in quotes):
            return None
        direction = -1 if closing else 1
        prices = tuple(
            self.leg_price(direction * leg.side, q) for leg, q in zip(legs, quotes, strict=True)
        )
        mids = tuple((q.bid + q.ask) / 2.0 for q in quotes)

        def package(values: tuple[float, ...]) -> float:
            return sum(
                direction * leg.side * leg.ratio * v for leg, v in zip(legs, values, strict=True)
            )

        return ComboFill(package(prices), package(mids), prices, mids)

    def costs(self, legs: tuple[Leg, ...], quantity: int) -> tuple[float, float]:
        """(commission, fees) in dollars for trading every leg once."""
        contracts = sum(leg.ratio for leg in legs) * quantity
        cfg = self.config
        return contracts * cfg.commission_per_contract, contracts * cfg.fee_per_contract
