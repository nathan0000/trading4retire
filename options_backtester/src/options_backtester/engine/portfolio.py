"""Position tracking, margin and portfolio-level Greeks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from options_backtester.engine.execution import ComboFill
from options_backtester.regimes.classifier import RegimeState
from options_backtester.strategies.base import Leg, LegQuote

GREEKS = ("delta", "gamma", "theta", "vega")


def expiration_payoff(legs: tuple[Leg, ...], underlying: float) -> float:
    """Per-unit value of the package at expiration (single-expiration structures)."""
    return sum(leg.side * leg.ratio * leg.intrinsic(underlying) for leg in legs)


def max_loss_per_unit(legs: tuple[Leg, ...], entry_value: float) -> float:
    """Worst-case loss in option points, or ``inf`` when risk is undefined.

    Multi-expiration structures (calendars) are treated as debit-limited when net
    long the later expiration; otherwise their risk is reported as undefined.
    """
    if len({leg.expiration for leg in legs}) > 1:
        return max(entry_value, 0.0) if entry_value > 0 else float("inf")
    net_calls = sum(leg.side * leg.ratio for leg in legs if leg.option_type == "CALL")
    if net_calls < 0:
        return float("inf")
    points = [0.0, *sorted({leg.strike for leg in legs})]
    worst = min(expiration_payoff(legs, s) - entry_value for s in points)
    return max(-worst, 0.0)


def margin_per_unit(
    legs: tuple[Leg, ...], quotes: tuple[LegQuote, ...], entry_value: float, underlying: float
) -> float:
    """Margin requirement in option points: max loss, or a Reg-T style naked estimate."""
    loss = max_loss_per_unit(legs, entry_value)
    if np.isfinite(loss):
        return loss
    naked, premiums = [], []
    for leg, quote in zip(legs, quotes, strict=True):
        if leg.side >= 0:
            continue
        if leg.option_type == "CALL":
            otm, floor = max(leg.strike - underlying, 0.0), 0.10 * underlying
        else:
            otm, floor = max(underlying - leg.strike, 0.0), 0.10 * leg.strike
        naked.append(leg.ratio * (max(0.20 * underlying - otm, floor) + quote.mid))
        premiums.append(leg.ratio * quote.mid)
    if not naked:
        return max(entry_value, 0.0)
    worst = int(np.argmax(naked))
    return naked[worst] + sum(p for i, p in enumerate(premiums) if i != worst)


@dataclass
class Position:
    trade_id: int
    strategy: str
    symbol: str
    legs: tuple[Leg, ...]
    quantity: int
    entry_time: pd.Timestamp
    entry_underlying: float
    entry_fill: ComboFill
    entry_quotes: tuple[LegQuote, ...]
    entry_commission: float
    entry_fees: float
    margin: float  # dollars
    max_loss: float  # dollars, inf when undefined
    entry_regime: RegimeState | None = None
    # mutable mark state
    mark_value: float = 0.0  # per-unit value at the last complete mid mark
    mark_quotes: tuple[LegQuote, ...] = ()
    peak_pnl: float = 0.0
    skipped_marks: int = 0
    assignment_risk_flags: int = 0

    def __post_init__(self) -> None:
        self.mark_value = self.entry_fill.mid_value
        self.mark_quotes = self.entry_quotes

    @property
    def entry_value(self) -> float:
        return self.entry_fill.value

    @property
    def basis(self) -> float:
        """Credit received or debit paid, per unit, in option points."""
        return abs(self.entry_value)

    @property
    def is_credit(self) -> bool:
        return self.entry_value < 0

    @property
    def mark_pnl(self) -> float:
        """Per-unit mark-to-mid P&L in option points."""
        return self.mark_value - self.entry_value

    def greeks(self, multiplier: int) -> dict[str, float]:
        scale = multiplier * self.quantity
        return {
            name: scale
            * float(
                np.nansum(
                    [
                        leg.side * leg.ratio * getattr(quote, name)
                        for leg, quote in zip(self.legs, self.mark_quotes, strict=True)
                    ]
                )
            )
            for name in GREEKS
        }


@dataclass
class Portfolio:
    initial_capital: float
    multiplier: int = 100
    cash: float = field(init=False)
    positions: dict[int, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def open(self, position: Position) -> None:
        notional = position.entry_value * self.multiplier * position.quantity
        self.cash -= notional + position.entry_commission + position.entry_fees
        self.positions[position.trade_id] = position

    def close(self, trade_id: int, exit_value: float, commission: float, fees: float) -> Position:
        """Close at ``exit_value`` (per-unit signed package value received on exit)."""
        position = self.positions.pop(trade_id)
        self.cash += exit_value * self.multiplier * position.quantity - commission - fees
        return position

    @property
    def market_value(self) -> float:
        return sum(p.mark_value * self.multiplier * p.quantity for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def margin_used(self) -> float:
        return sum(p.margin for p in self.positions.values())

    @property
    def buying_power(self) -> float:
        return self.equity - self.margin_used

    def greeks(self) -> dict[str, float]:
        totals = dict.fromkeys(GREEKS, 0.0)
        for position in self.positions.values():
            for name, value in position.greeks(self.multiplier).items():
                totals[name] += value
        return totals
