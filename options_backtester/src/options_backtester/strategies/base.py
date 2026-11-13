"""Declarative strategy configuration, the vectorized ``LegBuilder`` and ``OptionStrategy``.

A strategy never hardcodes a strike or a date: it describes *how* to choose
them (by delta, percent OTM or standard deviations; wings by dollars, delta or
percent of the underlying) and the builder resolves that against each chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from options_backtester.data.base import OptionChainSnapshot
from options_backtester.regimes.classifier import RegimeState

# --------------------------------------------------------------------------
# Configuration schema (JSON / YAML / dict driven)
# --------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrikeRule(_Model):
    """How to choose the anchor strike of a structure."""

    method: Literal["delta", "pct_otm", "stdev"] = "delta"
    #: |delta| (0-1), fraction OTM (0.05 = 5%), or number of standard deviations.
    value: float = Field(gt=0)
    #: For ``delta``: reject if the nearest available |delta| is further away than this.
    max_delta_error: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def _check(self) -> StrikeRule:
        if self.method == "delta" and self.value >= 1:
            raise ValueError("delta strike rule needs 0 < value < 1")
        return self


class WingRule(_Model):
    """How far the protective leg sits beyond the anchor strike."""

    method: Literal["dollar", "delta", "pct_underlying"] = "dollar"
    #: Dollar width, |delta| differential, or fraction of the underlying price.
    value: float = Field(gt=0)


class EntryRules(_Model):
    allowed_trend: list[str] | None = None
    allowed_volatility: list[str] | None = None
    allowed_vol_dynamics: list[str] | None = None
    min_iv_rank: float | None = Field(default=None, ge=0, le=100)
    max_iv_rank: float | None = Field(default=None, ge=0, le=100)
    #: Per-leg (ask - bid) / mid ceiling.
    max_spread_pct: float | None = Field(default=None, gt=0)
    min_open_interest: int | None = Field(default=None, ge=0)
    #: Bounds on the executable net credit (credit strategies) per unit, in option points.
    min_credit: float | None = None
    max_credit: float | None = None
    #: Bound on the executable net debit (debit strategies) per unit.
    max_debit: float | None = None
    #: New York time-of-day ("HH:MM") a snapshot must carry to allow entry; None = any.
    entry_time: str | None = None
    min_days_between_entries: int = Field(default=1, ge=0)
    max_open_positions: int = Field(default=1, ge=1)


class ExitRules(_Model):
    #: Fraction of the credit received / debit paid to capture.
    profit_target_pct: float | None = Field(default=0.5, gt=0)
    #: Loss, as a multiple of the credit received / debit paid, that stops the trade.
    #: A 2.0 stop on a 1.50 credit triggers when the cost to close reaches 4.50.
    stop_loss_multiple: float | None = Field(default=2.0, gt=0)
    exit_dte: int | None = Field(default=21, ge=0)
    max_hold_days: int | None = Field(default=None, ge=1)
    #: Exit when the underlying trades through any short strike.
    underlying_breach: bool = False
    #: Exit when |net position delta| per unit (in deltas, e.g. 0.30) exceeds this.
    max_position_delta: float | None = Field(default=None, gt=0)
    #: Give-back from peak profit, as a fraction of credit/debit, that exits the trade.
    trailing_stop_pct: float | None = Field(default=None, gt=0)
    #: Peak profit (fraction of credit/debit) required before the trailing stop arms.
    trailing_activation_pct: float = Field(default=0.25, ge=0)
    exit_on_regime_shift: bool = False


class StrategyConfig(_Model):
    name: str = "strategy"
    template: Literal["iron_condor", "vertical", "strangle", "calendar"]
    min_dte: int = Field(default=30, ge=0)
    max_dte: int = Field(default=45, ge=0)
    #: Preferred DTE inside the window; defaults to its midpoint.
    target_dte: int | None = None
    strike: StrikeRule = StrikeRule(method="delta", value=0.16)
    wing: WingRule | None = None
    #: Verticals and calendars: which side of the chain to trade.
    option_side: Literal["put", "call"] = "put"
    #: Verticals: sell the anchor (credit) or buy it (debit).
    direction: Literal["credit", "debit"] = "credit"
    #: Calendars: DTE window of the long back-month leg.
    back_min_dte: int | None = None
    back_max_dte: int | None = None
    quantity: int = Field(default=1, ge=1)
    entry: EntryRules = EntryRules()
    exit: ExitRules = ExitRules()

    @model_validator(mode="after")
    def _check(self) -> StrategyConfig:
        if self.min_dte > self.max_dte:
            raise ValueError("min_dte cannot exceed max_dte")
        if self.template in ("iron_condor", "vertical") and self.wing is None:
            raise ValueError(f"{self.template} requires a wing rule")
        if self.template == "calendar":
            if self.back_min_dte is None or self.back_max_dte is None:
                raise ValueError("calendar requires back_min_dte and back_max_dte")
            if self.back_min_dte <= self.max_dte:
                raise ValueError("calendar back_min_dte must exceed the front max_dte")
        return self


# --------------------------------------------------------------------------
# Domain objects
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Leg:
    expiration: date
    strike: float
    option_type: str  # "CALL" | "PUT"
    side: int  # +1 long, -1 short
    ratio: int = 1

    def intrinsic(self, underlying: float) -> float:
        if self.option_type == "CALL":
            return max(underlying - self.strike, 0.0)
        return max(self.strike - underlying, 0.0)


@dataclass(frozen=True, slots=True)
class LegQuote:
    bid: float
    ask: float
    mid: float
    volume: float = float("nan")
    open_interest: float = float("nan")
    implied_volatility: float = float("nan")
    delta: float = float("nan")
    gamma: float = float("nan")
    theta: float = float("nan")
    vega: float = float("nan")

    @classmethod
    def from_row(cls, row: pd.Series) -> LegQuote:
        return cls(**{name: float(row[name]) for name in cls.__dataclass_fields__})

    @property
    def tradable(self) -> bool:
        return bool(
            np.isfinite(self.bid) and np.isfinite(self.ask) and self.ask > 0 <= self.bid <= self.ask
        )


def quote_legs(chain: OptionChainSnapshot, legs: tuple[Leg, ...]) -> tuple[LegQuote, ...] | None:
    """Quotes for every leg from one snapshot, or ``None`` if any leg is not tradable."""
    quotes = []
    for leg in legs:
        row = chain.lookup(leg.expiration, leg.strike, leg.option_type)
        if row is None:
            return None
        quote = LegQuote.from_row(row)
        if not quote.tradable:
            return None
        quotes.append(quote)
    return tuple(quotes)


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Everything a strategy may see at ``timestamp``; all of it is known at that time."""

    timestamp: pd.Timestamp
    chain: OptionChainSnapshot
    regime: RegimeState | None = None

    @property
    def iv_rank(self) -> float | None:
        if self.chain.iv_rank is not None:
            return self.chain.iv_rank
        return self.regime.iv_rank if self.regime is not None else None


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    timestamp: pd.Timestamp
    strategy: str
    symbol: str
    underlying_price: float
    legs: tuple[Leg, ...]
    quotes: tuple[LegQuote, ...]

    @property
    def mid_value(self) -> float:
        """Signed value of the package at mid: negative for a net credit."""
        return sum(
            leg.side * leg.ratio * q.mid for leg, q in zip(self.legs, self.quotes, strict=True)
        )


@dataclass(frozen=True, slots=True)
class Rejection:
    reason: str
    detail: str = ""


# --------------------------------------------------------------------------
# Leg selection
# --------------------------------------------------------------------------


class LegBuilder:
    """Vectorized expiration / strike / wing selection over a canonical chain frame."""

    @staticmethod
    def pick_expiration(
        quotes: pd.DataFrame, min_dte: int, max_dte: int, target_dte: int | None = None
    ) -> pd.Timestamp | None:
        window = quotes.loc[
            (quotes["dte"] >= min_dte) & (quotes["dte"] <= max_dte), ["expiration", "dte"]
        ].drop_duplicates()
        if window.empty:
            return None
        target = (min_dte + max_dte) / 2.0 if target_dte is None else target_dte
        distance = (window["dte"] - target).abs().to_numpy()
        # Ties resolve to the longer-dated expiration (more time, less gamma).
        order = np.lexsort((-window["dte"].to_numpy(), distance))
        return pd.Timestamp(window["expiration"].iloc[order[0]])

    @staticmethod
    def side_frame(quotes: pd.DataFrame, expiration: pd.Timestamp, option_type: str):
        mask = (quotes["expiration"] == expiration) & (quotes["option_type"] == option_type)
        return quotes.loc[mask].sort_values("strike").reset_index(drop=True)

    @staticmethod
    def pick_strike(side: pd.DataFrame, rule: StrikeRule, underlying: float) -> pd.Series | None:
        """Choose the anchor row on one expiration/side frame."""
        if side.empty:
            return None
        is_call = side["option_type"].iloc[0] == "CALL"
        strikes = side["strike"].to_numpy()
        if rule.method == "delta":
            delta = side["delta"].abs().to_numpy()
            valid = np.isfinite(delta)
            if not valid.any():
                return None
            error = np.where(valid, np.abs(delta - rule.value), np.inf)
            # Ties resolve to the strike further out of the money.
            otm_rank = -strikes if is_call else strikes
            best = np.lexsort((otm_rank, error))[0]
            if error[best] > rule.max_delta_error:
                return None
            return side.iloc[best]
        if rule.method == "pct_otm":
            target = underlying * (1 + rule.value if is_call else 1 - rule.value)
        else:  # stdev
            atm = side.iloc[int(np.argmin(np.abs(strikes - underlying)))]
            iv, dte = float(atm["implied_volatility"]), float(atm["dte"])
            if not np.isfinite(iv) or iv <= 0:
                return None
            move = rule.value * iv * np.sqrt(max(dte, 1.0) / 365.0)
            target = underlying * np.exp(move if is_call else -move)
        return side.iloc[int(np.argmin(np.abs(strikes - target)))]

    @staticmethod
    def pick_wing(
        side: pd.DataFrame, anchor: pd.Series, rule: WingRule, underlying: float
    ) -> pd.Series | None:
        """Choose the row strictly further OTM than ``anchor`` that best matches ``rule``."""
        is_call = anchor["option_type"] == "CALL"
        strikes = side["strike"].to_numpy()
        beyond = strikes > anchor["strike"] if is_call else strikes < anchor["strike"]
        if not beyond.any():
            return None
        if rule.method == "delta":
            target = abs(float(anchor["delta"])) - rule.value
            delta = side["delta"].abs().to_numpy()
            if not np.isfinite(target) or target <= 0:
                return None
            error = np.where(beyond & np.isfinite(delta), np.abs(delta - target), np.inf)
        else:
            width = rule.value if rule.method == "dollar" else underlying * rule.value
            target = anchor["strike"] + (width if is_call else -width)
            error = np.where(beyond, np.abs(strikes - target), np.inf)
        best = int(np.argmin(error))
        return None if not np.isfinite(error[best]) else side.iloc[best]


def make_leg(row: pd.Series, side: int, ratio: int = 1) -> tuple[Leg, LegQuote]:
    leg = Leg(
        expiration=pd.Timestamp(row["expiration"]).date(),
        strike=float(row["strike"]),
        option_type=str(row["option_type"]),
        side=side,
        ratio=ratio,
    )
    return leg, LegQuote.from_row(row)


# --------------------------------------------------------------------------
# Strategy base
# --------------------------------------------------------------------------


class OptionStrategy(ABC):
    """Base class: templates only describe which legs make up the structure."""

    builder = LegBuilder

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def select_legs(self, chain: OptionChainSnapshot) -> list[tuple[Leg, LegQuote]] | Rejection:
        """Resolve the declarative rules against one chain snapshot."""

    def generate(self, context: MarketContext) -> TradeCandidate | Rejection:
        from options_backtester.strategies import conditions

        rejection = conditions.check_pre_entry(self.config.entry, context)
        if rejection is not None:
            return rejection
        selected = self.select_legs(context.chain)
        if isinstance(selected, Rejection):
            return selected
        legs = tuple(leg for leg, _ in selected)
        quotes = tuple(quote for _, quote in selected)
        rejection = conditions.check_liquidity(self.config.entry, legs, quotes)
        if rejection is not None:
            return rejection
        return TradeCandidate(
            timestamp=context.timestamp,
            strategy=self.config.name,
            symbol=context.chain.underlying_symbol,
            underlying_price=context.chain.underlying_price,
            legs=legs,
            quotes=quotes,
        )

    # Shared helpers for templates -------------------------------------
    def _front_expiration(self, chain: OptionChainSnapshot) -> pd.Timestamp | None:
        cfg = self.config
        return self.builder.pick_expiration(chain.quotes, cfg.min_dte, cfg.max_dte, cfg.target_dte)

    def _vertical(
        self,
        chain: OptionChainSnapshot,
        expiration: pd.Timestamp,
        option_type: str,
        anchor_side: int,
    ) -> list[tuple[Leg, LegQuote]] | Rejection:
        """Anchor leg plus a wing further OTM on the opposite side of the book."""
        cfg = self.config
        side = self.builder.side_frame(chain.quotes, expiration, option_type)
        anchor = self.builder.pick_strike(side, cfg.strike, chain.underlying_price)
        if anchor is None:
            return Rejection("NO_VALID_STRIKE", f"{option_type} anchor")
        assert cfg.wing is not None
        wing = self.builder.pick_wing(side, anchor, cfg.wing, chain.underlying_price)
        if wing is None:
            return Rejection("NO_VALID_WING", f"{option_type} wing")
        return [make_leg(anchor, anchor_side), make_leg(wing, -anchor_side)]
