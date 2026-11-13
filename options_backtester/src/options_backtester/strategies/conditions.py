"""Modular entry and exit rule evaluators.

Exit triggers are evaluated on the *mark* (mid) and have an explicit priority:
when several fire on the same bar the highest-priority reason is recorded.
Risk exits outrank profit exits, which outrank scheduled exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from options_backtester.regimes.classifier import RegimeState
from options_backtester.strategies.base import (
    EntryRules,
    ExitRules,
    Leg,
    LegQuote,
    MarketContext,
    Rejection,
)


class ExitReason(StrEnum):
    EXPIRATION = "EXPIRATION"
    STOP_LOSS = "STOP_LOSS"
    UNDERLYING_BREACH = "UNDERLYING_BREACH"
    DELTA_BREACH = "DELTA_BREACH"
    TRAILING_STOP = "TRAILING_STOP"
    PROFIT_TARGET = "PROFIT_TARGET"
    DTE_EXIT = "DTE_EXIT"
    MAX_HOLD = "MAX_HOLD"
    REGIME_SHIFT = "REGIME_SHIFT"
    END_OF_DATA = "END_OF_DATA"


#: Highest priority first.
EXIT_PRIORITY: tuple[ExitReason, ...] = (
    ExitReason.EXPIRATION,
    ExitReason.STOP_LOSS,
    ExitReason.UNDERLYING_BREACH,
    ExitReason.DELTA_BREACH,
    ExitReason.TRAILING_STOP,
    ExitReason.PROFIT_TARGET,
    ExitReason.DTE_EXIT,
    ExitReason.MAX_HOLD,
    ExitReason.REGIME_SHIFT,
)


# -- entry ---------------------------------------------------------------


def _regime_allowed(rules: EntryRules, regime: RegimeState | None) -> bool:
    checks = (
        (rules.allowed_trend, "trend"),
        (rules.allowed_volatility, "volatility"),
        (rules.allowed_vol_dynamics, "vol_dynamics"),
    )
    for allowed, attribute in checks:
        if allowed is None:
            continue
        if regime is None or getattr(regime, attribute) not in allowed:
            return False
    return True


def check_pre_entry(rules: EntryRules, context: MarketContext) -> Rejection | None:
    """Filters that do not depend on the selected legs."""
    if rules.entry_time is not None:
        local = context.timestamp.tz_convert("America/New_York").strftime("%H:%M")
        if local != rules.entry_time:
            return Rejection("OUTSIDE_ENTRY_TIME", local)
    if not _regime_allowed(rules, context.regime):
        return Rejection("REGIME_FILTER", "" if context.regime is None else context.regime.label)
    if rules.min_iv_rank is not None or rules.max_iv_rank is not None:
        rank = context.iv_rank
        if rank is None or not np.isfinite(rank):
            return Rejection("IV_RANK_UNAVAILABLE")
        if rules.min_iv_rank is not None and rank < rules.min_iv_rank:
            return Rejection("IV_RANK_TOO_LOW", f"{rank:.1f}")
        if rules.max_iv_rank is not None and rank > rules.max_iv_rank:
            return Rejection("IV_RANK_TOO_HIGH", f"{rank:.1f}")
    return None


def check_liquidity(
    rules: EntryRules, legs: tuple[Leg, ...], quotes: tuple[LegQuote, ...]
) -> Rejection | None:
    for leg, quote in zip(legs, quotes, strict=True):
        label = f"{leg.option_type} {leg.strike:g}"
        if not quote.tradable:
            return Rejection("NO_QUOTE", label)
        if leg.side < 0 and quote.bid <= 0:
            return Rejection("ZERO_BID", label)
        if rules.max_spread_pct is not None and quote.mid > 0:
            if (quote.ask - quote.bid) / quote.mid > rules.max_spread_pct:
                return Rejection("SPREAD_TOO_WIDE", label)
        if rules.min_open_interest is not None:
            if not quote.open_interest >= rules.min_open_interest:  # NaN fails
                return Rejection("OPEN_INTEREST_TOO_LOW", label)
    return None


def check_fill(rules: EntryRules, fill_value: float) -> Rejection | None:
    """Credit/debit bounds on the *executable* package price (negative = credit)."""
    credit = -fill_value
    if credit > 0:
        if rules.min_credit is not None and credit < rules.min_credit:
            return Rejection("CREDIT_TOO_LOW", f"{credit:.2f}")
        if rules.max_credit is not None and credit > rules.max_credit:
            return Rejection("CREDIT_TOO_HIGH", f"{credit:.2f}")
    else:
        if rules.min_credit is not None:
            return Rejection("CREDIT_TOO_LOW", f"{credit:.2f}")
        if rules.max_debit is not None and -credit > rules.max_debit:
            return Rejection("DEBIT_TOO_HIGH", f"{-credit:.2f}")
    return None


# -- exit ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionState:
    """Mark-based view of one open position at one timestamp (per unit, option points)."""

    basis: float  # |entry value|: credit received or debit paid
    pnl: float  # mark-to-mid P&L
    peak_pnl: float
    min_dte: int
    days_held: int
    underlying_price: float
    short_put_strikes: tuple[float, ...] = ()
    short_call_strikes: tuple[float, ...] = ()
    position_delta: float = float("nan")
    entry_regime: RegimeState | None = None
    regime: RegimeState | None = None


def triggered_exits(
    rules: ExitRules, state: PositionState, entry_rules: EntryRules | None = None
) -> list[ExitReason]:
    """Every exit rule that fires for ``state``, in priority order."""
    fired: set[ExitReason] = set()
    basis = state.basis
    if state.min_dte <= 0:
        fired.add(ExitReason.EXPIRATION)
    if rules.stop_loss_multiple is not None and state.pnl <= -rules.stop_loss_multiple * basis:
        fired.add(ExitReason.STOP_LOSS)
    if rules.underlying_breach and (
        any(state.underlying_price <= k for k in state.short_put_strikes)
        or any(state.underlying_price >= k for k in state.short_call_strikes)
    ):
        fired.add(ExitReason.UNDERLYING_BREACH)
    if (
        rules.max_position_delta is not None
        and abs(state.position_delta) > rules.max_position_delta
    ):
        fired.add(ExitReason.DELTA_BREACH)
    if (
        rules.trailing_stop_pct is not None
        and state.peak_pnl >= rules.trailing_activation_pct * basis
        and state.peak_pnl - state.pnl >= rules.trailing_stop_pct * basis
    ):
        fired.add(ExitReason.TRAILING_STOP)
    if rules.profit_target_pct is not None and state.pnl >= rules.profit_target_pct * basis:
        fired.add(ExitReason.PROFIT_TARGET)
    if rules.exit_dte is not None and state.min_dte <= rules.exit_dte:
        fired.add(ExitReason.DTE_EXIT)
    if rules.max_hold_days is not None and state.days_held >= rules.max_hold_days:
        fired.add(ExitReason.MAX_HOLD)
    if rules.exit_on_regime_shift and _regime_shifted(state, entry_rules):
        fired.add(ExitReason.REGIME_SHIFT)
    return [reason for reason in EXIT_PRIORITY if reason in fired]


def evaluate_exit(
    rules: ExitRules, state: PositionState, entry_rules: EntryRules | None = None
) -> ExitReason | None:
    fired = triggered_exits(rules, state, entry_rules)
    return fired[0] if fired else None


def _regime_shifted(state: PositionState, entry_rules: EntryRules | None) -> bool:
    now, then = state.regime, state.entry_regime
    if now is None:
        return False
    filtered = entry_rules is not None and any(
        allowed is not None
        for allowed in (
            entry_rules.allowed_trend,
            entry_rules.allowed_volatility,
            entry_rules.allowed_vol_dynamics,
        )
    )
    if filtered:  # the regime left the set the strategy is allowed to trade
        return not _regime_allowed(entry_rules, now)
    if then is None:
        return False
    return (now.trend, now.volatility) != (then.trend, then.volatility)
