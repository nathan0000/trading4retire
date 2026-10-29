"""Phase 5 iron-condor candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from math import isfinite

from spx_0dte_backtest.data import OptionQuote
from spx_0dte_backtest.selection import (
    SelectionFailure,
    StrikeSelectionSettings,
    StrikeSelector,
)
from spx_0dte_backtest.strategies.base import (
    MarketContext,
    RejectionReason,
    StrategyType,
    TradeCandidate,
    TradeLeg,
    TradeLegRole,
)


@dataclass(frozen=True, slots=True)
class IronCondorSettings:
    min_credit: float = 1.0
    max_credit: float = 2.5
    first_entry: time = time(10, 45)
    last_entry: time = time(15, 15)
    entry_interval_minutes: int = 15
    strike_selection: StrikeSelectionSettings = field(default_factory=StrikeSelectionSettings)

    def __post_init__(self) -> None:
        if not all(isfinite(value) and value >= 0 for value in (self.min_credit, self.max_credit)):
            raise ValueError("credit bounds must be finite and non-negative")
        if self.min_credit > self.max_credit:
            raise ValueError("minimum credit cannot exceed maximum credit")
        if self.first_entry > self.last_entry:
            raise ValueError("first entry cannot be later than last entry")
        if self.entry_interval_minutes <= 0:
            raise ValueError("entry interval must be positive")


class IronCondorStrategy:
    """Generate quoted iron-condor candidates without simulating execution."""

    def __init__(self, settings: IronCondorSettings | None = None):
        self.settings = settings or IronCondorSettings()
        self._selector = StrikeSelector(self.settings.strike_selection)

    def generate_trade(self, context: MarketContext) -> TradeCandidate:
        if not self._is_entry_time(context):
            return self._rejected(context, RejectionReason.OUTSIDE_TRADING_WINDOW)

        selection = self._selector.select(
            context.option_quotes,
            context.timestamp,
            context.expiration,
            context.spx_price,
        )
        if not selection.is_success:
            reason = {
                SelectionFailure.NO_OPTION_DATA: RejectionReason.NO_OPTION_DATA,
                SelectionFailure.NO_VALID_STRIKE: RejectionReason.NO_VALID_STRIKE,
                SelectionFailure.NO_VALID_WING: RejectionReason.NO_VALID_STRIKE,
                SelectionFailure.INCORRECT_EXPIRATION: RejectionReason.INCORRECT_EXPIRATION,
            }[selection.failure]
            return self._rejected(context, reason, selection.detail)

        short_put = selection.short_put
        long_put = selection.long_put
        short_call = selection.short_call
        long_call = selection.long_call
        assert short_put and long_put and short_call and long_call

        put_credit = short_put.bid - long_put.ask
        call_credit = short_call.bid - long_call.ask
        conservative_credit = put_credit + call_credit
        mid_credit = (
            self._mid(short_put)
            - self._mid(long_put)
            + self._mid(short_call)
            - self._mid(long_call)
        )
        economics = {
            "theoretical_mid_credit": mid_credit,
            "conservative_executable_credit": conservative_credit,
            "put_credit": put_credit,
            "call_credit": call_credit,
        }
        if conservative_credit < self.settings.min_credit:
            return self._rejected(context, RejectionReason.CREDIT_TOO_LOW, economics=economics)
        if conservative_credit > self.settings.max_credit:
            return self._rejected(context, RejectionReason.CREDIT_TOO_HIGH, economics=economics)

        width = self.settings.strike_selection.wing_width
        return TradeCandidate(
            strategy=StrategyType.IRON_CONDOR,
            timestamp=context.timestamp,
            expiration=context.expiration,
            spx_price=context.spx_price,
            legs=(
                TradeLeg(TradeLegRole.LONG_PUT, long_put, 1),
                TradeLeg(TradeLegRole.SHORT_PUT, short_put, -1),
                TradeLeg(TradeLegRole.SHORT_CALL, short_call, -1),
                TradeLeg(TradeLegRole.LONG_CALL, long_call, 1),
            ),
            **economics,
            max_profit_points=conservative_credit,
            max_loss_points=width - conservative_credit,
            short_put_delta=short_put.delta,
            short_call_delta=short_call.delta,
            context_features=context.features,
        )

    def _is_entry_time(self, context: MarketContext) -> bool:
        local = context.timestamp.tz_convert("America/New_York")
        current = local.time().replace(tzinfo=None)
        if current < self.settings.first_entry or current > self.settings.last_entry:
            return False
        first_minutes = self.settings.first_entry.hour * 60 + self.settings.first_entry.minute
        current_minutes = current.hour * 60 + current.minute
        return (current_minutes - first_minutes) % self.settings.entry_interval_minutes == 0

    @staticmethod
    def _mid(quote: OptionQuote) -> float:
        return (quote.bid + quote.ask) / 2.0

    @staticmethod
    def _rejected(
        context: MarketContext,
        reason: RejectionReason,
        detail: str | None = None,
        economics: dict[str, float] | None = None,
    ) -> TradeCandidate:
        return TradeCandidate(
            strategy=StrategyType.IRON_CONDOR,
            timestamp=context.timestamp,
            expiration=context.expiration,
            spx_price=context.spx_price,
            rejection_reason=reason,
            rejection_detail=detail,
            context_features=context.features,
            **(economics or {}),
        )
