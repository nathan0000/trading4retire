"""Realistic, deterministic per-leg quote execution for Phase 6."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

import pandas as pd

from spx_0dte_backtest.data import OptionQuote
from spx_0dte_backtest.strategies import TradeCandidate, TradeLeg, TradeLegRole


class MissingExitQuote(ValueError):
    """Raised when a snapshot has no quote for one of the open legs."""


class ExecutionMode(StrEnum):
    MID = "MID"
    BID_ASK = "BID_ASK"
    BID_ASK_PLUS_SLIPPAGE = "BID_ASK_PLUS_SLIPPAGE"


class ExecutionPurpose(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class CostModel:
    commission_per_contract: float = 0.65
    exchange_fee_per_contract: float = 0.10

    def __post_init__(self) -> None:
        values = (self.commission_per_contract, self.exchange_fee_per_contract)
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("per-contract costs must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    mode: ExecutionMode = ExecutionMode.BID_ASK_PLUS_SLIPPAGE
    slippage_points: float = 0.05
    contract_multiplier: float = 100.0
    cost_model: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        if not isfinite(self.slippage_points) or self.slippage_points < 0:
            raise ValueError("slippage must be finite and non-negative")
        if not isfinite(self.contract_multiplier) or self.contract_multiplier <= 0:
            raise ValueError("contract multiplier must be finite and positive")


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    role: TradeLegRole
    side: OrderSide
    quantity: int
    quote: OptionQuote
    fill_price: float
    cashflow_points: float


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    purpose: ExecutionPurpose
    mode: ExecutionMode
    fills: tuple[ExecutionFill, ...]
    gross_cashflow_points: float
    gross_cashflow_dollars: float
    commission: float
    fees: float
    net_cashflow_dollars: float

    @property
    def entry_credit_points(self) -> float | None:
        return self.gross_cashflow_points if self.purpose is ExecutionPurpose.ENTRY else None

    @property
    def exit_debit_points(self) -> float | None:
        return -self.gross_cashflow_points if self.purpose is ExecutionPurpose.EXIT else None


@dataclass(frozen=True, slots=True)
class _ContractKey:
    expiration: object
    option_type: object
    strike: float
    contract_symbol: str | None


class ExecutionEngine:
    """Execute every option leg independently against its historical quote."""

    def __init__(self, settings: ExecutionSettings | None = None):
        self.settings = settings or ExecutionSettings()

    def execute_entry(self, candidate: TradeCandidate) -> ExecutionResult:
        if not candidate.is_accepted:
            raise ValueError("only an accepted four-leg candidate can be executed")
        orders = tuple(
            (leg, OrderSide.BUY if leg.quantity > 0 else OrderSide.SELL, leg.quote)
            for leg in candidate.legs
        )
        return self._execute(orders, ExecutionPurpose.ENTRY)

    def execute_exit(
        self,
        open_legs: tuple[TradeLeg, ...],
        exit_quotes: Iterable[OptionQuote],
    ) -> ExecutionResult:
        orders = tuple(
            (leg, OrderSide.SELL if leg.quantity > 0 else OrderSide.BUY, quote)
            for leg, quote in self._match_exit_quotes(open_legs, exit_quotes)
        )
        return self._execute(orders, ExecutionPurpose.EXIT)

    def mark_to_close(
        self,
        open_legs: tuple[TradeLeg, ...],
        quotes: Iterable[OptionQuote],
    ) -> float:
        """Mid-price cost to close the open legs, in points, floored at zero."""

        cost = sum(
            -leg.quantity * (quote.bid + quote.ask) / 2.0
            for leg, quote in self._match_exit_quotes(open_legs, quotes)
        )
        return max(0.0, cost)

    def _match_exit_quotes(
        self,
        open_legs: tuple[TradeLeg, ...],
        exit_quotes: Iterable[OptionQuote],
    ) -> tuple[tuple[TradeLeg, OptionQuote], ...]:
        if not open_legs:
            raise ValueError("open legs are required")
        lookup: dict[_ContractKey, OptionQuote] = {}
        for quote in exit_quotes:
            key = self._key(quote)
            if key in lookup:
                raise ValueError(f"duplicate exit quote for {quote.contract_symbol or key}")
            lookup[key] = quote

        matched: list[tuple[TradeLeg, OptionQuote]] = []
        timestamps: set[pd.Timestamp] = set()
        for leg in open_legs:
            quote = lookup.get(self._key(leg.quote))
            if quote is None:
                raise MissingExitQuote(
                    f"missing exit quote for {leg.quote.contract_symbol or self._key(leg.quote)}"
                )
            if pd.Timestamp(quote.timestamp) <= pd.Timestamp(leg.quote.timestamp):
                raise ValueError("exit quote must be later than its entry quote")
            timestamps.add(pd.Timestamp(quote.timestamp))
            matched.append((leg, quote))
        if len(timestamps) != 1:
            raise ValueError("all exit quotes must come from one exact snapshot")
        return tuple(matched)

    def _execute(
        self,
        orders: tuple[tuple[TradeLeg, OrderSide, OptionQuote], ...],
        purpose: ExecutionPurpose,
    ) -> ExecutionResult:
        fills: list[ExecutionFill] = []
        contracts = 0
        for leg, side, quote in orders:
            quantity = abs(leg.quantity)
            fill_price = self._fill_price(side, quote)
            sign = 1.0 if side is OrderSide.SELL else -1.0
            fills.append(
                ExecutionFill(
                    role=leg.role,
                    side=side,
                    quantity=quantity,
                    quote=quote,
                    fill_price=fill_price,
                    cashflow_points=sign * quantity * fill_price,
                )
            )
            contracts += quantity

        gross_points = sum(fill.cashflow_points for fill in fills)
        commission = contracts * self.settings.cost_model.commission_per_contract
        fees = contracts * self.settings.cost_model.exchange_fee_per_contract
        gross_dollars = gross_points * self.settings.contract_multiplier
        return ExecutionResult(
            purpose=purpose,
            mode=self.settings.mode,
            fills=tuple(fills),
            gross_cashflow_points=gross_points,
            gross_cashflow_dollars=gross_dollars,
            commission=commission,
            fees=fees,
            net_cashflow_dollars=gross_dollars - commission - fees,
        )

    def _fill_price(self, side: OrderSide, quote: OptionQuote) -> float:
        if self.settings.mode is ExecutionMode.MID:
            return (quote.bid + quote.ask) / 2.0
        if side is OrderSide.BUY:
            price = quote.ask
            if self.settings.mode is ExecutionMode.BID_ASK_PLUS_SLIPPAGE:
                price += self.settings.slippage_points
            return price
        price = quote.bid
        if self.settings.mode is ExecutionMode.BID_ASK_PLUS_SLIPPAGE:
            price -= self.settings.slippage_points
        return max(0.0, price)

    @staticmethod
    def _key(quote: OptionQuote) -> _ContractKey:
        return _ContractKey(
            expiration=quote.expiration,
            option_type=quote.option_type,
            strike=quote.strike,
            contract_symbol=quote.contract_symbol,
        )
