"""Main event loop.

One step per underlying bar. At each step the engine (1) settles expirations and
marks/exits open positions, (2) evaluates a new entry, (3) records equity and
portfolio Greeks. Decisions and fills use only the chain snapshot and regime
state known at that step's timestamp.

Design points carried over from the 0DTE engine's code review:

* exit triggers are evaluated on the mid mark, fills happen at executable prices;
* a missing or one-sided quote skips that mark (counted) instead of aborting the run;
* credit/debit filters are applied to the executable fill, not to mid;
* entries respect position-count and margin limits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from options_backtester.data.base import (
    DataProvider,
    DataUnavailableError,
    OptionChainSnapshot,
    market_date,
)
from options_backtester.engine.execution import ExecutionConfig, ExecutionModel
from options_backtester.engine.portfolio import (
    GREEKS,
    Portfolio,
    Position,
    margin_per_unit,
    max_loss_per_unit,
)
from options_backtester.regimes.classifier import RegimeClassifier, RegimeState, RegimeTimeline
from options_backtester.strategies import conditions
from options_backtester.strategies.base import (
    Leg,
    LegQuote,
    MarketContext,
    OptionStrategy,
    Rejection,
    quote_legs,
)
from options_backtester.strategies.conditions import ExitReason, PositionState


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    start: date
    end: date
    initial_capital: float = Field(default=100_000.0, gt=0)
    execution: ExecutionConfig = ExecutionConfig()
    #: New entries are refused once margin in use would exceed this share of equity.
    max_margin_utilization: float = Field(default=0.5, gt=0, le=1)
    settlement: Literal["cash", "physical"] = "cash"
    #: Dollars per in-the-money contract when ``settlement`` is physical.
    assignment_fee: float = Field(default=0.0, ge=0)
    american_exercise: bool = True
    #: A short ITM leg with less extrinsic value than this is flagged as assignment risk.
    early_assignment_extrinsic: float = Field(default=0.05, ge=0)
    #: Ex-dividend date -> cash amount; raises the short-call assignment threshold.
    dividends: dict[date, float] = Field(default_factory=dict)
    #: Calendar days of bars loaded before ``start`` to warm up regime indicators.
    regime_warmup_days: int = Field(default=420, ge=0)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    legs: pd.DataFrame
    equity: pd.DataFrame
    rejections: dict[str, int]
    diagnostics: dict[str, Any]
    strategy_config: dict[str, Any]
    backtest_config: dict[str, Any]
    provider: str
    synthetic_data: bool
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)


class Backtester:
    def __init__(
        self,
        provider: DataProvider,
        strategy: OptionStrategy,
        config: BacktestConfig,
        regimes: RegimeTimeline | None = None,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.config = config
        self.execution = ExecutionModel(config.execution)
        self.regimes = regimes
        self.portfolio = Portfolio(config.initial_capital, config.execution.multiplier)
        self._trades: list[dict[str, Any]] = []
        self._legs: list[dict[str, Any]] = []
        self._equity: list[dict[str, Any]] = []
        self._rejections: Counter[str] = Counter()
        self._next_id = 1
        self._last_entry_day: date | None = None
        self._closes = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        cfg = self.config
        bars = self.provider.get_underlying_bars(cfg.symbol, cfg.start, cfg.end)
        if bars.empty:
            raise DataUnavailableError(f"no underlying bars for {cfg.symbol}")
        if self.regimes is None:
            warm = self.provider.get_underlying_bars(
                cfg.symbol, cfg.start - timedelta(days=cfg.regime_warmup_days), cfg.end
            )
            self.regimes = RegimeClassifier().classify(warm)
        days = [market_date(ts) for ts in bars["timestamp"]]
        self._closes = pd.Series(bars["close"].to_numpy(), index=pd.to_datetime(days))

        missing_chain_days = 0
        chain: OptionChainSnapshot | None = None
        now, day, spot = bars["timestamp"].iloc[0], days[0], float(bars["close"].iloc[0])
        for bar_time, day, close in zip(bars["timestamp"], days, bars["close"], strict=True):
            try:
                chain = self.provider.get_option_chain(cfg.symbol, day)
            except DataUnavailableError:
                chain = None
                missing_chain_days += 1
            # A snapshot carries its own known-at time, which may differ from the bar's.
            now = chain.timestamp if chain is not None else bar_time
            spot = chain.underlying_price if chain is not None else float(close)
            regime = self.regimes.at(now)
            self._manage(now, day, spot, chain, regime)
            # Nothing is opened on the final session: it could only be force-closed at once.
            if chain is not None and day != days[-1]:
                self._enter(now, day, chain, regime)
            self._record(now, spot, regime)
        if missing_chain_days == len(bars):
            raise DataUnavailableError(
                f"{self.provider.name}: no option chain on any of {len(bars)} sessions"
            )
        for position in list(self.portfolio.positions.values()):
            self._close(position, now, day, spot, chain, ExitReason.END_OF_DATA)
        if self._equity:
            self._equity[-1].update(self._equity_row(now, spot, self.regimes.at(now)))

        return BacktestResult(
            trades=pd.DataFrame(self._trades),
            legs=pd.DataFrame(self._legs),
            equity=pd.DataFrame(self._equity),
            rejections=dict(self._rejections),
            diagnostics={
                "sessions": len(bars),
                "missing_chain_days": missing_chain_days,
                "skipped_marks": int(sum(t["skipped_marks"] for t in self._trades)),
                "estimated_exits": int(sum(t["exit_estimated"] for t in self._trades)),
            },
            strategy_config=self.strategy.config.model_dump(mode="json"),
            backtest_config=cfg.model_dump(mode="json"),
            provider=self.provider.name,
            synthetic_data=self.provider.synthetic,
            regimes=self.regimes.frame,
        )

    # ------------------------------------------------------------------
    def _manage(
        self,
        now: pd.Timestamp,
        day: date,
        spot: float,
        chain: OptionChainSnapshot | None,
        regime: RegimeState | None,
    ) -> None:
        rules = self.strategy.config.exit
        for position in list(self.portfolio.positions.values()):
            if min(leg.expiration for leg in position.legs) <= day:
                self._close(position, now, day, spot, chain, ExitReason.EXPIRATION)
                continue
            quotes = quote_legs(chain, position.legs) if chain is not None else None
            if quotes is None:
                position.skipped_marks += 1
                continue
            position.mark_quotes = quotes
            position.mark_value = sum(
                leg.side * leg.ratio * q.mid for leg, q in zip(position.legs, quotes, strict=True)
            )
            position.peak_pnl = max(position.peak_pnl, position.mark_pnl)
            if self._assignment_risk(position, quotes, spot, day):
                position.assignment_risk_flags += 1
            shorts = [leg for leg in position.legs if leg.side < 0]
            state = PositionState(
                basis=position.basis,
                pnl=position.mark_pnl,
                peak_pnl=position.peak_pnl,
                min_dte=min((leg.expiration - day).days for leg in position.legs),
                days_held=(day - market_date(position.entry_time)).days,
                underlying_price=spot,
                short_put_strikes=tuple(s.strike for s in shorts if s.option_type == "PUT"),
                short_call_strikes=tuple(s.strike for s in shorts if s.option_type == "CALL"),
                position_delta=_net(position.legs, quotes, "delta"),
                entry_regime=position.entry_regime,
                regime=regime,
            )
            reason = conditions.evaluate_exit(rules, state, self.strategy.config.entry)
            if reason is not None:
                self._close(position, now, day, spot, chain, reason)

    def _assignment_risk(
        self, position: Position, quotes: tuple[LegQuote, ...], spot: float, day: date
    ) -> bool:
        cfg = self.config
        if not cfg.american_exercise:
            return False
        for leg, quote in zip(position.legs, quotes, strict=True):
            intrinsic = leg.intrinsic(spot)
            if leg.side >= 0 or intrinsic <= 0:
                continue
            threshold = cfg.early_assignment_extrinsic
            if leg.option_type == "CALL":
                horizon = min(leg.expiration, day + timedelta(days=5))
                threshold = max(
                    threshold, sum(a for d, a in cfg.dividends.items() if day < d <= horizon)
                )
            if quote.mid - intrinsic < threshold:
                return True
        return False

    # ------------------------------------------------------------------
    def _enter(
        self,
        now: pd.Timestamp,
        day: date,
        chain: OptionChainSnapshot,
        regime: RegimeState | None,
    ) -> None:
        strategy, entry = self.strategy.config, self.strategy.config.entry
        if len(self.portfolio.positions) >= entry.max_open_positions:
            return
        if (
            self._last_entry_day is not None
            and (day - self._last_entry_day).days < entry.min_days_between_entries
        ):
            return
        candidate = self.strategy.generate(MarketContext(now, chain, regime))
        if isinstance(candidate, Rejection):
            self._rejections[candidate.reason] += 1
            return
        fill = self.execution.fill(candidate.legs, candidate.quotes)
        if fill is None:
            self._rejections["NO_FILL"] += 1
            return
        rejection = conditions.check_fill(entry, fill.value)
        if rejection is not None:
            self._rejections[rejection.reason] += 1
            return
        multiplier, quantity = self.config.execution.multiplier, strategy.quantity
        scale = multiplier * quantity
        margin = (
            margin_per_unit(candidate.legs, candidate.quotes, fill.value, chain.underlying_price)
            * scale
        )
        limit = self.config.max_margin_utilization * self.portfolio.equity
        if self.portfolio.margin_used + margin > limit:
            self._rejections["INSUFFICIENT_BUYING_POWER"] += 1
            return
        commission, fees = self.execution.costs(candidate.legs, quantity)
        position = Position(
            trade_id=self._next_id,
            strategy=strategy.name,
            symbol=candidate.symbol,
            legs=candidate.legs,
            quantity=quantity,
            entry_time=now,
            entry_underlying=chain.underlying_price,
            entry_fill=fill,
            entry_quotes=candidate.quotes,
            entry_commission=commission,
            entry_fees=fees,
            margin=margin,
            max_loss=max_loss_per_unit(candidate.legs, fill.value) * scale,
            entry_regime=regime,
        )
        self.portfolio.open(position)
        self._next_id += 1
        self._last_entry_day = day

    # ------------------------------------------------------------------
    def _close(
        self,
        position: Position,
        now: pd.Timestamp,
        day: date,
        spot: float,
        chain: OptionChainSnapshot | None,
        reason: ExitReason,
    ) -> None:
        cfg, multiplier = self.config, self.config.execution.multiplier
        prices: list[float] = []
        mids: list[float] = []
        quotes: list[LegQuote | None] = []
        estimated = False
        traded: list[Leg] = []
        assigned = 0
        for index, leg in enumerate(position.legs):
            if leg.expiration <= day:
                # Settle on the last close at or before expiration (never a later one).
                settle = self._closes.asof(pd.Timestamp(leg.expiration))
                intrinsic = leg.intrinsic(spot if pd.isna(settle) else float(settle))
                prices.append(intrinsic)
                mids.append(intrinsic)
                quotes.append(None)
                assigned += leg.ratio if intrinsic > 0 else 0
                continue
            row = chain.lookup(leg.expiration, leg.strike, leg.option_type) if chain else None
            quote = LegQuote.from_row(row) if row is not None else None
            if quote is None or not quote.tradable:
                # No executable market: fall back to the last complete mark, and say so.
                quote, estimated = position.mark_quotes[index], True
                prices.append(quote.mid)
            else:
                prices.append(self.execution.leg_price(-leg.side, quote))
            mids.append(quote.mid)
            quotes.append(quote)
            traded.append(leg)

        def package(values: list[float]) -> float:
            return sum(
                leg.side * leg.ratio * v for leg, v in zip(position.legs, values, strict=True)
            )

        exit_value, exit_mid = package(prices), package(mids)
        commission, fees = self.execution.costs(tuple(traded), position.quantity)
        if cfg.settlement == "physical":
            fees += assigned * position.quantity * cfg.assignment_fee
        self.portfolio.close(position.trade_id, exit_value, commission, fees)

        scale = multiplier * position.quantity
        gross = (exit_value - position.entry_value) * scale
        total_commission = position.entry_commission + commission
        total_fees = position.entry_fees + fees
        slippage = (position.entry_fill.slippage + (exit_mid - exit_value)) * scale
        regime = position.entry_regime
        entry_day = market_date(position.entry_time)
        self._trades.append(
            {
                "trade_id": position.trade_id,
                "strategy": position.strategy,
                "symbol": position.symbol,
                "entry_time": position.entry_time,
                "exit_time": now,
                "days_in_trade": (day - entry_day).days,
                "quantity": position.quantity,
                "entry_dte": min((leg.expiration - entry_day).days for leg in position.legs),
                "entry_underlying": position.entry_underlying,
                "exit_underlying": spot,
                "entry_value": position.entry_value,
                "entry_mid_value": position.entry_fill.mid_value,
                "exit_value": exit_value,
                "exit_mid_value": exit_mid,
                "exit_trigger_pnl": position.mark_pnl,
                "gross_pnl": gross,
                "commission": total_commission,
                "fees": total_fees,
                "net_pnl": gross - total_commission - total_fees,
                "slippage_paid": slippage,
                "margin": position.margin,
                "max_loss": position.max_loss,
                "exit_reason": reason.value,
                "exit_estimated": estimated,
                "assigned_contracts": assigned * position.quantity
                if cfg.settlement == "physical"
                else 0,
                "skipped_marks": position.skipped_marks,
                "assignment_risk_flags": position.assignment_risk_flags,
                "entry_trend": regime.trend if regime else "UNKNOWN",
                "entry_volatility": regime.volatility if regime else "UNKNOWN",
                "entry_vol_dynamics": regime.vol_dynamics if regime else "UNKNOWN",
                "entry_iv_rank": regime.iv_rank if regime else np.nan,
            }
        )
        for index, leg in enumerate(position.legs):
            entry_quote, exit_quote = position.entry_quotes[index], quotes[index]
            entry_price = position.entry_fill.leg_prices[index]
            row: dict[str, Any] = {
                "trade_id": position.trade_id,
                "leg": index,
                "expiration": leg.expiration,
                "strike": leg.strike,
                "option_type": leg.option_type,
                "side": "LONG" if leg.side > 0 else "SHORT",
                "ratio": leg.ratio,
                "entry_time": position.entry_time,
                "exit_time": now,
                "entry_price": entry_price,
                "exit_price": prices[index],
                "entry_mid": entry_quote.mid,
                "exit_mid": mids[index],
                "entry_slippage": leg.side * (entry_price - entry_quote.mid),
                "exit_slippage": leg.side * (mids[index] - prices[index]),
                "pnl": leg.side * leg.ratio * (prices[index] - entry_price) * scale,
                "settled_at_expiration": exit_quote is None,
                "entry_iv": entry_quote.implied_volatility,
                "exit_iv": exit_quote.implied_volatility if exit_quote else np.nan,
            }
            for name in GREEKS:
                row[f"entry_{name}"] = getattr(entry_quote, name)
                row[f"exit_{name}"] = getattr(exit_quote, name) if exit_quote else np.nan
            self._legs.append(row)

    # ------------------------------------------------------------------
    def _equity_row(
        self, now: pd.Timestamp, spot: float, regime: RegimeState | None
    ) -> dict[str, Any]:
        portfolio = self.portfolio
        return {
            "timestamp": now,
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "margin_used": portfolio.margin_used,
            "open_positions": len(portfolio.positions),
            "underlying_price": spot,
            "trend": regime.trend if regime else "UNKNOWN",
            "volatility": regime.volatility if regime else "UNKNOWN",
            **portfolio.greeks(),
        }

    def _record(self, now: pd.Timestamp, spot: float, regime: RegimeState | None) -> None:
        self._equity.append(self._equity_row(now, spot, regime))


def _net(legs: tuple[Leg, ...], quotes: tuple[LegQuote, ...], name: str) -> float:
    values = [leg.side * leg.ratio * getattr(q, name) for leg, q in zip(legs, quotes, strict=True)]
    return float(np.sum(values))
