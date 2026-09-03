"""Narrow, provider-isolation ports (CLAUDE.md section 10).

Each port is scoped to one data family so an adapter never needs to
implement a vendor-shaped mega-interface. Core/domain code depends only
on these Protocols; adapters own auth, transport, retries, vendor symbol
mapping, and provenance. Only MarketCalendarPort has a real implementation
in Phase 1 (regime_engine.adapters.providers.fixture_market_calendar); the
others are declared now so later phases don't reshape this module.
"""

from collections.abc import Mapping
from datetime import date, datetime
from typing import Protocol

from regime_engine.domain.enums import MarketSession
from regime_engine.domain.models import Observation, PriceBar, SessionBounds


class PriceBarsPort(Protocol):
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        as_of: datetime,
    ) -> tuple[Observation[PriceBar], ...]: ...


class VolatilityIndexPort(Protocol):
    def get_value(self, symbol: str, as_of: datetime) -> Observation[float]: ...


class OptionsPositioningPort(Protocol):
    def get_positioning(self, symbol: str, as_of: datetime) -> Observation[Mapping[str, float]]: ...

    # Full OptionsPositioningSnapshot contract lands in Phase 8 (CLAUDE.md section 20.8).


class BreadthPort(Protocol):
    def get_breadth(self, as_of: datetime) -> Observation[Mapping[str, float]]: ...


class MarketCalendarPort(Protocol):
    def session_bounds(self, day: date) -> SessionBounds: ...

    def session_for(self, as_of: datetime) -> MarketSession: ...

    def is_trading_day(self, day: date) -> bool: ...

    def next_trading_day(self, day: date) -> date: ...

    def previous_trading_day(self, day: date) -> date: ...
