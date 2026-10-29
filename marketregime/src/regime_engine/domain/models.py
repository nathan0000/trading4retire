"""Immutable, typed canonical domain contracts (CLAUDE.md section 5).

Pydantic v2 frozen models sit on the serialized boundary and provide
runtime validation (score bounds, timezone-awareness) plus JSON Schema
generation. Core calculations elsewhere in the codebase stay ordinary
pure functions over these values (CLAUDE.md section 5) — nothing in this
module performs I/O, reads the clock, or reads environment variables.

Known limitation: `frozen=True` blocks attribute *reassignment*, not
in-place mutation of nested `dict` contents (`features`, `data_quality`,
`metadata`). Sequence fields therefore use `tuple`, which is genuinely
immutable; the two dict fields are left as plain `dict` for JSON-schema
friendliness. See docs/decisions/0001-pydantic-for-domain-contracts.md.
"""

from datetime import UTC, date, datetime, time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from regime_engine.domain.enums import (
    CalendarDayStatus,
    Direction,
    Gamma,
    Liquidity,
    MarketSession,
    Quality,
    Stability,
    Volatility,
)
from regime_engine.domain.errors import PointInTimeViolationError

T = TypeVar("T")


def _as_utc(value: datetime) -> datetime:
    """Require a timezone-aware datetime and normalize it to UTC."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


class Observation(BaseModel, Generic[T]):  # noqa: UP046 (PEP 695 generics not yet used with pydantic)
    """A single sourced value with full provenance (CLAUDE.md section 5.1).

    `available_time` is the earliest time the system could have known the
    value — it, not `received_time`, is what point-in-time filtering must
    key on. `available_time <= as_of` is mandatory during replay and
    backtesting; see `ensure_point_in_time` below.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    value: T | None
    event_time: datetime
    available_time: datetime
    received_time: datetime
    source: str
    source_symbol: str
    quality: Quality
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_time", "available_time", "received_time", mode="after")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)


def ensure_point_in_time(observation: "Observation[Any]", as_of: datetime) -> None:
    """Raise PointInTimeViolationError if the observation was not yet available at as_of.

    Pure, side-effect-free primitive. Bulk enforcement across a canonical
    dataset belongs in `regime_engine.data.validation` (Phase 2); this is
    the building block it will call.
    """
    if observation.available_time > as_of:
        raise PointInTimeViolationError(observation.name, observation.available_time, as_of)


class FeatureValue(BaseModel):
    """A calculated feature output (CLAUDE.md section 5.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float | str | bool | None
    unit: str | None = None
    window: str
    as_of: datetime
    quality: Quality
    inputs: tuple[str, ...] = ()
    calculation_version: str

    @field_validator("as_of", mode="after")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)


class PriceBar(BaseModel):
    """A single OHLCV bar, prior to feature calculation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    interval: str
    session: MarketSession
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int | None = None

    @field_validator("timestamp", mode="after")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def _check_ohlc_bounds(self) -> "PriceBar":
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError("OHLC inconsistent: low/high must bound open and close")
        return self


class SessionBounds(BaseModel):
    """Resolved trading-session boundaries for one calendar day, in exchange-local time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    status: CalendarDayStatus
    pre_market_open: time | None
    regular_open: time | None
    regular_close: time | None
    after_hours_close: time | None
    label: str | None = None


class RegimeSnapshot(BaseModel):
    """The one canonical, versioned regime-assessment result (CLAUDE.md section 5.3).

    Every consumer (reports, API, strategy engine) reads projections of
    this single contract; nothing downstream reruns inference from it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "0.1.0"
    snapshot_id: str = Field(min_length=1)
    as_of: datetime
    generated_at: datetime
    market_session: MarketSession
    universe: str = Field(min_length=1)

    direction: Direction
    volatility: Volatility
    liquidity: Liquidity
    gamma: Gamma
    stability: Stability = Stability.UNKNOWN  # peer axis per section 6, beyond the 5.3 minimum
    composite_regime: str = Field(min_length=1)

    trend_score: float | None = Field(default=None, ge=-100, le=100)
    volatility_score: float | None = Field(default=None, ge=0, le=100)
    cycle_score: float | None = Field(default=None, ge=-100, le=100)
    cycle_confidence: float | None = Field(default=None, ge=0, le=1)
    cycle_horizon: str | None = None
    transition_probability: float | None = Field(default=None, ge=0, le=1)
    transition_target: str | None = None
    confidence: float = Field(ge=0, le=1)

    features: dict[str, FeatureValue] = Field(default_factory=dict)
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    data_quality: dict[str, object] = Field(default_factory=dict)
    policy_hints: tuple[str, ...] = ()

    config_version: str = Field(min_length=1)
    classifier_version: str = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=1)

    @field_validator("as_of", "generated_at", mode="after")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)
