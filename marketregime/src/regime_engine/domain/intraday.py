"""Contracts for the intraday SPX regime gate: a five-indicator, single-session view.

This is a narrower, faster product than `RegimeSnapshot` (CLAUDE.md section 5.3):
it reads only today's regular-session bars and answers one question for an
intraday trend-following consumer, "is a directional entry structurally
supported right now?". It does not replace the multi-axis snapshot.

Every enum carries an `UNKNOWN` member beyond the original spec so that
warm-up, missing volume, and bad input surface explicitly instead of being
coerced into a neutral label (CLAUDE.md section 2.6).
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PrimaryRegime(StrEnum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    WHIPSAW_NEUTRAL = "WHIPSAW_NEUTRAL"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class VwapAlignment(StrEnum):
    ABOVE_VWAP = "ABOVE_VWAP"
    BELOW_VWAP = "BELOW_VWAP"
    UNKNOWN = "UNKNOWN"


class VolatilityState(StrEnum):
    NORMAL = "NORMAL"
    EXPANDING = "EXPANDING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class IntradayRegimeConfig(BaseModel):
    """Validated thresholds for the intraday gate (CLAUDE.md section 2.9).

    Timestamps in the input are interpreted per `timestamp_semantics`: with
    `bar_start` (the IBKR convention) the bar stamped 10:44 covers 10:44-10:45
    and its close is first known at 10:45.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_version: str = Field(default="intraday-gate-0.1.0", min_length=1)
    market_timezone: str = "America/New_York"
    session_open: time = time(9, 30)
    session_close: time = time(16, 0)
    bar_minutes: Literal[1, 5] = 1
    timestamp_semantics: Literal["bar_start", "bar_end"] = "bar_start"

    ema_fast: int = Field(default=20, ge=1)
    ema_slow: int = Field(default=40, ge=2)

    rsi_period: int = Field(default=14, ge=1)
    rsi_overbought: float = Field(default=70.0, gt=0, lt=100)
    rsi_oversold: float = Field(default=30.0, gt=0, lt=100)

    atr_period: int = Field(default=14, ge=1)
    # Baseline = mean true range of the bars in [open, open + baseline_minutes).
    baseline_minutes: int = Field(default=30, ge=1)
    expanding_ratio: float = Field(default=1.2, gt=0)
    critical_ratio: float = Field(default=1.5, gt=0)

    # Column holding the volume that weights VWAP. Cash SPX usually has none, so point
    # this at e.g. "spy_volume" explicitly; it is never substituted automatically.
    volume_column: str = "volume"
    proxy_columns: tuple[str, ...] = ("spy_close", "qqq_close")
    # A proxy "heavily" opposes SPX when |EMA fast - EMA slow| / EMA slow >= this, in bp.
    proxy_conflict_bps: float = Field(default=3.0, ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "IntradayRegimeConfig":
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be shorter than ema_slow")
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold must be below rsi_overbought")
        if self.expanding_ratio >= self.critical_ratio:
            raise ValueError("expanding_ratio must be below critical_ratio")
        if self.baseline_minutes % self.bar_minutes:
            raise ValueError("baseline_minutes must be a multiple of bar_minutes")
        if self.session_open >= self.session_close:
            raise ValueError("session_open must be before session_close")
        return self


@dataclass(frozen=True, slots=True)
class RegimeSnapshotPayload:
    """Immutable result of one evaluation; safe to share across threads.

    The first eight fields are the consumer contract. The rest are the indicator
    values and reasons behind the label (CLAUDE.md section 2.5); a field is None
    when its indicator has not warmed up or its input is unusable.
    """

    timestamp: datetime  # the evaluation as_of (UTC), not wall-clock time
    spx_spot: float | None  # close of the last bar completed at `timestamp`
    primary_regime: PrimaryRegime
    ema_gate_open: bool
    vwap_alignment: VwapAlignment
    rsi_status: float | None
    volatility_state: VolatilityState
    internal_bias_match: bool

    ema_fast: float | None = None
    ema_slow: float | None = None
    vwap: float | None = None
    atr: float | None = None
    atr_baseline: float | None = None
    volatility_ratio: float | None = None
    proxy_bias: tuple[tuple[str, float], ...] = ()  # (column, EMA spread in bp)
    bars_used: int = 0
    last_bar_close_time: datetime | None = None
    reasons: tuple[str, ...] = ()
    config_version: str = ""
