"""Stable, closed vocabularies for the regime taxonomy (CLAUDE.md section 6).

Internal code should prefer these enums over bare strings; serialized
boundaries (JSON, reports) use the plain string values below, which are
part of the public contract and must not be renamed without a schema
version bump (CLAUDE.md section 5.3).
"""

from enum import StrEnum


class Direction(StrEnum):
    """Directional axis of the regime taxonomy."""

    STRONG_BEAR = "STRONG_BEAR"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    BULL = "BULL"
    STRONG_BULL = "STRONG_BULL"
    UNKNOWN = "UNKNOWN"


class Volatility(StrEnum):
    """Volatility-regime axis."""

    COMPRESSED = "COMPRESSED"
    NORMAL = "NORMAL"
    EXPANDING = "EXPANDING"
    STRESS = "STRESS"
    UNKNOWN = "UNKNOWN"


class Gamma(StrEnum):
    """Dealer gamma-positioning axis."""

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"


class Liquidity(StrEnum):
    """Market-liquidity axis."""

    NORMAL = "NORMAL"
    THIN = "THIN"
    DISLOCATED = "DISLOCATED"
    UNKNOWN = "UNKNOWN"


class Stability(StrEnum):
    """Regime-stability axis: how likely the current label is to persist."""

    STABLE = "STABLE"
    TRANSITION_RISK = "TRANSITION_RISK"
    TRANSITIONING = "TRANSITIONING"
    UNKNOWN = "UNKNOWN"


class Quality(StrEnum):
    """Data-quality flag carried on every Observation and FeatureValue."""

    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    MISSING = "MISSING"
    SUSPECT = "SUSPECT"


class MarketSession(StrEnum):
    """Time-of-day session state relative to the exchange calendar."""

    PRE_MARKET = "PRE_MARKET"
    REGULAR = "REGULAR"
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED = "CLOSED"


class CalendarDayStatus(StrEnum):
    """Day-level trading-calendar status, distinct from MarketSession's time-of-day."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EARLY_CLOSE = "EARLY_CLOSE"


class CompositeRegimeLabel(StrEnum):
    """Known composite labels (CLAUDE.md section 6) for internal convenience only.

    RegimeSnapshot.composite_regime is typed `str` (open vocabulary) since
    composite labels are summaries, not a closed set the classifier is
    limited to (CLAUDE.md section 6).
    """

    RANGE_POSITIVE_GAMMA = "RANGE_POSITIVE_GAMMA"
    BULL_TREND_NORMAL_VOL = "BULL_TREND_NORMAL_VOL"
    BEAR_TREND_VOL_EXPANSION = "BEAR_TREND_VOL_EXPANSION"
    HIGH_VOL_MEAN_REVERSION = "HIGH_VOL_MEAN_REVERSION"
    COMPRESSION_BREAKOUT_RISK = "COMPRESSION_BREAKOUT_RISK"
    MARKET_STRESS = "MARKET_STRESS"
    UNKNOWN = "UNKNOWN"
