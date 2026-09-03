"""Domain-owned exception vocabulary.

Adapters and application code may raise these; the domain layer itself
only raises them, never catches vendor-specific exceptions (CLAUDE.md
section 2: domain depends on interfaces/models, never vendors).
"""

from datetime import date, datetime


class RegimeEngineError(Exception):
    """Base class for all regime-engine domain errors."""


class PointInTimeViolationError(RegimeEngineError):
    """Raised when an observation's available_time is after the as_of being evaluated.

    CLAUDE.md section 1/5.1: available_time <= as_of is mandatory during
    replay and backtesting; this is the guard that enforces it.
    """

    def __init__(self, observation_name: str, available_time: datetime, as_of: datetime) -> None:
        self.observation_name = observation_name
        self.available_time = available_time
        self.as_of = as_of
        super().__init__(
            f"Observation {observation_name!r} available_time="
            f"{available_time.isoformat()} is after as_of={as_of.isoformat()}"
        )


class CalendarCoverageError(RegimeEngineError):
    """Raised when a trading-calendar query falls outside the adapter's coverage window."""

    def __init__(self, day: date) -> None:
        self.day = day
        super().__init__(f"No calendar coverage for {day.isoformat()}")


class SnapshotValidationError(RegimeEngineError):
    """Cross-field/business validation failures beyond field-level constraints."""


class ConfigurationError(RegimeEngineError):
    """Resolved configuration is invalid, incomplete, or unversioned (CLAUDE.md section 17)."""
