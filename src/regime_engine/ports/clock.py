"""Clock port: "what time is it right now," distinct from `as_of`.

`as_of` is an explicit parameter threaded through the (future) application
pipeline (CLAUDE.md section 11) so every calculation is reproducible for
an arbitrary historical instant; it is never read from a clock. `Clock`
exists only for infrastructure-edge needs like stamping `generated_at`
on a snapshot, and for deterministic replay/testing via `FixedClock`.
"""

from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    """Live wall-clock time, always timezone-aware UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """Deterministic clock for tests and replay.

    Mutable via `set()` so a single instance can be advanced across a
    replay run without re-wiring dependents, supporting the mission's
    "arbitrary as_of, historical replay, intraday updates" requirement
    (CLAUDE.md section 1).
    """

    def __init__(self, fixed: datetime) -> None:
        self._fixed = self._require_aware(fixed)

    def now(self) -> datetime:
        return self._fixed

    def set(self, new_time: datetime) -> None:
        self._fixed = self._require_aware(new_time)

    @staticmethod
    def _require_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        return value.astimezone(UTC)
