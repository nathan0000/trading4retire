"""Point-in-time-safe exchange session handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd


@dataclass(frozen=True, slots=True)
class SessionBounds:
    session_date: date
    open: pd.Timestamp
    close: pd.Timestamp
    is_early_close: bool


class SessionService:
    """Resolve authoritative XNYS sessions, including holidays and early closes."""

    def __init__(self, calendar_name: str = "XNYS", timezone: str = "America/New_York"):
        self._calendar = xcals.get_calendar(calendar_name)
        self._timezone = ZoneInfo(timezone)

    @property
    def timezone(self) -> ZoneInfo:
        return self._timezone

    def is_session(self, session_date: date) -> bool:
        return bool(self._calendar.is_session(pd.Timestamp(session_date)))

    def bounds(self, session_date: date) -> SessionBounds | None:
        label = pd.Timestamp(session_date)
        if not self._calendar.is_session(label):
            return None
        session_open = self._calendar.session_open(label).tz_convert(self._timezone)
        session_close = self._calendar.session_close(label).tz_convert(self._timezone)
        return SessionBounds(
            session_date=session_date,
            open=session_open,
            close=session_close,
            is_early_close=(session_close.hour, session_close.minute) < (16, 0),
        )

    def containing_session(self, timestamp: datetime | pd.Timestamp) -> SessionBounds | None:
        local = pd.Timestamp(timestamp)
        if local.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        local = local.tz_convert(self._timezone)
        bounds = self.bounds(local.date())
        if bounds is None or not (bounds.open <= local <= bounds.close):
            return None
        return bounds

    def expected_timestamps(
        self,
        session_date: date,
        frequency_minutes: int = 5,
        timestamp_semantics: str = "bar_start",
    ) -> pd.DatetimeIndex:
        bounds = self.bounds(session_date)
        if bounds is None:
            return pd.DatetimeIndex([], tz=self._timezone)
        frequency = pd.Timedelta(f"{frequency_minutes}min")
        if timestamp_semantics == "bar_start":
            first, last = bounds.open, bounds.close - frequency
        elif timestamp_semantics == "bar_end":
            first, last = bounds.open + frequency, bounds.close
        else:
            raise ValueError("timestamp_semantics must be 'bar_start' or 'bar_end'")
        return pd.date_range(first, last, freq=frequency)
