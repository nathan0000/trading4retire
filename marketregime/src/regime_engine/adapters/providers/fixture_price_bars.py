"""Fixture PriceBarsPort adapter for tests and replay — no live/paid data service.

Reads daily OHLCV rows from a CSV and wraps each as an `Observation[PriceBar]`
with provenance: `event_time` is the bar's session close in America/New_York,
`available_time` adds a configurable publish lag, and `get_bars` filters out
any bar not yet available at the given `as_of` — the concrete demonstration
of the point-in-time rule (CLAUDE.md section 1) this fixture exists to prove.
"""

import csv
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from regime_engine.domain.enums import MarketSession, Quality
from regime_engine.domain.models import Observation, PriceBar

_NY_TZ = ZoneInfo("America/New_York")
_SESSION_CLOSE = time(16, 0)
_DEFAULT_PUBLISH_LAG = timedelta(minutes=5)


class FixturePriceBarsAdapter:
    """PriceBarsPort implementation backed by a daily-bar CSV fixture."""

    def __init__(self, csv_path: Path, publish_lag: timedelta = _DEFAULT_PUBLISH_LAG) -> None:
        self._path = csv_path
        self._publish_lag = publish_lag

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        as_of: datetime,
    ) -> tuple[Observation[PriceBar], ...]:
        observations = [
            observation
            for observation in self._load()
            if observation.value is not None
            and observation.value.symbol == symbol
            and observation.value.interval == interval
            and start <= observation.event_time <= end
            and observation.available_time <= as_of
        ]
        return tuple(sorted(observations, key=lambda observation: observation.event_time))

    def _load(self) -> list[Observation[PriceBar]]:
        observations: list[Observation[PriceBar]] = []
        with self._path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                bar_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                event_time = datetime.combine(bar_date, _SESSION_CLOSE, tzinfo=_NY_TZ).astimezone(
                    UTC
                )
                available_time = event_time + self._publish_lag
                bar = PriceBar(
                    symbol=row["symbol"],
                    interval="1d",
                    session=MarketSession.REGULAR,
                    timestamp=event_time,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]) if row["volume"] else None,
                )
                observations.append(
                    Observation[PriceBar](
                        name=f"{row['symbol']}_daily_bar",
                        value=bar,
                        event_time=event_time,
                        available_time=available_time,
                        received_time=available_time,
                        source="fixture",
                        source_symbol=row["symbol"],
                        quality=Quality.GOOD,
                        metadata={},
                    )
                )
        return observations
