"""Provider-neutral historical-data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import pandas as pd


def _date(value: date | str) -> date:
    return pd.Timestamp(value).date()


@dataclass(frozen=True, slots=True)
class HistoricalDataRequest:
    start: date | str
    end: date | str
    interval_minutes: int = 5
    spx_root: str = "SPX"
    vix_root: str = "VIX"
    include_vix: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _date(self.start))
        object.__setattr__(self, "end", _date(self.end))
        if self.start > self.end:
            raise ValueError("historical-data start cannot be after end")
        if (
            isinstance(self.interval_minutes, bool)
            or not isinstance(self.interval_minutes, int)
            or self.interval_minutes <= 0
        ):
            raise ValueError("interval minutes must be a positive integer")
        if not self.spx_root.strip() or not self.vix_root.strip():
            raise ValueError("provider roots cannot be empty")


@dataclass(frozen=True, slots=True)
class HistoricalDataset:
    provider: str
    spx_bars: pd.DataFrame
    option_quotes: pd.DataFrame
    vix: pd.DataFrame | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class HistoricalDataProvider(Protocol):
    name: str

    def download(self, request: HistoricalDataRequest) -> HistoricalDataset: ...


@dataclass(frozen=True, slots=True)
class DownloadArtifacts:
    output_directory: Path
    spx_path: Path
    options_path: Path
    vix_path: Path | None
    manifest_path: Path
