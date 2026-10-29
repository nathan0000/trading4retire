"""Provider-neutral loader contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from spx_0dte_backtest.data.validators import ValidationReport


@dataclass(frozen=True, slots=True)
class LoadResult:
    bars: pd.DataFrame
    validation: ValidationReport
    source_path: Path


class MarketBarLoader(Protocol):
    def load(
        self,
        source: Path | str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> LoadResult: ...
