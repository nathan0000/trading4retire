"""Local Parquet cache: ``{root}/{provider}/{symbol}/{date}.parquet``.

Chains are stored one file per market date. Bars are stored in one file per
symbol together with a small JSON sidecar recording the date range that has
been fetched, so a covered request never reaches the network again.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from options_backtester.data.base import MARKET_TZ


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.^-]", "_", name)


class ParquetCache:
    def __init__(self, root: str | Path = "data_cache") -> None:
        self.root = Path(root)

    def _dir(self, provider: str, symbol: str) -> Path:
        return self.root / _safe(provider) / _safe(symbol)

    def chain_path(self, provider: str, symbol: str, day: date) -> Path:
        return self._dir(provider, symbol) / f"{day.isoformat()}.parquet"

    def read_chain(self, provider: str, symbol: str, day: date) -> pd.DataFrame | None:
        path = self.chain_path(provider, symbol, day)
        return pd.read_parquet(path) if path.exists() else None

    def write_chain(self, provider: str, symbol: str, day: date, frame: pd.DataFrame) -> Path:
        path = self.chain_path(provider, symbol, day)
        _atomic_parquet(frame, path)
        return path

    def cached_chain_dates(self, provider: str, symbol: str) -> list[date]:
        folder = self._dir(provider, symbol)
        if not folder.exists():
            return []
        days = []
        for path in folder.glob("*.parquet"):
            try:
                days.append(date.fromisoformat(path.stem))
            except ValueError:
                continue
        return sorted(days)

    def _bars_paths(self, provider: str, symbol: str) -> tuple[Path, Path]:
        folder = self._dir(provider, symbol)
        return folder / "bars.parquet", folder / "bars.json"

    def bars_coverage(self, provider: str, symbol: str) -> tuple[date, date] | None:
        data_path, meta_path = self._bars_paths(provider, symbol)
        if not (data_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        return date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])

    def read_bars(self, provider: str, symbol: str, start: date, end: date) -> pd.DataFrame | None:
        covered = self.bars_coverage(provider, symbol)
        if covered is None or start < covered[0] or end > covered[1]:
            return None
        bars = pd.read_parquet(self._bars_paths(provider, symbol)[0])
        days = bars["timestamp"].dt.tz_convert(MARKET_TZ).dt.date
        return bars[(days >= start) & (days <= end)].reset_index(drop=True)

    def write_bars(
        self, provider: str, symbol: str, bars: pd.DataFrame, start: date, end: date
    ) -> None:
        data_path, meta_path = self._bars_paths(provider, symbol)
        # Never record today or the future as covered: that session is not complete.
        yesterday = pd.Timestamp.now(tz=MARKET_TZ).date() - timedelta(days=1)
        covered_end = min(end, yesterday)
        if covered_end < start:
            return
        _atomic_parquet(bars, data_path)
        meta_path.write_text(
            json.dumps({"start": start.isoformat(), "end": covered_end.isoformat()})
        )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
