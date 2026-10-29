from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from spx_0dte_backtest.providers import (
    HistoricalDataRequest,
    HistoricalDataset,
    ProviderDownloader,
    ProviderRegistry,
)


class StubProvider:
    name = "stub"

    def download(self, request: HistoricalDataRequest) -> HistoricalDataset:
        timestamp = pd.Timestamp(request.start, tz="America/New_York")
        return HistoricalDataset(
            provider=self.name,
            spx_bars=pd.DataFrame(
                {
                    "timestamp": [timestamp],
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [0],
                    "symbol": ["SPX"],
                    "source": ["STUB"],
                }
            ),
            option_quotes=pd.DataFrame(
                {
                    "timestamp": pd.DatetimeIndex([], tz="America/New_York"),
                    "expiration": [],
                    "strike": [],
                    "option_type": [],
                    "bid": [],
                    "ask": [],
                    "last": [],
                    "volume": [],
                    "open_interest": [],
                    "delta": [],
                    "gamma": [],
                    "theta": [],
                    "vega": [],
                    "implied_volatility": [],
                    "contract_symbol": [],
                    "source": [],
                }
            ),
            vix=None,
            metadata={"test": True},
        )


def test_registry_allows_future_provider_extensions() -> None:
    registry = ProviderRegistry()
    registry.register(StubProvider())
    assert registry.get("stub").name == "stub"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(StubProvider())


def test_downloader_writes_canonical_parquet_and_manifest(tmp_path: Path) -> None:
    request = HistoricalDataRequest(start=date(2026, 8, 12), end=date(2026, 8, 12))
    artifacts = ProviderDownloader().write(StubProvider().download(request), request, tmp_path)
    assert artifacts.spx_path.exists()
    assert artifacts.options_path.exists()
    assert artifacts.vix_path is None
    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest["provider"] == "stub"
    assert manifest["request"]["start"] == "2026-08-12"
    assert manifest["row_counts"] == {"spx": 1, "options": 0, "vix": 0}


def test_downloader_preserves_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    request = HistoricalDataRequest(start="2026-08-12", end="2026-08-12")
    with pytest.raises(FileExistsError):
        ProviderDownloader().write(StubProvider().download(request), request, output)
