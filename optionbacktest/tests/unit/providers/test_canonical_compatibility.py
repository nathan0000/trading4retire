from __future__ import annotations

from pathlib import Path

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.data.loaders import OptionChainLoader, SpxBarLoader
from spx_0dte_backtest.providers import (
    HistoricalDataRequest,
    ProviderDownloader,
    ThetaDataAdapter,
    ThetaDataSettings,
)
from tests.unit.providers.test_thetadata import FakeTransport, index_response, option_response


def test_thetadata_files_feed_existing_validated_loaders(tmp_path: Path) -> None:
    dataset = ThetaDataAdapter(
        ThetaDataSettings(), transport=FakeTransport([index_response(), option_response()])
    ).download(HistoricalDataRequest(start="2026-08-12", end="2026-08-12"))
    artifacts = ProviderDownloader().write(
        dataset, HistoricalDataRequest(start="2026-08-12", end="2026-08-12"), tmp_path / "download"
    )

    spx = SpxBarLoader(AppConfig()).load(artifacts.spx_path)
    options = OptionChainLoader(AppConfig()).load(artifacts.options_path)
    assert spx.validation.is_valid
    assert options.validation.is_valid
    assert len(spx.bars) == 1
    assert len(options.quotes) == 1
