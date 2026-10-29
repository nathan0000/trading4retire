"""Extensible historical market-data provider adapters."""

from spx_0dte_backtest.providers.base import (
    DownloadArtifacts,
    HistoricalDataProvider,
    HistoricalDataRequest,
    HistoricalDataset,
)
from spx_0dte_backtest.providers.download import ProviderDownloader
from spx_0dte_backtest.providers.http import (
    JsonTransport,
    ProviderConnectionError,
    ProviderResponse,
)
from spx_0dte_backtest.providers.registry import ProviderRegistry
from spx_0dte_backtest.providers.thetadata import ThetaDataAdapter, ThetaDataSettings
from spx_0dte_backtest.providers.yfinance import YFinanceAdapter, YFinanceSettings

__all__ = [
    "DownloadArtifacts",
    "HistoricalDataProvider",
    "HistoricalDataRequest",
    "HistoricalDataset",
    "JsonTransport",
    "ProviderConnectionError",
    "ProviderDownloader",
    "ProviderRegistry",
    "ProviderResponse",
    "ThetaDataAdapter",
    "ThetaDataSettings",
    "YFinanceAdapter",
    "YFinanceSettings",
]
