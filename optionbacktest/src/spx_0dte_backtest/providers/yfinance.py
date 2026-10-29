"""Yahoo Finance intraday SPX adapter for routine underlying-bar collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from spx_0dte_backtest.providers.base import HistoricalDataRequest, HistoricalDataset
from spx_0dte_backtest.providers.thetadata import _empty_options


@dataclass(frozen=True, slots=True)
class YFinanceSettings:
    ticker: str = "^GSPC"
    timezone: str = "America/New_York"
    auto_adjust: bool = False


class YFinanceAdapter:
    """Download recent SPX index bars and emit the canonical provider schema."""

    name = "yfinance"

    def __init__(
        self,
        settings: YFinanceSettings | None = None,
        *,
        download: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        self.settings = settings or YFinanceSettings()
        self._download = download

    def download(self, request: HistoricalDataRequest) -> HistoricalDataset:
        if request.interval_minutes != 1:
            raise ValueError("yfinance adapter supports one-minute downloads only")
        return HistoricalDataset(
            provider=self.name,
            spx_bars=self._canonical_bars(self._download_frame(request)),
            option_quotes=_empty_options(self.settings.timezone),
            vix=None,
            metadata={
                "ticker": self.settings.ticker,
                "interval": "1m",
                "timezone": self.settings.timezone,
                "scope": "UNDERLYING_ONLY",
                "limitations": "Yahoo intraday retention applies; no option quotes are supplied",
            },
        )

    def _download_frame(self, request: HistoricalDataRequest) -> pd.DataFrame:
        download = self._download
        if download is None:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise RuntimeError("install yfinance to use the yfinance provider") from exc
            download = yf.download
        return download(
            self.settings.ticker,
            start=request.start.isoformat(),
            end=(request.end + timedelta(days=1)).isoformat(),
            interval="1m",
            auto_adjust=self.settings.auto_adjust,
            prepost=False,
            progress=False,
            threads=False,
        )

    def _canonical_bars(self, raw: pd.DataFrame) -> pd.DataFrame:
        columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "source"]
        if raw.empty:
            return pd.DataFrame(
                {
                    "timestamp": pd.DatetimeIndex([], tz=self.settings.timezone),
                    **{column: [] for column in columns[1:]},
                }
            )
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.rename(columns={str(column): str(column).lower() for column in frame.columns})
        timestamps = pd.DatetimeIndex(frame.index)
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize("UTC")
        frame.index = timestamps.tz_convert(self.settings.timezone)
        frame = frame.between_time("09:30", "16:00", inclusive="both")
        result = frame.reset_index(names="timestamp")
        result["symbol"] = "SPX"
        result["source"] = "YFINANCE_GSPC_1M"
        missing = {"open", "high", "low", "close", "volume"} - set(result.columns)
        if missing:
            raise ValueError(f"yfinance response missing columns: {sorted(missing)}")
        return (
            result[columns]
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )
