"""Data adapters. ``build_provider`` constructs one by name from settings."""

from __future__ import annotations

from typing import Any

from options_backtester.data.base import DataProvider
from options_backtester.data.cache import ParquetCache
from options_backtester.data.greeks_engine import GreeksConfig

PROVIDERS = ("synthetic", "synthetic-yfinance", "yfinance", "thetadata", "ibkr", "schwab")


def build_provider(
    name: str,
    cache: ParquetCache | None = None,
    greeks: GreeksConfig | None = None,
    options: dict[str, Any] | None = None,
) -> DataProvider:
    """Build a provider. ``options`` are that adapter's keyword arguments from settings."""
    opts = dict(options or {})
    if name in ("synthetic", "synthetic-yfinance"):
        from options_backtester.data.providers.synthetic import SyntheticConfig, SyntheticProvider
        from options_backtester.data.providers.yfinance_provider import YFinanceProvider

        source = YFinanceProvider(cache=cache) if name == "synthetic-yfinance" else None
        return SyntheticProvider(source, SyntheticConfig(**opts), cache=cache, greeks=greeks)
    if name == "yfinance":
        from options_backtester.data.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider(cache=cache, greeks=greeks, **opts)
    if name == "thetadata":
        from options_backtester.data.providers.thetadata import ThetaDataProvider

        return ThetaDataProvider(cache=cache, greeks=greeks, **opts)
    if name == "ibkr":
        from options_backtester.data.providers.ibkr import IBKRProvider

        return IBKRProvider(cache=cache, greeks=greeks, **opts)
    if name == "schwab":
        from options_backtester.data.providers.schwab import SchwabAuth, SchwabProvider

        token_path = opts.pop("token_path", "~/.config/options_backtester/schwab_token.json")
        auth = SchwabAuth.from_env(_expand(token_path))
        return SchwabProvider(auth, cache=cache, greeks=greeks, **opts)
    raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")


def _expand(path: str) -> str:
    from pathlib import Path

    return str(Path(path).expanduser())
