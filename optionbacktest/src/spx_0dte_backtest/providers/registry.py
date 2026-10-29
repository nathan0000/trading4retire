"""Runtime provider registry for optional vendor extensions."""

from spx_0dte_backtest.providers.base import HistoricalDataProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, HistoricalDataProvider] = {}

    def register(self, provider: HistoricalDataProvider) -> None:
        name = provider.name.lower()
        if name in self._providers:
            raise ValueError(f"provider already registered: {name}")
        self._providers[name] = provider

    def get(self, name: str) -> HistoricalDataProvider:
        try:
            return self._providers[name.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown historical-data provider: {name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)
