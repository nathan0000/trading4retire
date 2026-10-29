"""Minimal portfolio ledger for completed Phase 7 trade records."""

from __future__ import annotations

from dataclasses import dataclass, field

from spx_0dte_backtest.backtest.position import TradeRecord


@dataclass(slots=True)
class Portfolio:
    """Collect completed records without netting or capital assumptions."""

    _records: list[TradeRecord] = field(default_factory=list, repr=False)

    @property
    def records(self) -> tuple[TradeRecord, ...]:
        return tuple(self._records)

    @property
    def net_pnl(self) -> float:
        return sum(record.net_pnl for record in self._records)

    def add(self, record: TradeRecord) -> None:
        self._records.append(record)
