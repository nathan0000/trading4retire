from __future__ import annotations

import sqlite3
from pathlib import Path

from spx_0dte_backtest.database import SQLiteStore


def test_initialization_creates_versioned_schema(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteStore(path)
    store.initialize()

    with sqlite3.connect(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert version == 1
    assert {
        "backtest_runs",
        "market_bars",
        "option_quotes",
        "market_features",
        "trade_candidates",
        "trades",
        "trade_legs",
        "daily_summary",
    }.issubset(tables)


def test_read_methods_return_empty_results_for_new_database(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "runs.sqlite")
    store.initialize()
    assert store.list_runs() == []
    assert store.load_trades("missing").empty
