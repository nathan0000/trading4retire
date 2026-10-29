from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from spx_0dte_backtest.database import SQLiteStore


def test_newer_schema_version_is_not_downgraded(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")
    with pytest.raises(RuntimeError, match="newer than supported"):
        SQLiteStore(path).initialize()
