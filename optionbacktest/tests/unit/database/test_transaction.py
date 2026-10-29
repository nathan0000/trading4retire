from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spx_0dte_backtest.database import SQLiteStore


def test_failed_persistence_rolls_back_run_header(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "runs.sqlite")
    malformed_bars = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-08-12 09:30", tz="America/New_York")]}
    )
    empty = pd.DataFrame()

    with pytest.raises(AttributeError):
        store.persist_run(
            run_id="rollback_me",
            metadata={
                "strategy": "IRON_CONDOR",
                "data_classification": "SYNTHETIC",
                "code_version": "test",
                "configuration": {},
            },
            output_directory=tmp_path / "output",
            market_bars=malformed_bars,
            option_quotes=empty,
            market_features=empty,
            candidates=[],
            records=[],
            trades=empty,
        )

    assert store.list_runs() == []
