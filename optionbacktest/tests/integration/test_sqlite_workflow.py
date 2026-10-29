from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from spx_0dte_backtest.cli import main
from spx_0dte_backtest.database import SQLiteStore
from tests.integration.test_run_backtest_cli import ROOT, write_market_data


def run_with_database(
    tmp_path: Path,
    capsys,
    *,
    run_id: str = "sqlite_run_001",
    output_name: str = "run_001",
) -> tuple[int, Path, Path]:
    spx_path, options_path = write_market_data(tmp_path)
    output = tmp_path / output_name
    database = tmp_path / "research.sqlite"
    code = main(
        [
            "run-backtest",
            "--spx",
            str(spx_path),
            "--options",
            str(options_path),
            "--config",
            str(ROOT / "config/settings.yaml"),
            "--output",
            str(output),
            "--run-id",
            run_id,
            "--database",
            str(database),
            "--json",
        ]
    )
    capsys.readouterr()
    return code, output, database


def test_cli_persists_complete_run_atomically(tmp_path: Path, capsys) -> None:
    code, output, database = run_with_database(tmp_path, capsys)
    assert code == 0

    with sqlite3.connect(database) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "backtest_runs",
                "market_bars",
                "option_quotes",
                "market_features",
                "trade_candidates",
                "trades",
                "trade_legs",
                "daily_summary",
            )
        }
        run = connection.execute(
            "SELECT status, metadata_json FROM backtest_runs WHERE run_id = ?",
            ("sqlite_run_001",),
        ).fetchone()
        leg = connection.execute(
            "SELECT entry_quote_json, exit_quote_json FROM trade_legs LIMIT 1"
        ).fetchone()
    assert counts == {
        "backtest_runs": 1,
        "market_bars": 78,
        "option_quotes": 8,
        "market_features": 78,
        "trade_candidates": 19,
        "trades": 1,
        "trade_legs": 4,
        "daily_summary": 1,
    }
    assert run[0] == "COMPLETED"
    assert json.loads(run[1])["data_classification"] == "SYNTHETIC"
    assert json.loads(leg[0])["timestamp"].endswith("-04:00")
    assert json.loads(leg[1])["timestamp"].endswith("-04:00")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["files"]["database"] == str(database)


def test_duplicate_run_id_rolls_back_second_transaction(tmp_path: Path, capsys) -> None:
    first, _, database = run_with_database(tmp_path, capsys)
    assert first == 0
    second, _, _ = run_with_database(
        tmp_path,
        capsys,
        run_id="sqlite_run_001",
        output_name="run_002",
    )
    assert second == 2
    store = SQLiteStore(database)
    assert len(store.list_runs()) == 1
    assert len(store.load_trades("sqlite_run_001")) == 1
