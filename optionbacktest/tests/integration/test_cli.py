from __future__ import annotations

import json
from pathlib import Path

from spx_0dte_backtest.cli import main

ROOT = Path(__file__).parents[2]


def test_cli_validates_example_as_json(capsys) -> None:
    exit_code = main(
        [
            "validate-spx",
            "--input",
            str(ROOT / "data/examples/spx_5min_example.csv"),
            "--config",
            str(ROOT / "config/settings.yaml"),
            "--json",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["accepted_rows"] == 6
    assert output["valid"] is True


def test_cli_returns_nonzero_for_invalid_data(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,open,high,low,close\n,1,2,1,2\n")
    exit_code = main(
        [
            "validate-spx",
            "--input",
            str(path),
            "--config",
            str(ROOT / "config/settings.yaml"),
        ]
    )
    assert exit_code == 1
