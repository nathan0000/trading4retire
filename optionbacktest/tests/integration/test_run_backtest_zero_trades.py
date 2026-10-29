from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spx_0dte_backtest.cli import main
from tests.integration.test_run_backtest_cli import ROOT, write_market_data


def test_valid_run_with_no_entry_snapshot_reports_zero_trades(tmp_path: Path, capsys) -> None:
    spx_path, options_path = write_market_data(tmp_path)
    options = pd.read_csv(options_path)
    options = options[options["timestamp"].str.contains("15:55")]
    options.to_csv(options_path, index=False)
    output = tmp_path / "no_trades"

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
            "--json",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["completed_trades"] == 0
    assert json.loads((output / "metrics.json").read_text())["total_trades"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["generated_charts"] == {}
    assert manifest["skipped_charts"]["equity_curve"] == "no completed trades"
