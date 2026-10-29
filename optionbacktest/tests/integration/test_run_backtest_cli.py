from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spx_0dte_backtest.cli import main

ROOT = Path(__file__).parents[2]


def write_market_data(directory: Path) -> tuple[Path, Path]:
    timestamps = pd.date_range(
        "2026-08-12 09:30", "2026-08-12 15:55", freq="5min", tz="America/New_York"
    )
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPX",
            "open": 6000.0,
            "high": 6001.0,
            "low": 5999.0,
            "close": 6000.0,
            "volume": 100,
            "source": "SYNTHETIC_TEST_ONLY",
        }
    )
    spx_path = directory / "spx.csv"
    bars.to_csv(spx_path, index=False)

    rows: list[dict[str, object]] = []
    contracts = [
        (5940, "PUT", -0.04, 0.30, 0.40),
        (5970, "PUT", -0.10, 1.20, 1.40),
        (6030, "CALL", 0.10, 1.30, 1.50),
        (6060, "CALL", 0.04, 0.30, 0.40),
    ]
    for timestamp in (
        pd.Timestamp("2026-08-12 10:45", tz="America/New_York"),
        pd.Timestamp("2026-08-12 15:55", tz="America/New_York"),
    ):
        for strike, option_type, delta, entry_bid, entry_ask in contracts:
            is_exit = timestamp.hour == 15
            bid = 0.10 if is_exit else entry_bid
            ask = 0.20 if is_exit else entry_ask
            rows.append(
                {
                    "timestamp": timestamp,
                    "expiration": "2026-08-12",
                    "strike": strike,
                    "option_type": option_type,
                    "bid": bid,
                    "ask": ask,
                    "last": (bid + ask) / 2,
                    "volume": 10,
                    "open_interest": 100,
                    "delta": delta,
                    "gamma": 0.01,
                    "theta": -0.5,
                    "vega": 0.1,
                    "implied_volatility": 0.2,
                    "contract_symbol": f"{option_type}_{strike}",
                    "source": "SYNTHETIC_TEST_ONLY",
                }
            )
    options_path = directory / "options.csv"
    pd.DataFrame(rows).to_csv(options_path, index=False)
    return spx_path, options_path


def test_run_backtest_cli_executes_complete_workflow(tmp_path: Path, capsys) -> None:
    spx_path, options_path = write_market_data(tmp_path)
    output = tmp_path / "run_001"

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
            "run_001",
            "--json",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["scheduled_entries"] == 19
    assert summary["accepted_candidates"] == 1
    assert summary["rejected_candidates"] == 18
    assert summary["completed_trades"] == 1
    assert summary["rejection_counts"] == {"NO_OPTION_DATA": 18}
    assert (output / "trades.parquet").exists()
    assert (output / "candidates.parquet").exists()
    assert (output / "metrics.json").exists()
    assert (output / "workflow_summary.json").exists()
    assert (output / "charts" / "equity_curve.png").exists()
    trades = pd.read_parquet(output / "trades.parquet")
    assert trades.loc[0, "exit_reason"] == "FORCED_EXIT"
    assert trades.loc[0, "spx_exit"] == 6000.0
    assert trades.loc[0, "factor_score_version"] == "ic_suitability_v1"


def test_run_backtest_cli_preserves_existing_output(tmp_path: Path, capsys) -> None:
    spx_path, options_path = write_market_data(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

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
        ]
    )

    assert code == 2
    assert "not empty" in capsys.readouterr().out
    assert marker.read_text(encoding="utf-8") == "keep"


def test_run_backtest_cli_fails_when_forced_exit_quotes_are_missing(tmp_path: Path, capsys) -> None:
    spx_path, options_path = write_market_data(tmp_path)
    options = pd.read_csv(options_path)
    options = options[~options["timestamp"].str.contains("15:55")]
    options.to_csv(options_path, index=False)

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
            str(tmp_path / "failed"),
        ]
    )

    assert code == 2
    assert "forced-exit" in capsys.readouterr().out.lower()
