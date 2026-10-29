from __future__ import annotations

import json
from pathlib import Path

from spx_0dte_backtest import cli
from spx_0dte_backtest.providers import HistoricalDataRequest
from tests.unit.providers.test_registry_and_download import StubProvider


def test_download_cli_writes_provider_dataset(tmp_path: Path, capsys, monkeypatch) -> None:
    stub = StubProvider()
    monkeypatch.setattr(cli, "ThetaDataAdapter", lambda settings: stub)
    output = tmp_path / "download"

    code = cli.main(
        [
            "download-data",
            "--provider",
            "thetadata",
            "--start",
            "2026-08-12",
            "--end",
            "2026-08-12",
            "--output",
            str(output),
            "--json",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["provider"] == "stub"
    assert summary["spx_rows"] == 1
    assert (output / "spx_5min.parquet").exists()
    assert (output / "spx_0dte_options.parquet").exists()
    assert (output / "download_manifest.json").exists()


def test_download_request_used_by_stub_is_compatible() -> None:
    dataset = StubProvider().download(HistoricalDataRequest(start="2026-08-12", end="2026-08-12"))
    assert dataset.provider == "stub"
