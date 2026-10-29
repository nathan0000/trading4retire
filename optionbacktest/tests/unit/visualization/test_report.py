from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from spx_0dte_backtest.visualization import ReportGenerator
from tests.unit.visualization.test_charts import sensitivity, trades


def test_report_bundle_contains_data_metrics_charts_and_metadata(tmp_path: Path) -> None:
    output = tmp_path / "run_001"
    artifacts = ReportGenerator().generate(
        trades(),
        output,
        sensitivity=sensitivity(),
        metadata={"run_id": "run_001", "code_version": "test"},
    )

    assert artifacts.output_directory == output
    assert artifacts.metrics_path.exists()
    assert artifacts.trades_path.exists()
    assert artifacts.daily_pnl_path.exists()
    assert artifacts.manifest_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.charts.generated
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metrics["total_trades"] == 8
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["run_id"] == "run_001"
    pd.testing.assert_frame_equal(pd.read_parquet(artifacts.trades_path), trades())


def test_report_rejects_existing_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ReportGenerator().generate(trades(), output)
    assert (output / "user-file.txt").read_text(encoding="utf-8") == "preserve"


def test_metadata_must_be_json_serializable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        ReportGenerator().generate(trades(), tmp_path / "bad", metadata={"bad": object()})
