"""Reproducible on-disk report bundles for completed backtests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from spx_0dte_backtest.analysis import SensitivityResult
from spx_0dte_backtest.backtest import calculate_metrics, daily_summary
from spx_0dte_backtest.visualization.charts import ChartGenerator, ChartManifest


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    output_directory: Path
    metrics_path: Path
    trades_path: Path
    daily_pnl_path: Path
    manifest_path: Path
    summary_path: Path
    charts: ChartManifest


class ReportGenerator:
    """Write normalized data, metrics, charts, and an audit manifest."""

    def __init__(self, chart_generator: ChartGenerator | None = None) -> None:
        self.chart_generator = chart_generator or ChartGenerator()

    def generate(
        self,
        data: pd.DataFrame,
        output_directory: Path | str,
        *,
        sensitivity: SensitivityResult | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReportArtifacts:
        output = Path(output_directory)
        supplied_metadata = dict(metadata or {})
        try:
            json.dumps(supplied_metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("report metadata must be JSON serializable") from exc
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"report directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)

        trades_path = output / "trades.parquet"
        daily_path = output / "daily_pnl.parquet"
        metrics_path = output / "metrics.json"
        manifest_path = output / "manifest.json"
        summary_path = output / "summary.md"
        if data.empty:
            (output / "charts").mkdir(exist_ok=True)
            charts = ChartManifest(
                {},
                {name: "no completed trades" for name in ChartGenerator.CHART_NAMES},
            )
        else:
            charts = self.chart_generator.generate_all(
                data,
                output / "charts",
                sensitivity=sensitivity,
            )
        metrics = calculate_metrics(data).as_dict()
        data.to_parquet(trades_path, index=False)
        daily_summary(data).to_parquet(daily_path, index=False)
        metrics_path.write_text(
            json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest = {
            "metadata": supplied_metadata,
            "generated_charts": {
                name: str(path.relative_to(output)) for name, path in charts.generated.items()
            },
            "skipped_charts": dict(charts.skipped),
            "files": {
                "trades": trades_path.name,
                "daily_pnl": daily_path.name,
                "metrics": metrics_path.name,
                "summary": summary_path.name,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        summary_path.write_text(_summary(metrics, charts), encoding="utf-8")
        return ReportArtifacts(
            output_directory=output,
            metrics_path=metrics_path,
            trades_path=trades_path,
            daily_pnl_path=daily_path,
            manifest_path=manifest_path,
            summary_path=summary_path,
            charts=charts,
        )


def _json_safe(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value if not isinstance(value, float) or isfinite(value) else None
        for name, value in values.items()
    }


def _summary(metrics: Mapping[str, Any], charts: ChartManifest) -> str:
    lines = [
        "# Backtest report",
        "",
        f"- Total trades: {metrics['total_trades']}",
        f"- Net P&L: {metrics['net_pnl']:.2f}",
        f"- Win rate: {metrics['win_rate']:.2%}",
        f"- Maximum drawdown: {metrics['max_drawdown']:.2f}",
        "",
        "## Charts",
        "",
    ]
    lines.extend(f"- [{name}](charts/{path.name})" for name, path in charts.generated.items())
    if charts.skipped:
        lines.extend(["", "## Skipped charts", ""])
        lines.extend(f"- {name}: {reason}" for name, reason in charts.skipped.items())
    return "\n".join(lines) + "\n"
