"""Headless deterministic chart generation for completed backtests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spx_0dte_backtest.analysis import SensitivityResult
from spx_0dte_backtest.backtest import (
    delta_analysis,
    entry_time_analysis,
    equity_and_drawdown,
    expected_move_analysis,
)


@dataclass(frozen=True, slots=True)
class ChartSettings:
    dpi: int = 120
    file_format: str = "png"
    figsize: tuple[float, float] = (9.0, 5.0)

    def __post_init__(self) -> None:
        if isinstance(self.dpi, bool) or not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("chart DPI must be a positive integer")
        if self.file_format not in {"png", "svg"}:
            raise ValueError("chart format must be png or svg")
        if len(self.figsize) != 2 or any(value <= 0 for value in self.figsize):
            raise ValueError("chart dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ChartManifest:
    generated: Mapping[str, Path]
    skipped: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated", MappingProxyType(dict(self.generated)))
        object.__setattr__(self, "skipped", MappingProxyType(dict(self.skipped)))


class ChartGenerator:
    """Generate every supported chart whose required historical fields exist."""

    CHART_NAMES = (
        "equity_curve",
        "drawdown_curve",
        "pnl_by_entry_time",
        "pnl_by_vix_regime",
        "pnl_by_delta",
        "pnl_by_expected_move",
        "pnl_distribution",
        "factor_score_performance",
        "monthly_pnl_heatmap",
        "entry_exit_examples",
        "sensitivity_surface",
    )

    CORE_COLUMNS = {
        "entry_timestamp",
        "exit_timestamp",
        "gross_pnl",
        "commission",
        "fees",
        "net_pnl",
    }

    def __init__(self, settings: ChartSettings | None = None) -> None:
        self.settings = settings or ChartSettings()

    def generate_all(
        self,
        data: pd.DataFrame,
        output_directory: Path | str,
        *,
        sensitivity: SensitivityResult | None = None,
    ) -> ChartManifest:
        missing = self.CORE_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("cannot chart an empty backtest")
        frame = data.copy(deep=True)
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        generated: dict[str, Path] = {}
        skipped: dict[str, str] = {}

        self._save_series(
            "equity_curve",
            equity_and_drawdown(frame),
            "timestamp",
            "equity",
            "Cumulative net P&L",
            "Net P&L ($)",
            output,
            generated,
        )
        self._save_series(
            "drawdown_curve",
            equity_and_drawdown(frame),
            "timestamp",
            "drawdown",
            "Peak-to-trough drawdown",
            "Drawdown ($)",
            output,
            generated,
        )
        self._save_bar(
            "pnl_by_entry_time",
            entry_time_analysis(frame),
            "entry_time",
            "average_pnl",
            "Average P&L by entry time",
            output,
            generated,
        )
        self._optional_bar(
            frame,
            "pnl_by_vix_regime",
            "vix_regime",
            "Average P&L by VIX regime",
            output,
            generated,
            skipped,
        )
        self._save_bar(
            "pnl_by_delta",
            delta_analysis(frame),
            "delta_bucket",
            "average_pnl",
            "Average P&L by short-leg delta",
            output,
            generated,
        )
        self._optional_expected_move(frame, output, generated, skipped)
        self._distribution(frame, output, generated)
        self._optional_bar(
            frame,
            "factor_score_performance",
            "factor_score",
            "Average P&L by factor score",
            output,
            generated,
            skipped,
        )
        self._monthly_heatmap(frame, output, generated)
        self._entry_exit(frame, output, generated, skipped)
        self._sensitivity(sensitivity, output, generated, skipped)
        return ChartManifest(generated, skipped)

    def _path(self, output: Path, name: str) -> Path:
        return output / f"{name}.{self.settings.file_format}"

    def _finish(self, figure: plt.Figure, output: Path, name: str) -> Path:
        path = self._path(output, name)
        figure.tight_layout()
        figure.savefig(path, dpi=self.settings.dpi, format=self.settings.file_format)
        plt.close(figure)
        return path

    def _save_series(
        self,
        name: str,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        ylabel: str,
        output: Path,
        generated: dict[str, Path],
    ) -> None:
        figure, axis = plt.subplots(figsize=self.settings.figsize)
        axis.plot(data[x], data[y], color="#225ea8", linewidth=1.8)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(title=title, xlabel="Time", ylabel=ylabel)
        generated[name] = self._finish(figure, output, name)

    def _save_bar(
        self,
        name: str,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        output: Path,
        generated: dict[str, Path],
    ) -> None:
        figure, axis = plt.subplots(figsize=self.settings.figsize)
        labels = data[x].astype(str)
        values = data[y].fillna(0)
        axis.bar(labels, values, color=np.where(values >= 0, "#2ca25f", "#de2d26"))
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(title=title, xlabel=x.replace("_", " ").title(), ylabel="Average net P&L ($)")
        axis.tick_params(axis="x", rotation=45)
        generated[name] = self._finish(figure, output, name)

    def _optional_bar(
        self,
        frame: pd.DataFrame,
        name: str,
        column: str,
        title: str,
        output: Path,
        generated: dict[str, Path],
        skipped: dict[str, str],
    ) -> None:
        if column not in frame or frame[column].dropna().empty:
            skipped[name] = f"missing usable column: {column}"
            return
        grouped = (
            frame.dropna(subset=[column])
            .groupby(column, observed=True, sort=True)["net_pnl"]
            .mean()
            .rename("average_pnl")
            .reset_index()
        )
        self._save_bar(name, grouped, column, "average_pnl", title, output, generated)

    def _optional_expected_move(
        self,
        frame: pd.DataFrame,
        output: Path,
        generated: dict[str, Path],
        skipped: dict[str, str],
    ) -> None:
        name = "pnl_by_expected_move"
        column = "expected_move_distance"
        if column not in frame or frame[column].dropna().empty:
            skipped[name] = f"missing usable column: {column}"
            return
        self._save_bar(
            name,
            expected_move_analysis(frame),
            "expected_move_bucket",
            "average_pnl",
            "Average P&L by expected-move distance",
            output,
            generated,
        )

    def _distribution(self, frame: pd.DataFrame, output: Path, generated: dict[str, Path]) -> None:
        name = "pnl_distribution"
        figure, axis = plt.subplots(figsize=self.settings.figsize)
        axis.hist(frame["net_pnl"].astype(float), bins="auto", color="#756bb1", edgecolor="white")
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set(title="Trade P&L distribution", xlabel="Net P&L ($)", ylabel="Trades")
        generated[name] = self._finish(figure, output, name)

    def _monthly_heatmap(
        self, frame: pd.DataFrame, output: Path, generated: dict[str, Path]
    ) -> None:
        name = "monthly_pnl_heatmap"
        local = frame["exit_timestamp"].dt.tz_convert("America/New_York")
        working = frame.assign(year=local.dt.year, month=local.dt.month)
        pivot = working.pivot_table(
            index="year", columns="month", values="net_pnl", aggfunc="sum", fill_value=0
        ).reindex(columns=range(1, 13), fill_value=0)
        figure, axis = plt.subplots(figsize=self.settings.figsize)
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
        axis.set(
            title="Monthly net P&L",
            xlabel="Month",
            ylabel="Year",
            xticks=range(12),
            xticklabels=range(1, 13),
            yticks=range(len(pivot.index)),
            yticklabels=pivot.index,
        )
        figure.colorbar(image, ax=axis, label="Net P&L ($)")
        generated[name] = self._finish(figure, output, name)

    def _entry_exit(
        self,
        frame: pd.DataFrame,
        output: Path,
        generated: dict[str, Path],
        skipped: dict[str, str],
    ) -> None:
        name = "entry_exit_examples"
        required = {"spx_entry", "spx_exit"}
        missing = required.difference(frame.columns)
        if missing or frame[list(required)].dropna().empty:
            skipped[name] = f"missing usable columns: {sorted(missing or required)}"
            return
        examples = frame.dropna(subset=list(required)).head(10)
        figure, axis = plt.subplots(figsize=self.settings.figsize)
        for position, (_, trade) in enumerate(examples.iterrows()):
            axis.plot([0, 1], [trade["spx_entry"], trade["spx_exit"]], marker="o", alpha=0.7)
            axis.text(1.02, trade["spx_exit"], str(position + 1), fontsize=7)
        axis.set(
            title="SPX entry and exit examples",
            xlabel="Trade lifecycle",
            ylabel="SPX",
            xticks=[0, 1],
            xticklabels=["Entry", "Exit"],
        )
        generated[name] = self._finish(figure, output, name)

    def _sensitivity(
        self,
        sensitivity: SensitivityResult | None,
        output: Path,
        generated: dict[str, Path],
        skipped: dict[str, str],
    ) -> None:
        name = "sensitivity_surface"
        if sensitivity is None:
            skipped[name] = "no sensitivity result supplied"
            return
        surface = sensitivity.surface
        parameters = tuple(sensitivity.best_parameters)
        figure, axis = plt.subplots(figsize=self.settings.figsize)
        if len(parameters) == 1:
            axis.plot(surface[parameters[0]], surface[sensitivity.objective], marker="o")
            axis.set(xlabel=parameters[0], ylabel=sensitivity.objective)
        else:
            pivot = surface.pivot(
                index=parameters[0], columns=parameters[1], values=sensitivity.objective
            )
            image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
            axis.set(
                xlabel=parameters[1],
                ylabel=parameters[0],
                xticks=range(len(pivot.columns)),
                xticklabels=pivot.columns,
                yticks=range(len(pivot.index)),
                yticklabels=pivot.index,
            )
            figure.colorbar(image, ax=axis, label=sensitivity.objective)
        axis.set_title("Training sensitivity surface")
        generated[name] = self._finish(figure, output, name)
