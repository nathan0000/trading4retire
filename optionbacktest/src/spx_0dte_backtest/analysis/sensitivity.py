"""Bounded one- and two-dimensional training-only sensitivity analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any

import pandas as pd

from spx_0dte_backtest.analysis.walk_forward import FixedSamples

MetricEvaluator = Callable[[pd.DataFrame, Mapping[str, Any]], Mapping[str, Real]]


class ParameterGrid:
    """An ordered grid containing exactly one or two parameter dimensions."""

    def __init__(self, dimensions: Mapping[str, Sequence[Any]]) -> None:
        if not 1 <= len(dimensions) <= 2:
            raise ValueError("parameter grid must contain one or two dimensions")
        normalized: dict[str, tuple[Any, ...]] = {}
        for name, values in dimensions.items():
            if not name:
                raise ValueError("parameter names cannot be empty")
            points = tuple(values)
            if not points:
                raise ValueError(f"parameter {name} must contain at least one value")
            if any(
                left == right for index, left in enumerate(points) for right in points[index + 1 :]
            ):
                raise ValueError(f"parameter {name} contains duplicate values")
            normalized[name] = points
        self._dimensions = MappingProxyType(normalized)

    @property
    def dimensions(self) -> Mapping[str, tuple[Any, ...]]:
        return self._dimensions

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._dimensions)


class SensitivitySettings:
    """Objective and local-isolation rule for a complete sensitivity surface."""

    def __init__(
        self,
        objective: str = "expectancy",
        maximize: bool = True,
        isolation_threshold: float = 0.25,
    ) -> None:
        if not objective:
            raise ValueError("objective cannot be empty")
        if not isfinite(isolation_threshold) or not 0 <= isolation_threshold <= 1:
            raise ValueError("isolation threshold must be finite and between zero and one")
        self.objective = objective
        self.maximize = maximize
        self.isolation_threshold = float(isolation_threshold)


class SensitivityResult:
    """Immutable selection metadata with a defensively copied full surface."""

    def __init__(
        self,
        surface: pd.DataFrame,
        best_parameters: Mapping[str, Any],
        best_objective: float,
        possible_overfit: bool,
        objective: str,
    ) -> None:
        self._surface = surface.copy(deep=True)
        self.best_parameters = MappingProxyType(dict(best_parameters))
        self.best_objective = float(best_objective)
        self.possible_overfit = bool(possible_overfit)
        self.objective = objective

    @property
    def surface(self) -> pd.DataFrame:
        return self._surface.copy(deep=True)


class SensitivityAnalyzer:
    """Evaluate a bounded grid using only the fixed split's training sample."""

    def __init__(self, settings: SensitivitySettings | None = None) -> None:
        self.settings = settings or SensitivitySettings()

    def analyze(
        self,
        samples: FixedSamples,
        grid: ParameterGrid,
        evaluator: MetricEvaluator,
    ) -> SensitivityResult:
        return self.analyze_training(samples.training, grid, evaluator)

    def analyze_training(
        self,
        training: pd.DataFrame,
        grid: ParameterGrid,
        evaluator: MetricEvaluator,
    ) -> SensitivityResult:
        """Analyze an explicitly supplied training window, including walk-forward fits."""

        if training.empty:
            raise ValueError("training sample cannot be empty")
        names = grid.names
        value_axes = tuple(grid.dimensions[name] for name in names)
        index_axes = tuple(range(len(values)) for values in value_axes)
        combinations = tuple(product(*value_axes))
        coordinates = tuple(product(*index_axes))
        rows: list[dict[str, Any]] = []
        metric_names: tuple[str, ...] | None = None

        for values in combinations:
            parameters = dict(zip(names, values, strict=True))
            metrics = self._metrics(
                evaluator(
                    training.copy(deep=True),
                    MappingProxyType(dict(parameters)),
                )
            )
            overlap = set(names).intersection(metrics) | {"possible_overfit"}.intersection(
                {*names, *metrics}
            )
            if overlap:
                raise ValueError(
                    f"parameter and result columns must be distinct: {sorted(overlap)}"
                )
            if metric_names is None:
                metric_names = tuple(metrics)
            elif tuple(metrics) != metric_names:
                raise ValueError("evaluator must return the same metrics for every grid point")
            rows.append({**parameters, **metrics})

        surface = pd.DataFrame(rows)
        objective = surface[self.settings.objective]
        best_position = int(objective.idxmax() if self.settings.maximize else objective.idxmin())
        isolated = self._isolated(objective, coordinates, best_position)
        surface["possible_overfit"] = False
        surface.loc[best_position, "possible_overfit"] = isolated
        best_parameters = {name: surface.loc[best_position, name] for name in names}
        return SensitivityResult(
            surface=surface,
            best_parameters=best_parameters,
            best_objective=float(objective.loc[best_position]),
            possible_overfit=isolated,
            objective=self.settings.objective,
        )

    def _metrics(self, values: Mapping[str, Real]) -> dict[str, float]:
        if self.settings.objective not in values:
            raise ValueError(f"evaluator result is missing objective: {self.settings.objective}")
        metrics: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"metric {name} must be finite numeric data")
            metrics[name] = float(value)
        return metrics

    def _isolated(
        self,
        objective: pd.Series,
        coordinates: tuple[tuple[int, ...], ...],
        best_position: int,
    ) -> bool:
        best_coordinate = coordinates[best_position]
        neighbor_positions = [
            position
            for position, coordinate in enumerate(coordinates)
            if sum(
                abs(left - right) for left, right in zip(coordinate, best_coordinate, strict=True)
            )
            == 1
        ]
        if not neighbor_positions:
            return False
        neighbors = objective.iloc[neighbor_positions]
        comparator = float(neighbors.max() if self.settings.maximize else neighbors.min())
        best = float(objective.iloc[best_position])
        improvement = best - comparator if self.settings.maximize else comparator - best
        scale = max(abs(best), abs(comparator), 1e-12)
        return improvement / scale >= self.settings.isolation_threshold


def evaluate_selected(
    data: pd.DataFrame,
    result: SensitivityResult,
    evaluator: MetricEvaluator,
) -> dict[str, float]:
    """Evaluate fixed training-selected parameters without performing reselection."""

    parameters = dict(result.best_parameters)
    raw = evaluator(data.copy(deep=True), MappingProxyType(parameters))
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"metric {name} must be finite numeric data")
        metrics[name] = float(value)
    return metrics
