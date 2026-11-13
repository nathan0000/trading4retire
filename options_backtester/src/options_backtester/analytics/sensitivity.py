"""Parameter sweeps and sensitivity matrices.

The whole surface is reported, never just the best cell: an isolated peak
surrounded by poor neighbours is a sign of overfitting, not of an edge.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any

import pandas as pd

from options_backtester.analytics.metrics import compute_metrics
from options_backtester.data.base import DataProvider
from options_backtester.engine.backtester import BacktestConfig, Backtester
from options_backtester.regimes.classifier import RegimeTimeline
from options_backtester.strategies import build_strategy


def set_path(config: dict[str, Any], path: str, value: Any) -> None:
    """Set ``config["a"]["b"]`` from the dotted path ``"a.b"``."""
    node = config
    *parents, leaf = path.split(".")
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into {path!r}: {key!r} is not a mapping")
    node[leaf] = value


def run_grid(
    provider: DataProvider,
    base_strategy: dict[str, Any],
    sweep: dict[str, list[Any]],
    backtest: BacktestConfig,
    regimes: RegimeTimeline | None = None,
) -> pd.DataFrame:
    """Run every combination of ``sweep`` (dotted path -> values); one row per run."""
    paths = list(sweep)
    rows = []
    for values in itertools.product(*(sweep[p] for p in paths)):
        config = copy.deepcopy(base_strategy)
        for path, value in zip(paths, values, strict=True):
            set_path(config, path, value)
        result = Backtester(provider, build_strategy(config), backtest, regimes).run()
        rows.append(
            {
                **dict(zip(paths, values, strict=True)),
                **compute_metrics(result.equity, result.trades),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_matrix(grid: pd.DataFrame, x: str, y: str, metric: str) -> pd.DataFrame:
    """Pivot grid results to a ``y`` (rows) by ``x`` (columns) table of ``metric``.

    Any other swept parameters are averaged over.
    """
    return grid.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
