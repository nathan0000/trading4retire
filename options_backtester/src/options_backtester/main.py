"""CLI entry point: run one backtest (``run``) or a parameter sweep (``grid``)."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from options_backtester.analytics.metrics import compute_metrics
from options_backtester.analytics.reporter import write_report, write_sensitivity
from options_backtester.analytics.sensitivity import run_grid
from options_backtester.data.base import DataProvider, DataUnavailableError
from options_backtester.data.cache import ParquetCache
from options_backtester.data.greeks_engine import GreeksConfig
from options_backtester.data.providers import PROVIDERS, build_provider
from options_backtester.engine.backtester import BacktestConfig, Backtester
from options_backtester.regimes.classifier import RegimeClassifier, RegimeTimeline
from options_backtester.strategies import build_strategy

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def build_regimes(
    provider: DataProvider, backtest: BacktestConfig, vix_symbol: str | None
) -> RegimeTimeline:
    """Classify regimes from the underlying, using a real implied-vol index if available."""
    start = backtest.start - timedelta(days=backtest.regime_warmup_days)
    bars = provider.get_underlying_bars(backtest.symbol, start, backtest.end)
    vix = None
    if vix_symbol:
        try:
            vix = provider.get_underlying_bars(vix_symbol, start, backtest.end)
        except (DataUnavailableError, RuntimeError) as exc:
            print(
                f"warning: no {vix_symbol} data ({exc}); using realized-vol proxy", file=sys.stderr
            )
    return RegimeClassifier().classify(bars, vix=vix)


def _setup(args: argparse.Namespace) -> tuple[DataProvider, BacktestConfig, dict[str, Any], dict]:
    settings = load_yaml(args.settings)
    grid = load_yaml(args.grid)
    data = settings.get("data", {})
    name = args.provider or data.get("provider", "synthetic")
    cache = (
        None
        if args.no_cache
        else ParquetCache(args.cache_dir or data.get("cache_dir", "data_cache"))
    )
    provider = build_provider(
        name,
        cache=cache,
        greeks=GreeksConfig(**settings.get("greeks", {})),
        options=data.get("providers", {}).get(name.split("-")[0], {}),
    )
    raw = dict(settings.get("backtest", {}))
    for key in ("symbol", "start", "end"):
        if getattr(args, key) is not None:
            raw[key] = getattr(args, key)
    return provider, BacktestConfig.model_validate(raw), grid, settings.get("report", {})


def _print_metrics(metrics: dict[str, float]) -> None:
    for key in (
        "net_pnl",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "trades",
        "win_rate",
        "profit_factor",
        "expectancy",
        "avg_days_in_trade",
    ):
        value = metrics.get(key)
        print(f"  {key:>18}: {value:,.4f}" if value is not None else f"  {key:>18}: n/a")


def cmd_run(args: argparse.Namespace) -> int:
    provider, backtest, grid, report = _setup(args)
    strategy = build_strategy(grid["base"])
    regimes = build_regimes(provider, backtest, args.vix_symbol)
    result = Backtester(provider, strategy, backtest, regimes).run()
    out = Path(args.out or report.get("out_dir", "results")) / strategy.config.name
    write_report(result, out, int(report.get("attribution_min_trades", 10)))
    if result.synthetic_data:
        print("NOTE: synthetic option prices - not a historical performance result.")
    _print_metrics(compute_metrics(result.equity, result.trades))
    print(f"report: {out / 'blueprint.md'}")
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    provider, backtest, grid, report = _setup(args)
    regimes = build_regimes(provider, backtest, args.vix_symbol)
    table = run_grid(provider, grid["base"], grid["sweep"], backtest, regimes)
    matrix = grid.get("matrix") or {}
    paths = list(grid["sweep"])
    x, y = matrix.get("x", paths[0]), matrix.get("y", paths[-1])
    out = Path(args.out or report.get("out_dir", "results")) / f"{grid['base']['name']}_grid"
    write_sensitivity(table, x, y, matrix.get("metrics", ["sharpe", "net_pnl"]), out)
    if provider.synthetic:
        print("NOTE: synthetic option prices - not a historical performance result.")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[[*paths, "trades", "net_pnl", "sharpe", "win_rate", "max_drawdown"]].round(3))
    print(f"report: {out / 'sensitivity.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="options-backtester", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler, text in (
        ("run", cmd_run, "backtest the `base` strategy and write a Strategy Blueprint"),
        ("grid", cmd_grid, "sweep parameters and write sensitivity matrices"),
    ):
        sub = commands.add_parser(name, help=text)
        sub.set_defaults(handler=handler)
        sub.add_argument("--settings", default=str(DEFAULT_CONFIG_DIR / "settings.yaml"))
        sub.add_argument("--grid", default=str(DEFAULT_CONFIG_DIR / "strategy_grid.yaml"))
        sub.add_argument("--provider", choices=PROVIDERS)
        sub.add_argument("--symbol")
        sub.add_argument("--start", type=date.fromisoformat)
        sub.add_argument("--end", type=date.fromisoformat)
        sub.add_argument(
            "--vix-symbol", help="implied-vol index for the volatility regime, e.g. ^VIX"
        )
        sub.add_argument("--cache-dir")
        sub.add_argument("--no-cache", action="store_true")
        sub.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
