"""Strategy Blueprint (Markdown + JSON), trade logs, metrics and plots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from options_backtester.analytics.metrics import compute_metrics, drawdown
from options_backtester.analytics.regime_attribution import RegimeAttribution, attribute
from options_backtester.analytics.sensitivity import sensitivity_matrix
from options_backtester.engine.backtester import BacktestResult

SYNTHETIC_BANNER = (
    "**SYNTHETIC OPTION PRICES.** Chains were model-priced, not observed. These results "
    "validate engine mechanics and relative parameter behaviour only; they are not a "
    "historical performance result."
)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _fmt(value: Any, style: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a" if value is None or math.isnan(value) else ("inf" if value > 0 else "-inf")
    if style == "pct":
        return f"{value:.1%}"
    if style == "usd":
        return f"${value:,.0f}"
    if style == "int":
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    if frame.empty:
        return "_none_\n"
    formats = formats or {}
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(_fmt(v, formats.get(str(c), "")) for c, v in row.items()) + " |"
        for row in frame.to_dict("records")
    ]
    return "\n".join([header, rule, *body]) + "\n"


def describe_rules(strategy: dict[str, Any]) -> dict[str, list[str]]:
    """Plain-language entry and exit rules derived from the strategy configuration."""
    strike, wing = strategy["strike"], strategy.get("wing")
    entry, exit_ = strategy["entry"], strategy["exit"]
    unit = {"delta": "delta", "pct_otm": "OTM fraction", "stdev": "standard deviations"}
    structure = [
        f"Template: {strategy['template']}"
        + (
            f" ({strategy['direction']} {strategy['option_side']})"
            if strategy["template"] == "vertical"
            else f" ({strategy['option_side']})"
            if strategy["template"] == "calendar"
            else ""
        ),
        f"Expiration: {strategy['min_dte']}-{strategy['max_dte']} DTE"
        + (f", target {strategy['target_dte']}" if strategy.get("target_dte") else ""),
        f"Anchor strike: {strike['value']:g} {unit[strike['method']]}",
    ]
    if wing:
        wing_unit = {
            "dollar": "points wide",
            "delta": "delta beyond the anchor",
            "pct_underlying": "of the underlying price wide",
        }
        structure.append(f"Wing: {wing['value']:g} {wing_unit[wing['method']]}")
    if strategy["template"] == "calendar":
        structure.append(
            f"Back leg: {strategy['back_min_dte']}-{strategy['back_max_dte']} DTE, same strike"
        )
    structure.append(f"Size: {strategy['quantity']} unit(s) per entry")

    entries = [
        f"At most {entry['max_open_positions']} open position(s); "
        f"at least {entry['min_days_between_entries']} day(s) between entries"
    ]
    labels = {
        "allowed_trend": "Trend regime in",
        "allowed_volatility": "Volatility regime in",
        "allowed_vol_dynamics": "Volatility dynamics in",
    }
    for key, label in labels.items():
        if entry.get(key):
            entries.append(f"{label} {{{', '.join(entry[key])}}}")
    simple = {
        "min_iv_rank": "IV rank >= {}",
        "max_iv_rank": "IV rank <= {}",
        "max_spread_pct": "Each leg's bid-ask spread <= {:.0%} of mid",
        "min_open_interest": "Each leg's open interest >= {}",
        "min_credit": "Executable net credit >= {}",
        "max_credit": "Executable net credit <= {}",
        "max_debit": "Executable net debit <= {}",
        "entry_time": "Snapshot time is {} New York",
    }
    for key, text in simple.items():
        if entry.get(key) is not None:
            entries.append(text.format(entry[key]))

    exits = ["Expiration: settle at intrinsic value (always active)"]
    if exit_.get("stop_loss_multiple") is not None:
        exits.append(
            f"Stop loss: mark-to-mid loss >= {exit_['stop_loss_multiple']:g}x the credit "
            "received / debit paid"
        )
    if exit_.get("underlying_breach"):
        exits.append("Underlying breach: underlying trades through a short strike")
    if exit_.get("max_position_delta") is not None:
        exits.append(f"Delta breach: |net position delta| > {exit_['max_position_delta']:g}")
    if exit_.get("trailing_stop_pct") is not None:
        exits.append(
            f"Trailing stop: give back {exit_['trailing_stop_pct']:.0%} of the credit/debit "
            f"after peaking above {exit_['trailing_activation_pct']:.0%}"
        )
    if exit_.get("profit_target_pct") is not None:
        exits.append(
            f"Profit target: mark-to-mid profit >= {exit_['profit_target_pct']:.0%} of the "
            "credit received / debit paid"
        )
    if exit_.get("exit_dte") is not None:
        exits.append(f"DTE exit: nearest leg at or below {exit_['exit_dte']} DTE")
    if exit_.get("max_hold_days") is not None:
        exits.append(f"Max hold: {exit_['max_hold_days']} calendar days")
    if exit_.get("exit_on_regime_shift"):
        exits.append("Regime shift: regime leaves the allowed set (or changes from entry)")
    return {"structure": structure, "entry": entries, "exit_priority_order": exits}


def build_blueprint(result: BacktestResult, min_trades: int = 10) -> dict[str, Any]:
    metrics = compute_metrics(result.equity, result.trades)
    attribution = attribute(result.trades, min_trades)
    reasons = (
        result.trades["exit_reason"].value_counts().to_dict() if not result.trades.empty else {}
    )
    return {
        "strategy": result.strategy_config["name"],
        "data": {
            "provider": result.provider,
            "synthetic_option_prices": result.synthetic_data,
            "symbol": result.backtest_config["symbol"],
            "start": result.backtest_config["start"],
            "end": result.backtest_config["end"],
        },
        "rules": describe_rules(result.strategy_config),
        "parameters": result.strategy_config,
        "backtest": result.backtest_config,
        "metrics": metrics,
        "ideal_market_regimes": attribution.ideal.to_dict("records"),
        "regimes_to_avoid": attribution.avoid.to_dict("records"),
        "regime_attribution": {k: v.to_dict("records") for k, v in attribution.tables.items()},
        "attribution_min_trades": min_trades,
        "exit_reasons": reasons,
        "rejections": result.rejections,
        "diagnostics": result.diagnostics,
    }


_METRIC_ROWS = [
    ("Net P&L", "net_pnl", "usd"),
    ("Total return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Sharpe", "sharpe", ""),
    ("Sortino", "sortino", ""),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Trades", "trades", "int"),
    ("Win rate", "win_rate", "pct"),
    ("Profit factor", "profit_factor", ""),
    ("Expectancy per trade", "expectancy", "usd"),
    ("Average days in trade", "avg_days_in_trade", ""),
    ("Worst trade", "worst_trade", "usd"),
    ("VaR 95% (per period, of equity)", "var_95", "pct"),
    ("CVaR 95% (per period, of equity)", "cvar_95", "pct"),
    ("Trade VaR 95%", "trade_var_95", "usd"),
    ("Trade CVaR 95%", "trade_cvar_95", "usd"),
]
_REGIME_COLUMNS = ["dimension", "regime", "trades", "win_rate", "avg_pnl", "total_pnl", "t_stat"]
_REGIME_FORMATS = {"win_rate": "pct", "avg_pnl": "usd", "total_pnl": "usd"}


def render_markdown(blueprint: dict[str, Any], attribution: RegimeAttribution) -> str:
    data, metrics = blueprint["data"], blueprint["metrics"]
    lines = [f"# Strategy Blueprint: {blueprint['strategy']}", ""]
    if data["synthetic_option_prices"]:
        lines += [f"> {SYNTHETIC_BANNER}", ""]
    lines += [
        f"{data['symbol']}, {data['start']} to {data['end']}, data provider `{data['provider']}`.",
        "",
    ]
    for title, key in (
        ("Structure", "structure"),
        ("Entry triggers", "entry"),
        ("Exit triggers (highest priority first)", "exit_priority_order"),
    ):
        lines += [f"## {title}", "", *[f"- {rule}" for rule in blueprint["rules"][key]], ""]
    lines += ["## Performance", "", "| Metric | Value |", "| --- | --- |"]
    lines += [
        f"| {label} | {_fmt(metrics.get(key), style)} |" for label, key, style in _METRIC_ROWS
    ]
    n = int(metrics.get("trades") or 0)
    lines += [
        "",
        (
            f"## Ideal market regimes (>= {attribution.min_trades} trades, "
            f"t-statistic >= {attribution.min_t_stat:g})"
        ),
        "",
    ]
    if n < 30:
        lines += [
            f"_Only {n} trades: regime conclusions below are not statistically reliable._",
            "",
        ]
    ideal = attribution.ideal.reindex(columns=_REGIME_COLUMNS)
    lines += [_table(ideal, _REGIME_FORMATS), "## Regimes to avoid", ""]
    lines += [_table(attribution.avoid.reindex(columns=_REGIME_COLUMNS), _REGIME_FORMATS)]
    lines += ["## Regime attribution", ""]
    for name, table in attribution.tables.items():
        columns = [c for c in _REGIME_COLUMNS[1:] if c in table.columns]
        lines += [f"### By {name.replace('_', ' ')}", "", _table(table[columns], _REGIME_FORMATS)]
    lines += ["## Exit reasons", ""]
    lines += [f"- {reason}: {count}" for reason, count in blueprint["exit_reasons"].items()] or [
        "_no trades_"
    ]
    lines += ["", "## Rejected entries", ""]
    lines += [f"- {reason}: {count}" for reason, count in blueprint["rejections"].items()] or [
        "_none_"
    ]
    lines += ["", "## Diagnostics", ""]
    lines += [f"- {key}: {value}" for key, value in blueprint["diagnostics"].items()]
    lines += [
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(blueprint["parameters"], indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_report(result: BacktestResult, out_dir: str | Path, min_trades: int = 10) -> Path:
    """Write the blueprint, trade logs, equity curve, metrics and plots to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blueprint = build_blueprint(result, min_trades)
    attribution = attribute(result.trades, min_trades)
    (out / "blueprint.md").write_text(render_markdown(blueprint, attribution))
    (out / "blueprint.json").write_text(json.dumps(_clean(blueprint), indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(_clean(blueprint["metrics"]), indent=2))
    for name, frame in (
        ("trades", result.trades),
        ("legs", result.legs),
        ("equity", result.equity),
    ):
        frame.to_csv(out / f"{name}.csv", index=False)
        frame.to_parquet(out / f"{name}.parquet", index=False)
    _plot_equity(result.equity, out / "equity.png", blueprint["strategy"])
    return out


def write_sensitivity(
    grid: pd.DataFrame, x: str, y: str, metrics: list[str], out_dir: str | Path
) -> Path:
    """Write the full grid plus one ``y`` by ``x`` matrix (CSV, Markdown, heatmap) per metric."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "grid.csv", index=False)
    sections = ["# Parameter sensitivity", "", f"Rows: `{y}`. Columns: `{x}`.", ""]
    for metric in metrics:
        matrix = sensitivity_matrix(grid, x, y, metric)
        matrix.to_csv(out / f"sensitivity_{metric}.csv")
        shown = matrix.reset_index()
        shown.columns = [str(c) for c in shown.columns]
        sections += [f"## {metric}", "", _table(shown)]
        _plot_heatmap(matrix, out / f"sensitivity_{metric}.png", metric, x, y)
    (out / "sensitivity.md").write_text("\n".join(sections) + "\n")
    return out


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_equity(equity: pd.DataFrame, path: Path, title: str) -> None:
    if equity.empty:
        return
    plt = _pyplot()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot(equity["timestamp"], equity["equity"], color="#1f4e79", linewidth=1.4)
    top.set_title(f"{title}: equity")
    top.set_ylabel("Equity ($)")
    top.grid(alpha=0.25)
    depth = drawdown(equity["equity"].astype(float)) * 100.0
    bottom.fill_between(equity["timestamp"], depth, 0.0, color="#b03a2e", alpha=0.6)
    bottom.set_ylabel("Drawdown (%)")
    bottom.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _plot_heatmap(matrix: pd.DataFrame, path: Path, metric: str, x: str, y: str) -> None:
    if matrix.empty:
        return
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(1.4 * len(matrix.columns) + 3, 0.7 * len(matrix) + 2.5))
    values = matrix.to_numpy(dtype=float)
    bound = float(abs(values[pd.notna(values)]).max()) if pd.notna(values).any() else 1.0
    image = axis.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(
        range(len(matrix.columns)),
        [f"{c:g}" if isinstance(c, float) else str(c) for c in matrix.columns],
    )
    axis.set_yticks(
        range(len(matrix.index)),
        [f"{r:g}" if isinstance(r, float) else str(r) for r in matrix.index],
    )
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(metric)
    for (row, column), value in pd.DataFrame(values).stack().items():
        axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
