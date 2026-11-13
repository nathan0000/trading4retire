from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from options_backtester.analytics.metrics import compute_metrics
from options_backtester.analytics.regime_attribution import attribute
from options_backtester.analytics.reporter import describe_rules, write_report, write_sensitivity
from options_backtester.analytics.sensitivity import run_grid, sensitivity_matrix, set_path
from options_backtester.data.base import market_close_utc
from options_backtester.data.providers.synthetic import SyntheticProvider
from options_backtester.engine.backtester import BacktestConfig, Backtester
from options_backtester.main import main
from options_backtester.regimes import indicators
from options_backtester.regimes.classifier import RegimeClassifier, RegimeConfig
from options_backtester.strategies import build_strategy


def bars(closes, start="2020-01-01"):
    days = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": [market_close_utc(d.date()) for d in days],
            "symbol": "X",
            "open": closes,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": 1.0,
        }
    )


@pytest.fixture(scope="module")
def noisy():
    rng = np.random.default_rng(3)
    return bars(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 700))))


# -- regimes -------------------------------------------------------------


def test_classifier_is_causal(noisy):
    full = RegimeClassifier().classify(noisy).frame
    part = RegimeClassifier().classify(noisy.iloc[:500]).frame
    pd.testing.assert_frame_equal(full.iloc[:500], part)


def test_trend_regimes():
    config = RegimeConfig(fast_sma=20, slow_sma=50)
    up = RegimeClassifier(config).classify(bars(100 * 1.004 ** np.arange(200))).frame
    down = RegimeClassifier(config).classify(bars(100 * 0.996 ** np.arange(200))).frame
    flat = RegimeClassifier(config).classify(bars(100 + np.sin(np.arange(200)))).frame
    assert up["trend"].iloc[-1] == "BULLISH" and down["trend"].iloc[-1] == "BEARISH"
    assert flat["trend"].iloc[-1] == "SIDEWAYS"
    assert up["trend"].iloc[0] == "UNKNOWN"  # warm-up is not guessed


def test_volatility_regime_from_vix_levels_and_term_structure(noisy):
    n = len(noisy)
    level = np.r_[np.full(n - 20, 16.0), np.linspace(16, 40, 20)]
    level[:300] = np.linspace(12, 20, 300)
    vix = bars(level).assign(symbol="VIX")
    back = bars(np.full(n, 22.0)).assign(symbol="VIX3M")
    frame = RegimeClassifier().classify(noisy, vix=vix, vix_back=back).frame
    last = frame.iloc[-1]
    assert last["volatility"] == "HIGH_IV" and last["vol_dynamics"] == "EXPANSION"
    assert last["term_structure"] == "BACKWARDATION" and last["iv_rank"] == pytest.approx(100.0)
    assert frame["term_structure"].iloc[400] == "CONTANGO"
    assert (frame["vol_source"] == "IMPLIED_INDEX").all()
    proxy = RegimeClassifier().classify(noisy).frame
    assert (proxy["vol_source"] == "REALIZED_PROXY").all()
    assert set(proxy["term_structure"]) == {"UNKNOWN"}


def test_timeline_lookup_is_at_or_before(noisy):
    timeline = RegimeClassifier().classify(noisy)
    stamp = noisy["timestamp"].iloc[400]
    assert timeline.at(stamp).timestamp == stamp
    assert timeline.at(stamp - timedelta(minutes=1)).timestamp == noisy["timestamp"].iloc[399]
    assert timeline.at(noisy["timestamp"].iloc[0] - timedelta(days=1)) is None


def test_indicators():
    series = pd.Series([10.0, 20.0, 15.0, 30.0, 25.0])
    assert indicators.iv_rank(series, 5, 2).iloc[-1] == pytest.approx(75.0)
    assert indicators.iv_percentile(series, 5, 2).iloc[-1] == pytest.approx(75.0)
    assert indicators.term_structure_ratio(pd.Series([20.0]), pd.Series([16.0])).iloc[0] == 1.25
    trend = pd.Series(100 * 1.01 ** np.arange(80))
    assert indicators.adx(trend * 1.01, trend * 0.99, trend).iloc[-1] > 40
    rng = np.random.default_rng(0)
    walk = pd.Series(100 + np.cumsum(rng.normal(0, 1, 600)))
    assert indicators.hurst_exponent(walk, 500).iloc[-1] == pytest.approx(0.5, abs=0.12)


# -- metrics -------------------------------------------------------------


def test_metrics_on_a_known_curve():
    stamps = pd.to_datetime(["2023-01-02", "2023-07-03", "2024-01-02"], utc=True)
    equity = pd.DataFrame({"timestamp": stamps, "equity": [100_000.0, 90_000.0, 121_000.0]})
    trades = pd.DataFrame(
        {
            "net_pnl": [300.0, -100.0, 200.0, -200.0],
            "days_in_trade": [10, 20, 30, 40],
            "margin": 1000.0,
        }
    )
    m = compute_metrics(equity, trades)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["cagr"] == pytest.approx(0.21, abs=1e-3)
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["win_rate"] == 0.5 and m["profit_factor"] == pytest.approx(500 / 300)
    assert m["expectancy"] == pytest.approx(50.0) and m["avg_days_in_trade"] == 25
    assert m["worst_trade"] == -200.0 and m["return_on_margin"] == pytest.approx(0.05)
    empty = compute_metrics(equity, pd.DataFrame())
    assert empty["trades"] == 0 and np.isnan(empty["win_rate"])


def test_regime_attribution_needs_enough_trades():
    trades = pd.DataFrame(
        {
            "net_pnl": [100.0, 120.0, 80.0, 110.0, -300.0, -250.0, -280.0, 500.0],
            "days_in_trade": 10,
            "entry_trend": ["SIDEWAYS"] * 4 + ["BEARISH"] * 3 + ["BULLISH"],
            "entry_volatility": ["HIGH_IV"] * 4 + ["LOW_IV"] * 4,
            "entry_vol_dynamics": "STABLE",
        }
    )
    result = attribute(trades, min_trades=3)
    assert result.ideal_labels()[0] in ("SIDEWAYS", "HIGH_IV", "SIDEWAYS|HIGH_IV")
    assert "BEARISH" in result.avoid_labels()
    # One +500 trade is the best average, but one trade proves nothing.
    assert "BULLISH" not in result.ideal_labels()
    by_trend = result.tables["trend"].set_index("regime")
    assert by_trend.loc["SIDEWAYS", "win_rate"] == 1.0 and by_trend.loc["BEARISH", "trades"] == 3


# -- sweeps, reports, CLI -------------------------------------------------

CONDOR = {
    "name": "ic",
    "template": "iron_condor",
    "strike": {"method": "delta", "value": 0.16},
    "wing": {"method": "dollar", "value": 10},
    "entry": {"max_open_positions": 2, "min_days_between_entries": 7},
}


@pytest.fixture(scope="module")
def synthetic_run():
    provider = SyntheticProvider()
    config = BacktestConfig(symbol="SPY", start=date(2023, 1, 3), end=date(2023, 9, 29))
    return provider, config, Backtester(provider, build_strategy(CONDOR), config).run()


def test_synthetic_backtest_is_internally_consistent(synthetic_run):
    _, config, result = synthetic_run
    trades, legs = result.trades, result.legs
    assert len(trades) > 5 and len(legs) == 4 * len(trades)
    assert result.synthetic_data
    assert (trades["entry_dte"].between(30, 45)).all()
    assert (trades["entry_value"] < 0).all()  # condors are sold for a credit
    assert (trades["slippage_paid"] > 0).all()
    np.testing.assert_allclose(
        legs.groupby("trade_id")["pnl"].sum(), trades["gross_pnl"], atol=1e-6
    )
    final = result.equity["equity"].iloc[-1]
    assert final == pytest.approx(config.initial_capital + trades["net_pnl"].sum())
    assert (trades["exit_time"] > trades["entry_time"]).all()
    assert set(trades["exit_reason"]) <= {"PROFIT_TARGET", "STOP_LOSS", "DTE_EXIT", "END_OF_DATA"}


def test_backtest_is_deterministic(synthetic_run):
    provider, config, result = synthetic_run
    again = Backtester(provider, build_strategy(CONDOR), config).run()
    pd.testing.assert_frame_equal(again.trades, result.trades)


def test_report_files_and_blueprint_content(synthetic_run, tmp_path):
    _, _, result = synthetic_run
    out = write_report(result, tmp_path, min_trades=3)
    names = {p.name for p in out.iterdir()}
    assert {
        "blueprint.md",
        "blueprint.json",
        "metrics.json",
        "trades.csv",
        "trades.parquet",
        "legs.parquet",
        "equity.parquet",
        "equity.png",
    } <= names
    text = (out / "blueprint.md").read_text()
    for needle in (
        "SYNTHETIC OPTION PRICES",
        "Ideal market regimes",
        "0.16 delta",
        "Stop loss",
        "Sharpe",
        "Exit reasons",
    ):
        assert needle in text
    assert len(pd.read_parquet(out / "legs.parquet")) == len(result.legs)


def test_rule_description_follows_priority_order():
    rules = describe_rules(build_strategy(CONDOR).config.model_dump(mode="json"))
    order = [line.split(":")[0] for line in rules["exit_priority_order"]]
    assert order == ["Expiration", "Stop loss", "Profit target", "DTE exit"]


def test_grid_and_sensitivity_matrix(synthetic_run, tmp_path):
    provider, config, _ = synthetic_run
    sweep = {"strike.value": [0.10, 0.20], "wing.value": [5, 10]}
    grid = run_grid(provider, CONDOR, sweep, config)
    assert len(grid) == 4 and {"sharpe", "net_pnl", "trades"} <= set(grid.columns)
    matrix = sensitivity_matrix(grid, "strike.value", "wing.value", "net_pnl")
    assert matrix.shape == (2, 2) and list(matrix.columns) == [0.10, 0.20]
    out = write_sensitivity(grid, "strike.value", "wing.value", ["net_pnl"], tmp_path)
    assert (out / "sensitivity_net_pnl.png").exists() and (out / "sensitivity.md").exists()
    config_dict = {"a": {"b": 1}}
    set_path(config_dict, "a.c.d", 2)
    assert config_dict == {"a": {"b": 1, "c": {"d": 2}}}


def test_cli_run_and_grid(tmp_path, capsys):
    common = [
        "--provider",
        "synthetic",
        "--no-cache",
        "--start",
        "2023-01-03",
        "--end",
        "2023-06-30",
        "--out",
        str(tmp_path),
    ]
    assert main(["run", *common]) == 0
    assert (tmp_path / "iron_condor_16d_45dte" / "blueprint.md").exists()
    assert "synthetic option prices" in capsys.readouterr().out
