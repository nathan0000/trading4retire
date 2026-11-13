"""Decision-matrix boundaries for the intraday gate (pure classifier, no data)."""

import pytest
from pydantic import ValidationError

from regime_engine.domain.intraday import IntradayRegimeConfig, PrimaryRegime, VolatilityState
from regime_engine.inference.intraday_gate import GateInputs, classify_intraday

CFG = IntradayRegimeConfig()


def _inputs(**overrides: object) -> GateInputs:
    fields: dict[str, object] = {
        "close": 101.0,
        "ema_fast": 100.5,
        "ema_slow": 100.0,
        "vwap": 100.2,
        "rsi": 60.0,
        "volatility_state": VolatilityState.NORMAL,
        "volatility_ratio": 1.0,
        "proxy_bias_bps": (("spy_close", 5.0), ("qqq_close", 4.0)),
    }
    fields.update(overrides)
    return GateInputs(**fields)  # type: ignore[arg-type]


def _bear(**overrides: object) -> GateInputs:
    base: dict[str, object] = {
        "close": 99.0,
        "ema_fast": 99.5,
        "vwap": 99.8,
        "rsi": 40.0,
        "proxy_bias_bps": (("spy_close", -5.0), ("qqq_close", -4.0)),
    }
    base.update(overrides)
    return _inputs(**base)


def test_bull_trend_confirmed() -> None:
    decision = classify_intraday(_inputs(), CFG)
    assert decision.regime is PrimaryRegime.BULL_TREND
    assert decision.internal_bias_match
    assert "proxies confirm" in decision.reasons[0]


def test_bear_trend_confirmed() -> None:
    decision = classify_intraday(_bear(), CFG)
    assert decision.regime is PrimaryRegime.BEAR_TREND
    assert decision.internal_bias_match


def test_bull_rejected_below_vwap() -> None:
    decision = classify_intraday(_inputs(close=100.1), CFG)
    assert decision.regime is PrimaryRegime.WHIPSAW_NEUTRAL
    assert decision.reasons == (
        "Bullish EMA detected but rejected by below-VWAP structural price action",
    )


def test_close_equal_to_vwap_confirms_neither_side() -> None:
    assert classify_intraday(_inputs(close=100.2), CFG).regime is PrimaryRegime.WHIPSAW_NEUTRAL
    assert classify_intraday(_bear(close=99.8), CFG).regime is PrimaryRegime.WHIPSAW_NEUTRAL


@pytest.mark.parametrize(("rsi", "regime"), [(69.9, "BULL_TREND"), (70.0, "WHIPSAW_NEUTRAL")])
def test_bull_rsi_exhaustion_boundary(rsi: float, regime: str) -> None:
    assert classify_intraday(_inputs(rsi=rsi), CFG).regime.value == regime


@pytest.mark.parametrize(("rsi", "regime"), [(30.1, "BEAR_TREND"), (30.0, "WHIPSAW_NEUTRAL")])
def test_bear_rsi_overextension_boundary(rsi: float, regime: str) -> None:
    assert classify_intraday(_bear(rsi=rsi), CFG).regime.value == regime


def test_both_rejections_reported() -> None:
    reason = classify_intraday(_inputs(close=100.0, rsi=80.0), CFG).reasons[0]
    assert "below-VWAP" in reason and "RSI 80.0" in reason


def test_critical_volatility_overrides_trend() -> None:
    decision = classify_intraday(
        _inputs(volatility_state=VolatilityState.CRITICAL, volatility_ratio=1.8), CFG
    )
    assert decision.regime is PrimaryRegime.EXTREME_VOLATILITY
    assert "1.80x" in decision.reasons[0]


def test_expanding_volatility_still_allows_trend() -> None:
    decision = classify_intraday(_inputs(volatility_state=VolatilityState.EXPANDING), CFG)
    assert decision.regime is PrimaryRegime.BULL_TREND


def test_flat_emas_are_whipsaw() -> None:
    decision = classify_intraday(_inputs(ema_fast=100.0), CFG)
    assert decision.regime is PrimaryRegime.WHIPSAW_NEUTRAL
    assert not decision.internal_bias_match


def test_all_proxies_heavily_opposed_is_whipsaw() -> None:
    decision = classify_intraday(
        _inputs(proxy_bias_bps=(("spy_close", -3.0), ("qqq_close", -6.0))), CFG
    )
    assert decision.regime is PrimaryRegime.WHIPSAW_NEUTRAL
    assert "bearish proxy internals" in decision.reasons[0]


def test_mild_opposition_keeps_trend_but_flags_no_match() -> None:
    decision = classify_intraday(
        _inputs(proxy_bias_bps=(("spy_close", -1.0), ("qqq_close", -2.9))), CFG
    )
    assert decision.regime is PrimaryRegime.BULL_TREND
    assert not decision.internal_bias_match


def test_split_proxies_is_not_complete_conflict() -> None:
    decision = classify_intraday(
        _inputs(proxy_bias_bps=(("spy_close", -9.0), ("qqq_close", 2.0))), CFG
    )
    assert decision.regime is PrimaryRegime.BULL_TREND
    assert not decision.internal_bias_match
    assert "proxies mixed" in decision.reasons[0]


def test_no_proxies_never_match() -> None:
    decision = classify_intraday(_inputs(proxy_bias_bps=()), CFG)
    assert decision.regime is PrimaryRegime.BULL_TREND
    assert not decision.internal_bias_match
    assert "internals unconfirmed" in decision.reasons[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"ema_slow": None},
        {"vwap": None},
        {"rsi": None},
        {"volatility_state": VolatilityState.UNKNOWN},
    ],
)
def test_missing_indicator_is_unknown(overrides: dict[str, object]) -> None:
    decision = classify_intraday(_inputs(**overrides), CFG)
    assert decision.regime is PrimaryRegime.UNKNOWN
    assert decision.reasons[0].startswith("Indicators not available")


@pytest.mark.parametrize(
    "overrides",
    [
        {"ema_fast": 40, "ema_slow": 20},
        {"rsi_oversold": 70, "rsi_overbought": 30},
        {"expanding_ratio": 2.0, "critical_ratio": 1.5},
        {"bar_minutes": 5, "baseline_minutes": 12},
    ],
)
def test_config_rejects_inconsistent_thresholds(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        IntradayRegimeConfig(**overrides)  # type: ignore[arg-type]
