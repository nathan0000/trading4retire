from __future__ import annotations

import pandas as pd
import pytest

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.features import FeatureEngine
from spx_0dte_backtest.regimes import MarketRegime, RegimeEngine, RegimeSettings
from tests.unit.features.test_feature_engine import make_bars

TIMESTAMP = pd.Timestamp("2026-08-12 11:00", tz="America/New_York")


def feature_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": TIMESTAMP,
        "data_cutoff_timestamp": TIMESTAMP,
        "close": 6002.0,
        "vwap": 6000.0,
        "vwap_slope": 0.5,
        "distance_from_vwap": 2.0,
        "atr": 4.0,
        "adx": 25.0,
        "return_30m": 0.003,
        "range_expansion": 1.1,
        "realized_vol_30m": 0.15,
        "vix": 20.0,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, MarketRegime.TREND_UP),
        (
            {
                "close": 5998.0,
                "vwap": 6000.0,
                "vwap_slope": -0.5,
                "distance_from_vwap": -2.0,
                "return_30m": -0.003,
            },
            MarketRegime.TREND_DOWN,
        ),
        (
            {"adx": 12.0, "distance_from_vwap": 0.5, "return_30m": 0.0001},
            MarketRegime.RANGE,
        ),
        (
            {"range_expansion": 1.7, "realized_vol_30m": 0.25},
            MarketRegime.HIGH_VOL_EXPANSION,
        ),
        (
            {
                "adx": 22.0,
                "return_30m": 0.0001,
                "range_expansion": 0.8,
                "realized_vol_30m": 0.08,
            },
            MarketRegime.LOW_VOL,
        ),
        ({"atr": None}, MarketRegime.UNKNOWN),
    ],
)
def test_classifies_required_regimes(overrides, expected: MarketRegime) -> None:
    result = RegimeEngine().classify(pd.DataFrame([feature_row(**overrides)]))
    assert result.iloc[0]["regime"] == expected.value
    assert result.iloc[0]["regime_version"] == "rule_based_v1"


def test_high_vol_expansion_has_precedence_over_trend() -> None:
    result = RegimeEngine().classify(
        pd.DataFrame([feature_row(range_expansion=2.0, realized_vol_30m=0.30)])
    )
    assert result.iloc[0]["regime"] == MarketRegime.HIGH_VOL_EXPANSION.value


def test_range_has_precedence_over_low_vol_when_both_match() -> None:
    row = feature_row(
        adx=10.0,
        distance_from_vwap=0.2,
        return_30m=0.0,
        range_expansion=0.8,
        realized_vol_30m=0.08,
    )
    result = RegimeEngine().classify(pd.DataFrame([row]))
    assert result.iloc[0]["regime"] == MarketRegime.RANGE.value


def test_thresholds_are_configurable_not_permanently_hard_coded() -> None:
    frame = pd.DataFrame([feature_row(return_30m=0.001)])
    assert RegimeEngine().classify(frame).iloc[0]["regime"] == MarketRegime.UNKNOWN.value
    settings = RegimeSettings(trend_return_threshold=0.0005)
    assert RegimeEngine(settings).classify(frame).iloc[0]["regime"] == MarketRegime.TREND_UP.value


def test_appending_future_rows_cannot_change_current_regime() -> None:
    current = pd.DataFrame([feature_row()])
    future = pd.DataFrame(
        [
            feature_row(),
            feature_row(
                timestamp=TIMESTAMP + pd.Timedelta("5min"),
                data_cutoff_timestamp=TIMESTAMP + pd.Timedelta("5min"),
                return_30m=-0.02,
                close=5900.0,
            ),
        ]
    )
    first = RegimeEngine().classify(current).iloc[0]
    with_future = RegimeEngine().classify(future).iloc[0]
    assert first["regime"] == with_future["regime"]
    assert first["regime_reason"] == with_future["regime_reason"]


def test_input_is_not_mutated() -> None:
    frame = pd.DataFrame([feature_row()])
    original = frame.copy(deep=True)
    RegimeEngine().classify(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_non_finite_signal_is_unknown() -> None:
    result = RegimeEngine().classify(pd.DataFrame([feature_row(adx=float("inf"))]))
    assert result.iloc[0]["regime"] == MarketRegime.UNKNOWN.value


def test_classifies_real_feature_output_with_prefix_invariance() -> None:
    bars = make_bars(30)
    feature_engine = FeatureEngine(AppConfig())
    through_t = feature_engine.compute(bars.iloc[:20])
    with_future = feature_engine.compute(bars)
    first = RegimeEngine().classify(through_t).iloc[-1]
    second = RegimeEngine().classify(with_future).iloc[19]
    assert first["regime"] == second["regime"]
    assert first["regime_reason"] == second["regime_reason"]


def test_future_data_cutoff_is_rejected() -> None:
    frame = pd.DataFrame([feature_row(data_cutoff_timestamp=TIMESTAMP + pd.Timedelta("5min"))])
    with pytest.raises(ValueError, match="cutoff"):
        RegimeEngine().classify(frame)


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        RegimeEngine().classify(pd.DataFrame({"timestamp": [TIMESTAMP]}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trend_return_threshold": -0.1},
        {"range_adx_max": -1},
        {"range_distance_atr_max": -1},
        {"high_vol_range_expansion_min": 0},
        {"low_vol_realized_vol_max": -1},
    ],
)
def test_invalid_settings_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        RegimeSettings(**kwargs)
