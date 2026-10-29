from __future__ import annotations

import pandas as pd
import pytest

from spx_0dte_backtest.analysis import FactorScoreEngine, FactorScoreSettings

TIMESTAMP = pd.Timestamp("2026-08-12 11:00", tz="America/New_York")


def row(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "timestamp": TIMESTAMP,
        "data_cutoff_timestamp": TIMESTAMP,
        "regime": "RANGE",
        "realized_vol_30m": 0.15,
        "range_expansion": 1.0,
        "momentum_classification": "NEUTRAL",
        "opening_range_breakout": "INSIDE",
        "close": 6000.0,
        "expected_move": 30.0,
        "short_put_strike": 5960.0,
        "short_call_strike": 6040.0,
        "estimated_gamma_exposure": None,
    }
    values.update(overrides)
    return values


def test_range_context_scores_as_high_ic_suitability() -> None:
    score = FactorScoreEngine().score_row(pd.Series(row()))
    assert score.trend_score == 2
    assert score.volatility_score == 2
    assert score.sr_score == 2
    assert score.momentum_score == 2
    assert score.gamma_score == 0
    assert score.factor_score == 8


@pytest.mark.parametrize("regime", ["TREND_UP", "TREND_DOWN"])
def test_trend_score_is_symmetric_not_directional(regime: str) -> None:
    score = FactorScoreEngine().score_row(pd.Series(row(regime=regime)))
    assert score.trend_score == -1


def test_high_vol_strong_momentum_is_low_suitability() -> None:
    score = FactorScoreEngine().score_row(
        pd.Series(
            row(
                regime="HIGH_VOL_EXPANSION",
                realized_vol_30m=0.35,
                range_expansion=2.0,
                momentum_classification="STRONG_UP",
                opening_range_breakout="ABOVE",
                short_put_strike=5990,
                short_call_strike=6010,
            )
        )
    )
    assert score.trend_score == -2
    assert score.volatility_score == -2
    assert score.sr_score == -2
    assert score.momentum_score == -2


def test_gamma_is_neutral_when_estimate_is_unavailable() -> None:
    score = FactorScoreEngine().score_row(pd.Series(row(estimated_gamma_exposure=None)))
    assert score.gamma_score == 0


@pytest.mark.parametrize(
    ("gex", "expected"),
    [(2_000_000_000.0, 2), (500_000_000.0, 1), (-500_000_000.0, -1), (-2_000_000_000.0, -2)],
)
def test_estimated_gamma_score_is_explicit_and_bounded(gex: float, expected: int) -> None:
    assert (
        FactorScoreEngine().score_row(pd.Series(row(estimated_gamma_exposure=gex))).gamma_score
        == expected
    )


def test_enrich_adds_all_score_columns_without_mutating_input() -> None:
    frame = pd.DataFrame(
        [
            row(),
            row(
                timestamp=TIMESTAMP + pd.Timedelta("5min"),
                data_cutoff_timestamp=TIMESTAMP + pd.Timedelta("5min"),
            ),
        ]
    )
    original = frame.copy(deep=True)
    result = FactorScoreEngine().enrich(frame)
    assert {
        "trend_score",
        "volatility_score",
        "sr_score",
        "momentum_score",
        "gamma_score",
        "factor_score",
    }.issubset(result.columns)
    pd.testing.assert_frame_equal(frame, original)


def test_future_rows_cannot_change_current_score() -> None:
    current = pd.DataFrame([row()])
    future = pd.concat(
        [
            current,
            pd.DataFrame(
                [
                    row(
                        timestamp=TIMESTAMP + pd.Timedelta("5min"),
                        data_cutoff_timestamp=TIMESTAMP + pd.Timedelta("5min"),
                        regime="TREND_UP",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    first = FactorScoreEngine().enrich(current).iloc[0]
    second = FactorScoreEngine().enrich(future).iloc[0]
    assert first["factor_score"] == second["factor_score"]


def test_future_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        FactorScoreEngine().enrich(
            pd.DataFrame([row(data_cutoff_timestamp=TIMESTAMP + pd.Timedelta("5min"))])
        )


def test_thresholds_are_configurable() -> None:
    settings = FactorScoreSettings(moderate_realized_vol_min=0.16)
    assert FactorScoreEngine(settings).score_row(pd.Series(row())).volatility_score == 1


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        FactorScoreSettings(moderate_realized_vol_min=0.30, moderate_realized_vol_max=0.10)
