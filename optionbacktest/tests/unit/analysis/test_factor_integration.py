from __future__ import annotations

import pytest

from spx_0dte_backtest.analysis import FactorScoreEngine
from spx_0dte_backtest.strategies import IronCondorStrategy
from tests.unit.strategies.test_iron_condor import context


def test_candidate_retains_immutable_point_in_time_features() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    assert candidate.context_features["regime"] == "RANGE"
    with pytest.raises(TypeError):
        candidate.context_features["regime"] = "TREND_UP"  # type: ignore[index]


def test_candidate_scoring_uses_actual_selected_short_strikes() -> None:
    candidate = IronCondorStrategy().generate_trade(context())
    features = dict(candidate.context_features)
    features.update(
        {
            "realized_vol_30m": 0.15,
            "range_expansion": 1.0,
            "momentum_classification": "NEUTRAL",
            "opening_range_breakout": "INSIDE",
            "expected_move": 20.0,
            "estimated_gamma_exposure": None,
        }
    )
    candidate = candidate.__class__(
        **{
            field: getattr(candidate, field)
            for field in candidate.__dataclass_fields__
            if field != "context_features"
        },
        context_features=features,
    )

    scored = FactorScoreEngine().enrich_candidate(candidate)

    assert scored.context_features["short_put_strike"] == 5970
    assert scored.context_features["short_call_strike"] == 6030
    assert scored.context_features["sr_score"] == 2
    assert scored.context_features["factor_score"] == 8
    assert scored.context_features["factor_score_version"] == "ic_suitability_v1"
