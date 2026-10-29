"""Unit tests for RegimeSnapshot: immutability, score bounds, enum/optional handling."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from regime_engine.domain.enums import (
    Direction,
    Gamma,
    Liquidity,
    MarketSession,
    Stability,
    Volatility,
)
from regime_engine.domain.models import RegimeSnapshot

AS_OF = datetime(2026, 1, 5, 10, 45, tzinfo=UTC)
GENERATED_AT = datetime(2026, 1, 5, 10, 45, 3, tzinfo=UTC)


def _make_snapshot(**overrides: object) -> RegimeSnapshot:
    fields: dict[str, object] = {
        "snapshot_id": "snap-0001",
        "as_of": AS_OF,
        "generated_at": GENERATED_AT,
        "market_session": MarketSession.REGULAR,
        "universe": "SPX",
        "direction": Direction.NEUTRAL,
        "volatility": Volatility.NORMAL,
        "liquidity": Liquidity.NORMAL,
        "gamma": Gamma.POSITIVE,
        "composite_regime": "RANGE_POSITIVE_GAMMA",
        "confidence": 0.75,
        "config_version": "0.1.0",
        "classifier_version": "0.1.0",
        "feature_set_version": "0.1.0",
        "input_fingerprint": "deadbeef",
    }
    fields.update(overrides)
    return RegimeSnapshot.model_validate(fields)


def test_construct_minimal_valid_snapshot() -> None:
    snapshot = _make_snapshot()
    assert snapshot.schema_version == "0.1.0"
    assert snapshot.stability is Stability.UNKNOWN
    assert snapshot.trend_score is None
    assert snapshot.features == {}
    assert snapshot.evidence_for == ()


def test_snapshot_is_immutable() -> None:
    snapshot = _make_snapshot()
    with pytest.raises(ValidationError):
        snapshot.confidence = 0.99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trend_score", 100.0),
        ("trend_score", -100.0),
        ("volatility_score", 0.0),
        ("volatility_score", 100.0),
        ("cycle_score", 100.0),
        ("cycle_confidence", 1.0),
        ("transition_probability", 1.0),
        ("confidence", 1.0),
    ],
)
def test_scores_accept_boundary_values(field: str, value: float) -> None:
    snapshot = _make_snapshot(**{field: value})
    assert getattr(snapshot, field) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trend_score", 100.01),
        ("trend_score", -100.01),
        ("volatility_score", -0.01),
        ("volatility_score", 100.01),
        ("cycle_score", -100.01),
        ("cycle_confidence", 1.01),
        ("cycle_confidence", -0.01),
        ("transition_probability", 1.01),
        ("confidence", 1.01),
        ("confidence", -0.01),
    ],
)
def test_scores_reject_out_of_bounds_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        _make_snapshot(**{field: value})


def test_snapshot_accepts_enum_string_values() -> None:
    snapshot = _make_snapshot(direction="BULL", volatility="EXPANDING")
    assert snapshot.direction is Direction.BULL
    assert snapshot.volatility is Volatility.EXPANDING


def test_snapshot_rejects_unknown_enum_value() -> None:
    with pytest.raises(ValidationError):
        _make_snapshot(direction="SIDEWAYS")


def test_snapshot_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _make_snapshot(unexpected_field=True)


def test_snapshot_rejects_naive_as_of() -> None:
    with pytest.raises(ValidationError):
        _make_snapshot(as_of=datetime(2026, 1, 5, 10, 45))


def test_snapshot_rejects_empty_snapshot_id() -> None:
    with pytest.raises(ValidationError):
        _make_snapshot(snapshot_id="")
