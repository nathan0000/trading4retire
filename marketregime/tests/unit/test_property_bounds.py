"""Property tests for score bounds, the point-in-time rule, and determinism (CLAUDE.md 15)."""

from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from regime_engine.domain.enums import Direction, Gamma, Liquidity, MarketSession, Volatility
from regime_engine.domain.errors import PointInTimeViolationError
from regime_engine.domain.models import RegimeSnapshot, ensure_point_in_time
from tests.unit.test_domain_models_observation import _make_observation

AS_OF = datetime(2026, 1, 5, 10, 45, tzinfo=UTC)


def _base_snapshot_fields() -> dict[str, object]:
    return {
        "snapshot_id": "snap-0001",
        "as_of": AS_OF,
        "generated_at": AS_OF,
        "market_session": MarketSession.REGULAR,
        "universe": "SPX",
        "direction": Direction.NEUTRAL,
        "volatility": Volatility.NORMAL,
        "liquidity": Liquidity.NORMAL,
        "gamma": Gamma.POSITIVE,
        "composite_regime": "RANGE_POSITIVE_GAMMA",
        "confidence": 0.5,
        "config_version": "0.1.0",
        "classifier_version": "0.1.0",
        "feature_set_version": "0.1.0",
        "input_fingerprint": "deadbeef",
    }


@given(st.floats(min_value=-100, max_value=100, allow_nan=False))
def test_trend_score_within_bounds_always_accepted(value: float) -> None:
    fields = _base_snapshot_fields()
    fields["trend_score"] = value
    snapshot = RegimeSnapshot.model_validate(fields)
    assert snapshot.trend_score == value


@given(
    st.one_of(
        st.floats(min_value=100.0001, max_value=1_000, allow_nan=False),
        st.floats(min_value=-1_000, max_value=-100.0001, allow_nan=False),
    )
)
def test_trend_score_outside_bounds_always_rejected(value: float) -> None:
    fields = _base_snapshot_fields()
    fields["trend_score"] = value
    try:
        RegimeSnapshot.model_validate(fields)
    except ValidationError:
        pass
    else:  # pragma: no cover - failure path only
        raise AssertionError(f"expected ValidationError for trend_score={value}")


@given(st.floats(min_value=0, max_value=1, allow_nan=False))
def test_confidence_within_bounds_always_accepted(value: float) -> None:
    fields = _base_snapshot_fields()
    fields["confidence"] = value
    snapshot = RegimeSnapshot.model_validate(fields)
    assert snapshot.confidence == value


@given(st.integers(min_value=0, max_value=10_000))
def test_point_in_time_guard_holds_for_any_lag(lag_seconds: int) -> None:
    event = datetime(2026, 1, 5, 16, 0, tzinfo=UTC)
    available = event + timedelta(seconds=lag_seconds)
    obs = _make_observation(event_time=event, available_time=available, received_time=available)

    ensure_point_in_time(obs, as_of=available)  # exactly available: allowed
    ensure_point_in_time(obs, as_of=available + timedelta(seconds=1))  # later: allowed

    if lag_seconds > 0:
        try:
            ensure_point_in_time(obs, as_of=available - timedelta(seconds=1))
        except PointInTimeViolationError:
            pass
        else:  # pragma: no cover - failure path only
            raise AssertionError("expected PointInTimeViolationError for as_of before available")


@given(st.floats(min_value=-100, max_value=100, allow_nan=False))
def test_serialization_is_deterministic(value: float) -> None:
    fields = _base_snapshot_fields()
    fields["trend_score"] = value
    snapshot = RegimeSnapshot.model_validate(fields)
    assert snapshot.model_dump(mode="json") == snapshot.model_dump(mode="json")
