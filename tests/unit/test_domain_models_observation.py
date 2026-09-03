"""Unit tests for Observation[T] and the ensure_point_in_time point-in-time guard."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from regime_engine.domain.enums import Quality
from regime_engine.domain.errors import PointInTimeViolationError
from regime_engine.domain.models import Observation, ensure_point_in_time

EVENT = datetime(2026, 1, 5, 16, 0, tzinfo=UTC)
AVAILABLE = EVENT + timedelta(minutes=5)
RECEIVED = AVAILABLE


def _make_observation(**overrides: object) -> Observation[float]:
    fields: dict[str, object] = {
        "name": "spx_close",
        "value": 5000.0,
        "event_time": EVENT,
        "available_time": AVAILABLE,
        "received_time": RECEIVED,
        "source": "fixture",
        "source_symbol": "SPX",
        "quality": Quality.GOOD,
    }
    fields.update(overrides)
    return Observation[float].model_validate(fields)


def test_construct_valid_observation() -> None:
    obs = _make_observation()
    assert obs.value == 5000.0
    assert obs.quality is Quality.GOOD
    assert obs.metadata == {}


def test_observation_is_immutable() -> None:
    obs = _make_observation()
    with pytest.raises(ValidationError):
        obs.value = 4000.0  # type: ignore[misc]


def test_observation_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        _make_observation(event_time=datetime(2026, 1, 5, 16, 0))


def test_observation_normalizes_non_utc_timezone_to_utc() -> None:
    minus_five = timezone(timedelta(hours=-5))
    local_event = EVENT.astimezone(minus_five)
    obs = _make_observation(event_time=local_event)
    assert obs.event_time == EVENT
    assert obs.event_time.tzinfo == UTC


def test_observation_rejects_unknown_quality_value() -> None:
    with pytest.raises(ValidationError):
        _make_observation(quality="BOGUS")


def test_ensure_point_in_time_passes_when_available_before_as_of() -> None:
    obs = _make_observation()
    ensure_point_in_time(obs, as_of=AVAILABLE + timedelta(seconds=1))
    ensure_point_in_time(obs, as_of=AVAILABLE)  # equal is allowed


def test_ensure_point_in_time_raises_when_available_after_as_of() -> None:
    obs = _make_observation()
    as_of = AVAILABLE - timedelta(seconds=1)
    with pytest.raises(PointInTimeViolationError) as exc_info:
        ensure_point_in_time(obs, as_of=as_of)
    assert exc_info.value.observation_name == "spx_close"
    assert exc_info.value.available_time == AVAILABLE
    assert exc_info.value.as_of == as_of


def test_observation_generic_accepts_missing_value() -> None:
    obs = _make_observation(value=None, quality=Quality.MISSING)
    assert obs.value is None
