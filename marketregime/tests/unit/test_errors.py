"""Unit tests for the domain error vocabulary."""

from datetime import UTC, date, datetime

import pytest

from regime_engine.domain.errors import (
    CalendarCoverageError,
    ConfigurationError,
    PointInTimeViolationError,
    RegimeEngineError,
    SnapshotValidationError,
)


def test_point_in_time_violation_error_message_and_attrs() -> None:
    available = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    as_of = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    err = PointInTimeViolationError("spx_close", available, as_of)

    assert isinstance(err, RegimeEngineError)
    assert err.observation_name == "spx_close"
    assert err.available_time == available
    assert err.as_of == as_of
    assert "spx_close" in str(err)
    assert available.isoformat() in str(err)
    assert as_of.isoformat() in str(err)


def test_calendar_coverage_error_message_and_attrs() -> None:
    day = date(2027, 1, 1)
    err = CalendarCoverageError(day)

    assert isinstance(err, RegimeEngineError)
    assert err.day == day
    assert "2027-01-01" in str(err)


def test_snapshot_validation_and_configuration_errors_are_regime_engine_errors() -> None:
    assert issubclass(SnapshotValidationError, RegimeEngineError)
    assert issubclass(ConfigurationError, RegimeEngineError)


def test_errors_are_raisable_and_catchable_by_base_class() -> None:
    with pytest.raises(RegimeEngineError):
        raise CalendarCoverageError(date(2020, 1, 1))
