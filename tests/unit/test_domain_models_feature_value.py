"""Unit tests for FeatureValue and PriceBar."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from regime_engine.domain.enums import MarketSession, Quality
from regime_engine.domain.models import FeatureValue, PriceBar

AS_OF = datetime(2026, 1, 5, 10, 45, tzinfo=UTC)


def _make_feature(**overrides: object) -> FeatureValue:
    fields: dict[str, object] = {
        "name": "trend_20d",
        "value": 12.5,
        "unit": "score",
        "window": "20d",
        "as_of": AS_OF,
        "quality": Quality.GOOD,
        "inputs": ("spx_close",),
        "calculation_version": "0.1.0",
    }
    fields.update(overrides)
    return FeatureValue.model_validate(fields)


def test_construct_valid_feature_value() -> None:
    feature = _make_feature()
    assert feature.value == 12.5
    assert feature.inputs == ("spx_close",)


def test_feature_value_is_immutable() -> None:
    feature = _make_feature()
    with pytest.raises(ValidationError):
        feature.value = 0.0  # type: ignore[misc]


def test_feature_value_inputs_tuple_defaults_empty() -> None:
    feature = _make_feature(inputs=())
    assert feature.inputs == ()


def test_feature_value_accepts_str_and_bool_values() -> None:
    assert _make_feature(value="STRONG_BULL").value == "STRONG_BULL"
    assert _make_feature(value=True).value is True


def test_feature_value_rejects_naive_as_of() -> None:
    with pytest.raises(ValidationError):
        _make_feature(as_of=datetime(2026, 1, 5, 10, 45))


def _make_price_bar(**overrides: object) -> PriceBar:
    fields: dict[str, object] = {
        "symbol": "SPX",
        "interval": "1d",
        "session": MarketSession.REGULAR,
        "timestamp": AS_OF,
        "open": 5000.0,
        "high": 5050.0,
        "low": 4980.0,
        "close": 5010.0,
        "volume": 1_000_000,
    }
    fields.update(overrides)
    return PriceBar.model_validate(fields)


def test_construct_valid_price_bar() -> None:
    bar = _make_price_bar()
    assert bar.low <= bar.open <= bar.high
    assert bar.low <= bar.close <= bar.high


def test_price_bar_rejects_open_outside_high_low_range() -> None:
    with pytest.raises(ValidationError):
        _make_price_bar(open=5100.0)


def test_price_bar_rejects_close_outside_high_low_range() -> None:
    with pytest.raises(ValidationError):
        _make_price_bar(close=4900.0)


def test_price_bar_allows_missing_volume() -> None:
    bar = _make_price_bar(volume=None)
    assert bar.volume is None
