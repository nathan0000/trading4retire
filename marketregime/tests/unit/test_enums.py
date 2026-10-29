"""Unit tests for the regime taxonomy enums (CLAUDE.md section 6)."""

from regime_engine.domain.enums import (
    CalendarDayStatus,
    CompositeRegimeLabel,
    Direction,
    Gamma,
    Liquidity,
    MarketSession,
    Quality,
    Stability,
    Volatility,
)


def test_direction_values_match_spec() -> None:
    assert [member.value for member in Direction] == [
        "STRONG_BEAR",
        "BEAR",
        "NEUTRAL",
        "BULL",
        "STRONG_BULL",
        "UNKNOWN",
    ]


def test_volatility_values_match_spec() -> None:
    assert [member.value for member in Volatility] == [
        "COMPRESSED",
        "NORMAL",
        "EXPANDING",
        "STRESS",
        "UNKNOWN",
    ]


def test_gamma_values_match_spec() -> None:
    assert [member.value for member in Gamma] == ["POSITIVE", "NEUTRAL", "NEGATIVE", "UNKNOWN"]


def test_liquidity_values_match_spec() -> None:
    assert [member.value for member in Liquidity] == [
        "NORMAL",
        "THIN",
        "DISLOCATED",
        "UNKNOWN",
    ]


def test_stability_values_match_spec() -> None:
    assert [member.value for member in Stability] == [
        "STABLE",
        "TRANSITION_RISK",
        "TRANSITIONING",
        "UNKNOWN",
    ]


def test_quality_values_match_spec() -> None:
    assert [member.value for member in Quality] == [
        "GOOD",
        "DEGRADED",
        "STALE",
        "MISSING",
        "SUSPECT",
    ]


def test_market_session_values() -> None:
    assert [member.value for member in MarketSession] == [
        "PRE_MARKET",
        "REGULAR",
        "AFTER_HOURS",
        "CLOSED",
    ]


def test_calendar_day_status_values() -> None:
    assert [member.value for member in CalendarDayStatus] == ["OPEN", "CLOSED", "EARLY_CLOSE"]


def test_composite_regime_label_values() -> None:
    assert CompositeRegimeLabel.MARKET_STRESS == "MARKET_STRESS"
    assert CompositeRegimeLabel.UNKNOWN == "UNKNOWN"


def test_enums_serialize_as_plain_strings() -> None:
    # StrEnum members compare/format as their plain string values at boundaries.
    assert f"{Direction.BULL}" == "BULL"
    assert str(Direction.BULL) == "BULL"
