import pytest

from spx_0dte_backtest.risk import RiskSettings


@pytest.mark.parametrize("multiplier", [0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50])
def test_stop_price_is_credit_plus_allowed_loss(multiplier: float) -> None:
    settings = RiskSettings(stop_multiplier=multiplier)
    assert settings.stop_loss_amount(2.0) == pytest.approx(2.0 * multiplier)
    assert settings.stop_loss_price(2.0) == pytest.approx(2.0 * (1 + multiplier))


def test_example_stop_price_is_three_eighty() -> None:
    assert RiskSettings(stop_multiplier=0.90).stop_loss_price(2.0) == pytest.approx(3.80)


def test_profit_target_is_optional_and_uses_entry_credit() -> None:
    assert RiskSettings().profit_target_price(2.0) is None
    assert RiskSettings(profit_target_fraction=0.50).profit_target_price(2.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stop_multiplier": -0.1},
        {"profit_target_fraction": -0.1},
        {"profit_target_fraction": 1.1},
    ],
)
def test_invalid_risk_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        RiskSettings(**kwargs)
