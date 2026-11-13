"""Hand-computed checks for the EMA, RSI, true range, ATR, and VWAP primitives."""

import numpy as np
import pytest

from regime_engine.features.intraday_structure import session_vwap
from regime_engine.features.realized_vol import atr_last, true_range
from regime_engine.features.trend import ema_last, rsi_last


def _a(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_ema_seeds_with_sma_then_recurses() -> None:
    # seed mean(1, 2, 3) = 2; alpha = 0.5 -> 3 -> 4
    assert ema_last(_a(1, 2, 3, 4, 5), 3) == pytest.approx(4.0)


def test_ema_none_until_warm() -> None:
    assert ema_last(_a(1, 2), 3) is None
    assert ema_last(_a(1, 2, 3), 3) == pytest.approx(2.0)


def test_rsi_wilder_hand_computed() -> None:
    # changes +1, -1, +1; seed gain 0.5 / loss 0.5; then gain 0.75, loss 0.25 -> RS 3
    assert rsi_last(_a(1, 2, 1, 2), 2) == pytest.approx(75.0)


@pytest.mark.parametrize(
    ("closes", "expected"),
    [((1, 2, 3, 4), 100.0), ((4, 3, 2, 1), 0.0), ((2, 2, 2, 2), 50.0)],
)
def test_rsi_edges(closes: tuple[float, ...], expected: float) -> None:
    assert rsi_last(_a(*closes), 3) == pytest.approx(expected)


def test_rsi_needs_period_plus_one() -> None:
    assert rsi_last(_a(1, 2, 3), 3) is None


def test_true_range_uses_prior_close_only() -> None:
    ranges = true_range(_a(2, 5, 3), _a(1, 4, 2), _a(1.5, 4.5, 2.5))
    # bar 0: 1; bar 1 gaps up: |5 - 1.5| = 3.5; bar 2 gaps down: |2 - 4.5| = 2.5
    np.testing.assert_allclose(ranges, [1.0, 3.5, 2.5])


def test_atr_wilder_hand_computed() -> None:
    # seed mean(1, 2) = 1.5; (1.5 + 3) / 2 = 2.25; (2.25 + 4) / 2 = 3.125
    assert atr_last(_a(1, 2, 3, 4), 2) == pytest.approx(3.125)
    assert atr_last(_a(1), 2) is None


def test_vwap_weights_typical_price() -> None:
    # typical 2 x vol 1 and 5 x vol 3 -> 17 / 4
    vwap = session_vwap(_a(3, 6), _a(1, 4), _a(2, 5), _a(1, 3))
    assert vwap == pytest.approx(4.25)


def test_vwap_undefined_without_volume() -> None:
    assert session_vwap(_a(3), _a(1), _a(2), _a(0)) is None


@pytest.mark.parametrize("fn", [ema_last, rsi_last])
def test_invalid_period_rejected(fn: object) -> None:
    with pytest.raises(ValueError, match="period"):
        fn(_a(1, 2), 0)  # type: ignore[operator]
