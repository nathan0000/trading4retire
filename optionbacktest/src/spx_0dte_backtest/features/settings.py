"""Immutable, configurable Phase 2 feature parameters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    ema_span: int = 12
    adx_period: int = 14
    atr_period: int = 14
    annualization_periods: int = 19_656  # 78 five-minute bars * 252 sessions
    opening_range_minutes: int = 30
    swing_window_bars: int = 12
    slope_bars: int = 3
    trend_strong_threshold: float = 1.0
    trend_threshold: float = 0.25
    momentum_strong_threshold: float = 1.0
    momentum_threshold: float = 0.25

    def __post_init__(self) -> None:
        integer_fields = (
            self.ema_span,
            self.adx_period,
            self.atr_period,
            self.annualization_periods,
            self.opening_range_minutes,
            self.swing_window_bars,
            self.slope_bars,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("feature windows and annualization periods must be positive")
        if self.trend_strong_threshold <= self.trend_threshold or self.trend_threshold <= 0:
            raise ValueError("trend thresholds must be positive and ordered")
        if (
            self.momentum_strong_threshold <= self.momentum_threshold
            or self.momentum_threshold <= 0
        ):
            raise ValueError("momentum thresholds must be positive and ordered")
