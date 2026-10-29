"""Configurable credit-spread stop and profit-target thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class RiskSettings:
    stop_multiplier: float = 0.90
    profit_target_fraction: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.stop_multiplier) or self.stop_multiplier < 0:
            raise ValueError("stop multiplier must be finite and non-negative")
        if self.profit_target_fraction is not None and (
            not isfinite(self.profit_target_fraction) or not 0 <= self.profit_target_fraction <= 1
        ):
            raise ValueError("profit target fraction must be between zero and one")

    def stop_loss_amount(self, initial_credit: float) -> float:
        self._validate_credit(initial_credit)
        return initial_credit * self.stop_multiplier

    def stop_loss_price(self, initial_credit: float) -> float:
        """Cost to close: entry credit plus the configured allowed loss."""

        return initial_credit + self.stop_loss_amount(initial_credit)

    def profit_target_price(self, initial_credit: float) -> float | None:
        self._validate_credit(initial_credit)
        if self.profit_target_fraction is None:
            return None
        return initial_credit * (1 - self.profit_target_fraction)

    @staticmethod
    def _validate_credit(initial_credit: float) -> None:
        if not isfinite(initial_credit) or initial_credit <= 0:
            raise ValueError("initial credit must be finite and positive")
