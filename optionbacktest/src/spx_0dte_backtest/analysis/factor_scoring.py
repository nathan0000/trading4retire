"""Configurable, point-in-time iron-condor suitability scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from spx_0dte_backtest.strategies import TradeCandidate


@dataclass(frozen=True, slots=True)
class FactorScore:
    """Five bounded suitability components and their unweighted total."""

    trend_score: int
    volatility_score: int
    sr_score: int
    momentum_score: int
    gamma_score: int

    def __post_init__(self) -> None:
        for name in (
            "trend_score",
            "volatility_score",
            "sr_score",
            "momentum_score",
            "gamma_score",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not -2 <= value <= 2:
                raise ValueError(f"{name} must be an integer from -2 to 2")

    @property
    def factor_score(self) -> int:
        return (
            self.trend_score
            + self.volatility_score
            + self.sr_score
            + self.momentum_score
            + self.gamma_score
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "trend_score": self.trend_score,
            "volatility_score": self.volatility_score,
            "sr_score": self.sr_score,
            "momentum_score": self.momentum_score,
            "gamma_score": self.gamma_score,
            "factor_score": self.factor_score,
        }


@dataclass(frozen=True, slots=True)
class FactorScoreSettings:
    """Versioned rule thresholds; defaults are hypotheses, not optimized values."""

    version: str = "ic_suitability_v1"
    moderate_realized_vol_min: float = 0.10
    moderate_realized_vol_max: float = 0.25
    high_realized_vol: float = 0.30
    favorable_range_expansion_max: float = 1.20
    high_range_expansion: float = 1.50
    strong_strike_buffer: float = 1.25
    adequate_strike_buffer: float = 1.00
    marginal_strike_buffer: float = 0.75
    poor_strike_buffer: float = 0.50
    moderate_gamma_exposure: float = 100_000_000.0
    high_gamma_exposure: float = 1_000_000_000.0

    def __post_init__(self) -> None:
        numeric = (
            self.moderate_realized_vol_min,
            self.moderate_realized_vol_max,
            self.high_realized_vol,
            self.favorable_range_expansion_max,
            self.high_range_expansion,
            self.strong_strike_buffer,
            self.adequate_strike_buffer,
            self.marginal_strike_buffer,
            self.poor_strike_buffer,
            self.moderate_gamma_exposure,
            self.high_gamma_exposure,
        )
        if not self.version.strip():
            raise ValueError("score version cannot be empty")
        if not all(isfinite(value) and value >= 0 for value in numeric):
            raise ValueError("factor score thresholds must be finite and non-negative")
        if not (
            self.moderate_realized_vol_min
            <= self.moderate_realized_vol_max
            <= self.high_realized_vol
        ):
            raise ValueError("realized-volatility thresholds must be ordered")
        if self.favorable_range_expansion_max > self.high_range_expansion:
            raise ValueError("range-expansion thresholds must be ordered")
        if not (
            self.strong_strike_buffer
            >= self.adequate_strike_buffer
            >= self.marginal_strike_buffer
            >= self.poor_strike_buffer
        ):
            raise ValueError("strike-buffer thresholds must be ordered")
        if self.moderate_gamma_exposure > self.high_gamma_exposure:
            raise ValueError("gamma-exposure thresholds must be ordered")


class FactorScoreEngine:
    """Score iron-condor suitability using one timestamp's data only."""

    SCORE_COLUMNS = tuple(FactorScore.__dataclass_fields__) + ("factor_score",)
    REQUIRED_COLUMNS = {
        "timestamp",
        "data_cutoff_timestamp",
        "regime",
        "realized_vol_30m",
        "range_expansion",
        "momentum_classification",
        "opening_range_breakout",
        "close",
        "expected_move",
        "short_put_strike",
        "short_call_strike",
        "estimated_gamma_exposure",
    }

    def __init__(self, settings: FactorScoreSettings | None = None) -> None:
        self.settings = settings or FactorScoreSettings()

    def score_row(self, row: pd.Series) -> FactorScore:
        self._validate_row(row)
        return FactorScore(
            trend_score=self._trend(str(row["regime"])),
            volatility_score=self._volatility(
                self._number(row["realized_vol_30m"]),
                self._number(row["range_expansion"]),
            ),
            sr_score=self._support_resistance(row),
            momentum_score=self._momentum(
                str(row["momentum_classification"]),
                str(row["opening_range_breakout"]),
            ),
            gamma_score=self._gamma(row["estimated_gamma_exposure"]),
        )

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a scored copy; rows are independent to prevent look-ahead."""

        missing = self.REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        result = frame.copy(deep=True)
        scores = pd.DataFrame(
            [self.score_row(row).as_dict() for _, row in result.iterrows()],
            index=result.index,
        )
        for column in self.SCORE_COLUMNS:
            result[column] = scores[column].astype("int64")
        return result

    def enrich_candidate(self, candidate: TradeCandidate) -> TradeCandidate:
        """Attach scores after strike selection while preserving entry-time context."""

        from spx_0dte_backtest.strategies import TradeLegRole

        features: dict[str, Any] = dict(candidate.context_features)
        strikes = {leg.role: leg.quote.strike for leg in candidate.legs}
        features.setdefault("timestamp", candidate.timestamp)
        features.setdefault("data_cutoff_timestamp", candidate.timestamp)
        features.setdefault("close", candidate.spx_price)
        features["short_put_strike"] = strikes.get(TradeLegRole.SHORT_PUT)
        features["short_call_strike"] = strikes.get(TradeLegRole.SHORT_CALL)
        expected_move = self._number(features.get("expected_move"))
        close = self._number(features.get("close"))
        short_put = self._number(features["short_put_strike"])
        short_call = self._number(features["short_call_strike"])
        if (
            expected_move is not None
            and expected_move > 0
            and close is not None
            and short_put is not None
            and short_call is not None
        ):
            features["expected_move_distance"] = (
                min(close - short_put, short_call - close) / expected_move
            )
        else:
            features["expected_move_distance"] = None
        score = self.score_row(pd.Series(features))
        features.update(score.as_dict())
        features["factor_score_version"] = self.settings.version
        return replace(candidate, context_features=features)

    def _validate_row(self, row: pd.Series) -> None:
        missing = self.REQUIRED_COLUMNS.difference(row.index)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        timestamp = pd.Timestamp(row["timestamp"])
        cutoff = pd.Timestamp(row["data_cutoff_timestamp"])
        if timestamp.tzinfo is None or cutoff.tzinfo is None:
            raise ValueError("timestamp and data cutoff must be timezone-aware")
        if cutoff > timestamp:
            raise ValueError("data cutoff cannot be later than scoring timestamp")

    @staticmethod
    def _number(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        return number if isfinite(number) else None

    @staticmethod
    def _trend(regime: str) -> int:
        return {
            "RANGE": 2,
            "LOW_VOL": 1,
            "UNKNOWN": 0,
            "TREND_UP": -1,
            "TREND_DOWN": -1,
            "HIGH_VOL_EXPANSION": -2,
        }.get(regime, 0)

    def _volatility(self, realized_vol: float | None, expansion: float | None) -> int:
        settings = self.settings
        if (
            realized_vol is not None
            and realized_vol >= settings.high_realized_vol
            or expansion is not None
            and expansion >= settings.high_range_expansion
        ):
            return -2
        if realized_vol is None or expansion is None:
            return 0
        if (
            settings.moderate_realized_vol_min <= realized_vol <= settings.moderate_realized_vol_max
            and expansion <= settings.favorable_range_expansion_max
        ):
            return 2
        if realized_vol < settings.moderate_realized_vol_min:
            return 1
        return -1 if realized_vol > settings.moderate_realized_vol_max else 0

    def _support_resistance(self, row: pd.Series) -> int:
        close = self._number(row["close"])
        expected_move = self._number(row["expected_move"])
        put = self._number(row["short_put_strike"])
        call = self._number(row["short_call_strike"])
        if None in {close, expected_move, put, call} or expected_move == 0:
            return 0
        assert (
            close is not None and expected_move is not None and put is not None and call is not None
        )
        buffer = min(close - put, call - close) / expected_move
        settings = self.settings
        if buffer >= settings.strong_strike_buffer:
            return 2
        if buffer >= settings.adequate_strike_buffer:
            return 1
        if buffer >= settings.marginal_strike_buffer:
            return 0
        if buffer >= settings.poor_strike_buffer:
            return -1
        return -2

    @staticmethod
    def _momentum(momentum: str, breakout: str) -> int:
        score = {
            "NEUTRAL": 2,
            "UP": 0,
            "DOWN": 0,
            "STRONG_UP": -2,
            "STRONG_DOWN": -2,
        }.get(momentum, 0)
        if breakout not in {"INSIDE", "NONE", "NEUTRAL"}:
            score -= 1
        return max(-2, min(2, score))

    def _gamma(self, value: object) -> int:
        gamma = self._number(value)
        if gamma is None:
            return 0
        magnitude = abs(gamma)
        if magnitude >= self.settings.high_gamma_exposure:
            return 2 if gamma > 0 else -2
        if magnitude >= self.settings.moderate_gamma_exposure:
            return 1 if gamma > 0 else -1
        return 0
