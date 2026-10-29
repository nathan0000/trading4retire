"""Configurable, row-local market regime engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np
import pandas as pd


class MarketRegime(StrEnum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOL_EXPANSION = "HIGH_VOL_EXPANSION"
    LOW_VOL = "LOW_VOL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RegimeSettings:
    """Unoptimized rule thresholds; callers may replace every boundary."""

    version: str = "rule_based_v1"
    trend_return_threshold: float = 0.0025
    trend_adx_min: float = 20.0
    trend_vwap_slope_min: float = 0.0
    range_adx_max: float = 18.0
    range_distance_atr_max: float = 0.50
    range_return_abs_max: float = 0.0010
    high_vol_range_expansion_min: float = 1.50
    high_vol_realized_vol_min: float = 0.20
    low_vol_realized_vol_max: float = 0.10
    low_vol_range_expansion_max: float = 1.00

    def __post_init__(self) -> None:
        non_negative = (
            self.trend_return_threshold,
            self.trend_adx_min,
            self.trend_vwap_slope_min,
            self.range_adx_max,
            self.range_distance_atr_max,
            self.range_return_abs_max,
            self.low_vol_realized_vol_max,
            self.low_vol_range_expansion_max,
        )
        if not self.version.strip():
            raise ValueError("regime version must not be empty")
        if not all(isfinite(value) and value >= 0 for value in non_negative):
            raise ValueError("regime thresholds must be finite and non-negative")
        if (
            not isfinite(self.high_vol_range_expansion_min)
            or self.high_vol_range_expansion_min <= 0
        ):
            raise ValueError("high-vol range expansion threshold must be positive")
        if not isfinite(self.high_vol_realized_vol_min) or self.high_vol_realized_vol_min < 0:
            raise ValueError("high-vol realized volatility threshold must be non-negative")


class RegimeEngine:
    """Classify each feature row using only values carried by that row.

    Precedence is explicit: missing data, high-vol expansion, trend, range,
    low-vol, then unknown. No rolling or future operation occurs in this module.
    """

    REQUIRED_COLUMNS = {
        "timestamp",
        "data_cutoff_timestamp",
        "close",
        "vwap",
        "vwap_slope",
        "distance_from_vwap",
        "atr",
        "adx",
        "return_30m",
        "range_expansion",
        "realized_vol_30m",
    }

    SIGNAL_COLUMNS = REQUIRED_COLUMNS - {"timestamp", "data_cutoff_timestamp"}

    def __init__(self, settings: RegimeSettings | None = None):
        self.settings = settings or RegimeSettings()

    def classify(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS.difference(features.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        result = features.copy(deep=True)
        if result.empty:
            result["regime"] = pd.Series(dtype="string")
            result["regime_reason"] = pd.Series(dtype="string")
            result["regime_version"] = pd.Series(dtype="string")
            return result
        for column in ("timestamp", "data_cutoff_timestamp"):
            if not pd.api.types.is_datetime64_any_dtype(result[column]):
                raise ValueError(f"{column} must be datetime data")
            if result[column].dt.tz is None:
                raise ValueError(f"{column} must be timezone-aware")
        if (result["data_cutoff_timestamp"] > result["timestamp"]).any():
            raise ValueError("feature cutoff cannot be later than its regime timestamp")

        classifications = result.apply(self._classify_row, axis=1, result_type="expand")
        classifications.columns = ["regime", "regime_reason"]
        result["regime"] = classifications["regime"].astype("string")
        result["regime_reason"] = classifications["regime_reason"].astype("string")
        result["regime_version"] = self.settings.version
        return result

    def _classify_row(self, row: pd.Series) -> tuple[str, str]:
        signals = pd.to_numeric(row[list(self.SIGNAL_COLUMNS)], errors="coerce")
        if signals.isna().any() or not np.isfinite(signals.astype(float)).all():
            return MarketRegime.UNKNOWN.value, "MISSING_REQUIRED_FEATURE"

        if (
            row["range_expansion"] >= self.settings.high_vol_range_expansion_min
            and row["realized_vol_30m"] >= self.settings.high_vol_realized_vol_min
        ):
            return MarketRegime.HIGH_VOL_EXPANSION.value, "RANGE_AND_REALIZED_VOL_EXPANSION"

        if (
            row["return_30m"] >= self.settings.trend_return_threshold
            and row["close"] > row["vwap"]
            and row["vwap_slope"] > self.settings.trend_vwap_slope_min
            and row["adx"] >= self.settings.trend_adx_min
        ):
            return MarketRegime.TREND_UP.value, "POSITIVE_MOMENTUM_ABOVE_RISING_VWAP"

        if (
            row["return_30m"] <= -self.settings.trend_return_threshold
            and row["close"] < row["vwap"]
            and row["vwap_slope"] < -self.settings.trend_vwap_slope_min
            and row["adx"] >= self.settings.trend_adx_min
        ):
            return MarketRegime.TREND_DOWN.value, "NEGATIVE_MOMENTUM_BELOW_FALLING_VWAP"

        distance_atr = (
            abs(row["distance_from_vwap"]) / row["atr"] if row["atr"] > 0 else float("inf")
        )
        if (
            row["adx"] <= self.settings.range_adx_max
            and distance_atr <= self.settings.range_distance_atr_max
            and abs(row["return_30m"]) <= self.settings.range_return_abs_max
        ):
            return MarketRegime.RANGE.value, "LOW_ADX_NEAR_VWAP"

        if (
            row["realized_vol_30m"] <= self.settings.low_vol_realized_vol_max
            and row["range_expansion"] <= self.settings.low_vol_range_expansion_max
        ):
            return MarketRegime.LOW_VOL.value, "LOW_REALIZED_VOL_AND_RANGE_CONTRACTION"

        return MarketRegime.UNKNOWN.value, "NO_RULE_MATCH"
