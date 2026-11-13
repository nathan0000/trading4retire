"""Standalone market-regime classifier.

Tags every bar timestamp with a trend regime and a volatility regime using only
data known at that timestamp. The resulting ``RegimeTimeline`` is queried with
``at(timestamp)``, which returns the latest state at or before the timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from options_backtester.regimes import indicators


class Trend(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class Volatility(StrEnum):
    LOW = "LOW_IV"
    NORMAL = "NORMAL_IV"
    HIGH = "HIGH_IV"
    UNKNOWN = "UNKNOWN"


class VolDynamics(StrEnum):
    EXPANSION = "EXPANSION"
    CONTRACTION = "CONTRACTION"
    STABLE = "STABLE"
    UNKNOWN = "UNKNOWN"


class TermStructure(StrEnum):
    CONTANGO = "CONTANGO"
    BACKWARDATION = "BACKWARDATION"
    UNKNOWN = "UNKNOWN"


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fast_sma: int = Field(default=50, ge=2)
    slow_sma: int = Field(default=200, ge=3)
    adx_window: int = Field(default=14, ge=2)
    #: Below this ADX the market is tagged SIDEWAYS regardless of the moving averages.
    adx_trend_min: float = Field(default=20.0, ge=0)
    realized_window: int = Field(default=20, ge=5)
    rank_window: int = Field(default=252, ge=20)
    rank_min_periods: int = Field(default=60, ge=5)
    low_rank: float = Field(default=25.0, ge=0, le=100)
    high_rank: float = Field(default=60.0, ge=0, le=100)
    #: Absolute VIX levels; applied only when a real implied-vol index is supplied.
    low_vix: float | None = 13.0
    high_vix: float | None = 25.0
    dynamics_window: int = Field(default=5, ge=1)
    dynamics_threshold: float = Field(default=0.15, gt=0)


@dataclass(frozen=True, slots=True)
class RegimeState:
    timestamp: pd.Timestamp
    trend: str
    volatility: str
    vol_dynamics: str
    term_structure: str
    iv_rank: float | None = None
    iv_percentile: float | None = None
    vol_level: float | None = None
    adx: float | None = None

    @property
    def label(self) -> str:
        return f"{self.trend}|{self.volatility}"

    @property
    def tags(self) -> frozenset[str]:
        return frozenset({self.trend, self.volatility, self.vol_dynamics, self.term_structure})


class RegimeTimeline:
    """Time-indexed regime states with a strict at-or-before lookup."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.sort_index()
        self._stamps = self.frame.index.as_unit("ns").asi8

    def at(self, timestamp: pd.Timestamp) -> RegimeState | None:
        stamp = pd.Timestamp(timestamp).tz_convert("UTC").value
        position = int(np.searchsorted(self._stamps, stamp, side="right")) - 1
        if position < 0:
            return None
        row = self.frame.iloc[position]

        def number(name: str) -> float | None:
            return None if pd.isna(row[name]) else float(row[name])

        return RegimeState(
            timestamp=self.frame.index[position],
            trend=row["trend"],
            volatility=row["volatility"],
            vol_dynamics=row["vol_dynamics"],
            term_structure=row["term_structure"],
            iv_rank=number("iv_rank"),
            iv_percentile=number("iv_percentile"),
            vol_level=number("vol_level"),
            adx=number("adx"),
        )


class RegimeClassifier:
    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def classify(
        self,
        bars: pd.DataFrame,
        vix: pd.DataFrame | None = None,
        vix_back: pd.DataFrame | None = None,
    ) -> RegimeTimeline:
        """Tag every bar. ``vix`` / ``vix_back`` are canonical bars of a front and a
        longer-dated implied-vol index (e.g. VIX and VIX3M). Without ``vix`` the
        volatility regime uses trailing realized volatility as a labelled proxy.
        """
        cfg = self.config
        frame = bars.sort_values("timestamp").set_index("timestamp")
        close, high, low = frame["close"], frame["high"], frame["low"]

        fast, slow = indicators.sma(close, cfg.fast_sma), indicators.sma(close, cfg.slow_sma)
        strength = indicators.adx(high, low, close, cfg.adx_window)
        ready = slow.notna() & strength.notna()
        trending = strength >= cfg.adx_trend_min
        trend = np.select(
            [
                ~ready,
                trending & (close > slow) & (fast > slow),
                trending & (close < slow) & (fast < slow),
            ],
            [Trend.UNKNOWN.value, Trend.BULLISH.value, Trend.BEARISH.value],
            default=Trend.SIDEWAYS.value,
        )

        implied = _asof(vix, frame.index)
        has_index = implied is not None
        level = (
            implied
            if has_index
            else indicators.realized_volatility(close, cfg.realized_window) * 100.0
        )
        rank = indicators.iv_rank(level, cfg.rank_window, cfg.rank_min_periods)
        percentile = indicators.iv_percentile(level, cfg.rank_window, cfg.rank_min_periods)
        high_vol = rank > cfg.high_rank
        low_vol = rank < cfg.low_rank
        if has_index and cfg.high_vix is not None:
            high_vol = high_vol | (level >= cfg.high_vix)
        if has_index and cfg.low_vix is not None:
            low_vol = low_vol | (level <= cfg.low_vix)
        volatility = np.select(
            [rank.isna(), high_vol, low_vol],
            [Volatility.UNKNOWN.value, Volatility.HIGH.value, Volatility.LOW.value],
            default=Volatility.NORMAL.value,
        )

        change = level / level.shift(cfg.dynamics_window) - 1.0
        dynamics = np.select(
            [change.isna(), change > cfg.dynamics_threshold, change < -cfg.dynamics_threshold],
            [VolDynamics.UNKNOWN.value, VolDynamics.EXPANSION.value, VolDynamics.CONTRACTION.value],
            default=VolDynamics.STABLE.value,
        )

        back = _asof(vix_back, frame.index)
        if has_index and back is not None:
            ratio = indicators.term_structure_ratio(level, back)
            structure = np.select(
                [ratio.isna(), ratio > 1.0],
                [TermStructure.UNKNOWN.value, TermStructure.BACKWARDATION.value],
                default=TermStructure.CONTANGO.value,
            )
        else:
            structure = np.full(len(frame), TermStructure.UNKNOWN.value)

        return RegimeTimeline(
            pd.DataFrame(
                {
                    "trend": trend,
                    "volatility": volatility,
                    "vol_dynamics": dynamics,
                    "term_structure": structure,
                    "iv_rank": rank,
                    "iv_percentile": percentile,
                    "vol_level": level,
                    "vol_source": "IMPLIED_INDEX" if has_index else "REALIZED_PROXY",
                    "adx": strength,
                },
                index=frame.index,
            )
        )


def _asof(source: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series | None:
    """Latest ``close`` of ``source`` known at or before each index timestamp."""
    if source is None or source.empty:
        return None
    series = source.sort_values("timestamp").set_index("timestamp")["close"]
    return series.reindex(series.index.union(index)).ffill().reindex(index)
