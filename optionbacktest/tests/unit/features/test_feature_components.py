from __future__ import annotations

import numpy as np

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.features import FeatureEngine

from .test_feature_engine import make_bars


def test_returns_use_exact_prior_bar_counts() -> None:
    result = FeatureEngine(AppConfig()).compute(make_bars(14))
    assert np.isnan(result.iloc[0]["return_5m"])
    assert result.iloc[3]["return_15m"] == result.iloc[3]["close"] / result.iloc[0]["close"] - 1
    assert result.iloc[6]["return_30m"] == result.iloc[6]["close"] / result.iloc[0]["close"] - 1
    assert result.iloc[12]["return_60m"] == result.iloc[12]["close"] / result.iloc[0]["close"] - 1


def test_consecutive_bar_counts_are_causal() -> None:
    bars = make_bars(5)
    bars["close"] = [6001, 6002, 6001, 6000, 6002]
    bars["open"] = bars["close"] - [1, 1, -1, -1, 1]
    bars["high"] = bars[["open", "close"]].max(axis=1) + 1
    bars["low"] = bars[["open", "close"]].min(axis=1) - 1
    result = FeatureEngine(AppConfig()).compute(bars)
    assert result["consecutive_up_bars"].tolist() == [1, 2, 0, 0, 1]
    assert result["consecutive_down_bars"].tolist() == [0, 0, 1, 2, 0]


def test_option_positioning_columns_are_explicitly_unavailable_without_chain() -> None:
    result = FeatureEngine(AppConfig()).compute(make_bars(2))
    columns = [
        "put_call_volume_ratio",
        "put_call_open_interest_ratio",
        "atm_iv",
        "otm_put_iv",
        "otm_call_iv",
        "iv_skew",
        "estimated_gamma_exposure",
    ]
    assert result[columns].isna().all().all()
