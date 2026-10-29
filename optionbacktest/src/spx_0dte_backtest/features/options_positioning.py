"""Option-positioning placeholders pending historical chains in Phase 3."""

import numpy as np
import pandas as pd

UNAVAILABLE_POSITIONING_COLUMNS = (
    "put_call_volume_ratio",
    "put_call_open_interest_ratio",
    "atm_iv",
    "implied_volatility",
    "iv_rv_ratio",
    "otm_put_iv",
    "otm_call_iv",
    "iv_skew",
    "estimated_gamma_exposure",
)


def add_unavailable_positioning(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in UNAVAILABLE_POSITIONING_COLUMNS:
        result[column] = np.nan
    return result
