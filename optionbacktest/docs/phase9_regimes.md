# Phase 9 point-in-time regime engine

Phase 9 classifies each completed feature row independently as `TREND_UP`,
`TREND_DOWN`, `RANGE`, `HIGH_VOL_EXPANSION`, `LOW_VOL`, or `UNKNOWN`. It does not
filter trades, score suitability, optimize thresholds, or use outcome data.

The engine consumes price/VWAP relationship, VWAP slope, ATR, ADX, 30-minute return,
range expansion, and 30-minute realized volatility. It performs no rolling or
cross-row calculation. The data cutoff must be timezone-aware and no later than the
row timestamp. Missing or non-finite inputs produce `UNKNOWN`. Prefix-invariance tests
require the same result with or without later rows.

Rules use explicit precedence: incomplete data, high-vol expansion, trend, range,
low volatility, then unknown. Expansion therefore takes priority over trend, while
range takes priority over low volatility.

All boundaries live in immutable `RegimeSettings`; none are optimized or treated as
permanent definitions. Each output records `regime`, `regime_reason`, and the configured
`regime_version` (`rule_based_v1` by default). Later tuning must respect training and
validation splits.
