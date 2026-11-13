# Phase 2 point-in-time features

Features reset by New York trading session. Intraday rolling state, EMA, VWAP, ATR,
ADX, opening range, ranges, returns, and consecutive-bar counts cannot cross session
boundaries. Previous-day levels are shifted summaries of completed earlier sessions.

The opening range expands during its configured window and freezes afterward. VIX uses
a backward as-of join, and expected move treats VIX as percentage points:

```text
SPX * (VIX / 100) / sqrt(252)
```

Every output row is labelled by the time its data is known, and records that time as
`data_cutoff_timestamp`. With `timestamp_semantics: bar_start`, a bar's close is only
known when the bar ends, so `timestamp` is moved to the bar end and the source label is
kept in `bar_start`. A 10:45 entry therefore uses the 10:40–10:45 bar, never the bar
that is still forming. Prefix-invariance
tests prove later SPX rows cannot alter an earlier row. Feature windows and rule-based
classification thresholds are held in immutable, injectable `FeatureSettings`.

VWAP is returned only when every observation in the session has a known volume and at
least one has positive volume. This prevents a partly missing SPX volume series from
silently producing a misleading value.

SPX cash bars cannot provide overnight levels and Phase 2 has no historical option
chains. Overnight levels and option-positioning outputs, including
`estimated_gamma_exposure`, are explicitly null rather than fabricated. Their
calculation belongs after the relevant data contracts and loaders are available.

Trend and momentum classifications are descriptive, configurable rule-based labels;
they are not strategy suitability scores and have not been optimized.
