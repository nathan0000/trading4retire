# Phase 10 factor suitability and attribution

Phase 10 adds a versioned, rule-based iron-condor suitability score and
descriptive P&L attribution. It does not optimize thresholds, filter trades, or
claim that a factor caused an outcome.

## Scoring contract

`FactorScoreEngine` produces five integer components bounded to `[-2, 2]`:

- `trend_score`: rewards range conditions and penalizes upward and downward
  trends symmetrically.
- `volatility_score`: rewards configurable moderate realized volatility and
  controlled range expansion.
- `sr_score`: measures the nearer selected short strike's distance from SPX in
  expected-move units.
- `momentum_score`: rewards neutral momentum and penalizes strong directional
  momentum and opening-range breakouts.
- `gamma_score`: uses only explicitly estimated gamma exposure. Missing data is
  neutral; the score must not be described as observable dealer positioning.

The unweighted `factor_score` is the sum and is bounded to `[-10, 10]`. It is a
measure of iron-condor suitability, not bullish direction. Default thresholds
are transparent hypotheses under version `ic_suitability_v1`, not values fitted
to backtest outcomes. `FactorScoreSettings` makes each threshold configurable
for later training-only sensitivity work.

## Point-in-time and lifecycle design

Every scored row requires timezone-aware `timestamp` and
`data_cutoff_timestamp` values, and rejects a cutoff later than the score time.
Rows are scored independently, so appending future observations cannot change a
prior result.

`TradeCandidate` retains an immutable copy of the entry-time feature context.
After strike selection, `enrich_candidate()` adds the actual selected short
strikes, component scores, total score, and score-version identifier to that
snapshot. Normalized completed-trade records expose these fields for analysis.
The strategy itself does not invoke scoring or reject candidates based on it.

## Attribution

`factor_performance()` groups completed trade performance by total score, each
component, and entry regime using the established Phase 8 metrics. It requires
all attribution columns and does not silently fabricate absent scores. These
tables are descriptive and must keep in-sample, validation, and out-of-sample
results separate once Phase 11 walk-forward support exists.

## Bias controls and limitations

- No future-row or full-day aggregation is performed.
- Score thresholds are not optimized in Phase 10.
- Estimated gamma exposure remains clearly labeled in the input data contract.
- Missing factor inputs are neutral only where the absence has an explicit
  meaning; missing required columns are rejected.
- Performance attribution is association, not causal evidence.
