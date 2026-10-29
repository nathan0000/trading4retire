# Market Regime Engine — Agent Development Guide

## 1. Mission

Build a production-quality market regime engine for SPX and 0DTE options decision support. The engine must convert point-in-time market observations into an explainable, versioned regime assessment that can be consumed by any strategy engine.

The regime engine is an advisory system. It does not select strikes, size positions, submit orders, or manage positions. A separate strategy engine consumes its output and owns those decisions.

Primary decision time is 10:45 America/New_York, but every component must support arbitrary `as_of` timestamps, historical replay, and intraday updates.

## 2. Non-negotiable engineering principles

1. **Point-in-time correctness:** never use data published or revised after `as_of`.
2. **Dependency direction:** domain code depends on interfaces and domain models, never vendors, databases, brokers, dashboards, or frameworks.
3. **One canonical result:** every consumer receives the same versioned `RegimeSnapshot` contract.
4. **Separate facts from judgment:** raw observations, validated data, features, scores, classification, recommendations, and presentation remain distinct.
5. **Explain every decision:** labels and confidence must include evidence, counter-evidence, data quality, and model/rule versions.
6. **Fail safely:** stale, missing, contradictory, or suspect inputs reduce confidence or yield `UNKNOWN`; they never silently become zero or neutral.
7. **Determinism:** the same input snapshot, configuration, and code/model version must produce the same result.
8. **No trading side effects:** the regime package cannot import execution, broker order, or portfolio-management modules.
9. **Configuration over literals:** thresholds, weights, calendars, symbols, and freshness limits belong in validated configuration.
10. **Prefer simple, testable rules first:** add statistical or ML models only after a point-in-time benchmark exists.

## 3. Bounded contexts and dependency flow

```text
provider adapters -> canonical data -> validation/quality -> features
                                                       -> regime inference
                                                       -> transition inference
                                                       -> policy hints
                                                       -> snapshot store/API/events
                                                       -> reports/monitoring

strategy engine -> reads RegimeSnapshot -> applies its own eligibility/risk rules
```

Allowed dependency direction:

```text
presentation/infrastructure -> application -> domain
provider adapters ----------> ports -------> domain
```

The domain layer must be importable and testable without network access, credentials, a running database, or a broker session.

## 4. Recommended repository layout

```text
src/regime_engine/
  domain/
    enums.py                 # stable vocabulary
    models.py                # immutable domain contracts
    errors.py
    policies.py              # pure classification/policy functions
  ports/
    market_data.py           # source interfaces
    snapshot_store.py
    publisher.py
    clock.py
  adapters/
    providers/               # ThetaData, IBKR, files, fixtures
    storage/                 # parquet/sql/database implementations
    publishers/              # JSON, API, message bus
  data/
    normalization.py         # timezone, symbols, units, schemas
    validation.py
    quality.py
  features/
    trend.py
    cycles.py               # seasonal/presidential/decennial priors
    realized_vol.py
    implied_vol.py
    vol_structure.py
    options_positioning.py
    intraday_structure.py
    breadth.py
    registry.py
  inference/
    scoring.py
    classifier.py
    transitions.py
    confidence.py
  application/
    build_snapshot.py        # orchestration use case
    replay.py
  reporting/
    serializers.py
    narrative.py
    dashboard_view.py
  api/
    schemas.py
    service.py
  config/
    defaults.yaml
tests/
  unit/
  contract/
  integration/
  replay/
  fixtures/
docs/
  architecture.md
  data_dictionary.md
  regime_taxonomy.md
  decisions/
```

Do not create a generic `utils.py`. Put code in the bounded context that owns it.

## 5. Canonical domain contracts

Use immutable, typed objects. Pydantic/dataclasses are acceptable at boundaries; core calculations should remain ordinary pure functions. Store timestamps in UTC and carry the market timezone explicitly. Decimal or integer basis-point representations are preferred where float ambiguity matters.

### 5.1 Observation envelope

Every source value must retain provenance:

```python
class Observation[T]:
    name: str
    value: T | None
    event_time: datetime       # when the market event occurred
    available_time: datetime   # earliest time the system could know it
    received_time: datetime
    source: str
    source_symbol: str
    quality: str               # GOOD, DEGRADED, STALE, MISSING, SUSPECT
    metadata: dict[str, object]
```

`available_time <= as_of` is mandatory during replay and backtesting. `received_time` is for operational latency; it must not replace market event time.

### 5.2 Feature value

```python
class FeatureValue:
    name: str
    value: float | str | bool | None
    unit: str | None
    window: str
    as_of: datetime
    quality: str
    inputs: tuple[str, ...]
    calculation_version: str
```

Feature functions accept canonical data plus explicit configuration and return `FeatureValue`. They must not fetch data, write files, read environment variables, or know which strategy will consume them.

### 5.3 Regime snapshot

The public contract should contain at least:

```python
class RegimeSnapshot:
    schema_version: str
    snapshot_id: str
    as_of: datetime
    generated_at: datetime
    market_session: str
    universe: str

    direction: str             # STRONG_BEAR..STRONG_BULL, UNKNOWN
    volatility: str            # COMPRESSED, NORMAL, EXPANDING, STRESS, UNKNOWN
    liquidity: str             # NORMAL, THIN, DISLOCATED, UNKNOWN
    gamma: str                 # POSITIVE, NEUTRAL, NEGATIVE, UNKNOWN
    composite_regime: str

    trend_score: float | None          # -100..100
    volatility_score: float | None     # 0..100
    cycle_score: float | None          # -100..100; slow prior, not a trigger
    cycle_confidence: float | None     # 0..1; sample/stability adjusted
    cycle_horizon: str | None
    transition_probability: float | None
    transition_target: str | None
    confidence: float                  # 0..1, quality-adjusted

    features: dict[str, FeatureValue]
    evidence_for: tuple[str, ...]
    evidence_against: tuple[str, ...]
    data_quality: dict[str, object]
    policy_hints: tuple[str, ...]

    config_version: str
    classifier_version: str
    feature_set_version: str
    input_fingerprint: str
```

Use enumerations internally and stable string values on serialized boundaries. Additive schema changes are preferred. Breaking changes require a new `schema_version` and contract tests.

## 6. Regime taxonomy

Do not collapse regime into only bullish/bearish/sideways. Classify independent axes first:

- Direction: `STRONG_BEAR`, `BEAR`, `NEUTRAL`, `BULL`, `STRONG_BULL`, `UNKNOWN`
- Volatility: `COMPRESSED`, `NORMAL`, `EXPANDING`, `STRESS`, `UNKNOWN`
- Gamma: `POSITIVE`, `NEUTRAL`, `NEGATIVE`, `UNKNOWN`
- Liquidity: `NORMAL`, `THIN`, `DISLOCATED`, `UNKNOWN`
- Stability: `STABLE`, `TRANSITION_RISK`, `TRANSITIONING`, `UNKNOWN`

Then derive a composite label such as:

- `RANGE_POSITIVE_GAMMA`
- `BULL_TREND_NORMAL_VOL`
- `BEAR_TREND_VOL_EXPANSION`
- `HIGH_VOL_MEAN_REVERSION`
- `COMPRESSION_BREAKOUT_RISK`
- `MARKET_STRESS`
- `UNKNOWN`

Composite labels are summaries, not substitutes for axis scores.

## 7. Feature families

Implement each family as an independently registered feature set. Do not require all providers to supply all families.

### Macro/daily context

- SPX 20/50-day trend and slope
- 5/20/60-day realized volatility and percentiles
- ATR percent and volatility-of-volatility
- prior-day and multi-day market structure

### Cycle composite and calendar context

Treat cycles as a probabilistic prior derived from historical distributions, never as a deterministic forecast or a standalone entry signal. The cycle module must combine independently observable components rather than importing a vendor's opaque composite chart.

Required components:

- **one-year seasonal cycle:** trading-day-of-year or session-relative seasonal return path;
- **four-year presidential cycle:** year 1 through 4 of the U.S. presidential term, with election/inauguration boundaries documented;
- **10-year decennial cycle:** calendar year last digit 0 through 9;
- **calendar micro-cycles:** day-of-week, turn-of-month, month-end/quarter-end, options-expiration week, and pre/post U.S. market holiday effects where statistically supportable;
- **optional longer cycles:** only behind configuration and only when adequate independent samples exist.

The canonical cycle output should include:

```python
class CycleComposite:
    score: float                 # -100 bearish to +100 bullish
    expected_return: float | None
    downside_risk: float | None
    confidence: float            # 0..1
    horizon: str                 # e.g. 1d, 5d, 20d
    components: dict[str, float]
    component_weights: dict[str, float]
    sample_sizes: dict[str, int]
    stability: dict[str, float]
    estimation_end: date
    methodology_version: str
```

Construction requirements:

1. Use total-return or price-return history consistently; document the selected series.
2. Align observations by exchange trading sessions, not naive calendar-day interpolation.
3. Estimate each component only from data available before the forecast date. For backtests, refit with an expanding or rolling window whose `estimation_end < as_of.date()`.
4. Normalize component forecasts using robust z-scores or percentile ranks before combining them.
5. Shrink small or unstable samples toward zero. Weight components by out-of-sample stability, effective sample size, and correlation redundancy—not only in-sample return.
6. Prevent double counting: annual seasonality, presidential phase, and decennial phase overlap. Measure component correlation and cap correlated contributions.
7. Publish multiple horizons separately. Do not apply a monthly seasonal expectation directly to an intraday trade without horizon conversion and validation.
8. Include uncertainty, component disagreement, sample sizes, estimation window, and methodology version in evidence and reports.
9. Return neutral/low-confidence output when evidence is weak; never force a bullish or bearish cycle label.

Suggested robust composite:

```text
raw_component_i = robust_z(expected_forward_return_i)
reliability_i   = f(sample_size, walk_forward_stability, recency, uncertainty)
adjusted_i      = raw_component_i * reliability_i
cycle_score     = 100 * tanh(sum(correlation_adjusted_weight_i * adjusted_i))
```

Weights must be learned or configured from walk-forward results and sum to one after unavailable components are removed. Persist both configured and effective weights.

Cycle interaction rules:

- For daily/swing regime classification, the cycle composite may begin with a maximum family weight of 10–15% pending validation.
- For the 10:45 ET SPX 0DTE snapshot, begin with a 5–10% maximum weight and let intraday price, volatility, breadth, and gamma dominate.
- Reduce the effective cycle weight as cycle confidence falls or component disagreement rises.
- Reduce or neutralize the cycle prior during `STRESS`, material event risk, structural breaks, or strong contradictory live evidence.
- A cycle/live-signal conflict must appear in `evidence_against`; it must not be silently averaged away.
- Cycles can tilt direction/confidence and transition priors, but cannot alone change the regime to `STRESS`, approve short gamma, or generate a trade.

Cycle-derived fields belong in the feature/snapshot layer. Any mapping such as “bullish seasonal phase permits a bullish butterfly” belongs in the separate strategy policy layer.

### Implied volatility and term structure

- VIX level, change, percentile, and acceleration
- VIX9D/VIX and VIX/VIX3M ratios
- contango/backwardation state
- implied-versus-realized volatility spread

### Options positioning

- aggregate GEX/DEX with source and methodology recorded
- gamma flip/zero-gamma distance
- call wall, put wall, and concentration distances
- explicitly distinguish observed exchange/vendor fields from derived estimates

### Intraday structure

- price relative to session VWAP
- opening-range location and breaks
- overnight high/low and gap state
- 5/15/30-minute slopes, efficiency ratio, ADX where justified
- VWAP crossing count and acceptance duration
- realized range versus expected move

### Breadth and liquidity

- advance/decline, up/down volume, sector participation when available
- bid/ask width, quote age, missing quote rate, halt/anomaly indicators

Every feature documents its formula, units, warm-up requirement, expected cadence, null behavior, and point-in-time assumptions in `docs/data_dictionary.md`.

## 8. Inference and confidence

Maintain separate scores:

- `trend_score` in `[-100, 100]`
- `volatility_score` in `[0, 100]`
- `cycle_score` in `[-100, 100]` with a separately calibrated `cycle_confidence`
- `transition_probability` in `[0, 1]`

Initial configuration may use these starting family weights for the 10:45 ET decision: trend 22%, volatility 20%, options positioning 20%, volatility structure 13%, breadth 10%, intraday price structure 10%, and cycle composite 5%. The cycle weight may rise toward 10% only when confidence and walk-forward calibration justify it. These are hypotheses, not universal truths; keep them versioned and validate them out of sample.

Confidence is not the magnitude of a score. Calculate it separately using:

- completeness and freshness
- cross-source agreement
- signal agreement/contradiction
- distance from classification thresholds
- historical calibration by regime
- out-of-distribution or exceptional-event flags

Missing critical families must cap confidence. Conflicting evidence must appear in `evidence_against`. Never invent precision: round public scores consistently and calibrate probability outputs.

## 9. Transition detection

Treat transition risk as a first-class output, not a note. Candidate evidence includes:

- price crossing/rejecting gamma flip or key positioning level
- VIX9D/VIX acceleration or curve inversion
- opening-range break with VWAP acceptance
- realized-volatility expansion from compression
- breadth deterioration/improvement inconsistent with the current label
- change-point or hidden-state model output, after a rule baseline exists

Transition inference must use only observations available at `as_of`. Publish the likely target regime, horizon, probability, and top drivers when supportable. Otherwise use `None`, not fabricated certainty.

## 10. Data provider isolation

Define narrow ports such as `PriceBarsPort`, `VolatilityIndexPort`, `OptionsPositioningPort`, `BreadthPort`, and `MarketCalendarPort`. Avoid a vendor-shaped mega-interface.

Each adapter owns:

- authentication and transport
- retries, backoff, rate limits, and pagination
- vendor symbol mapping
- vendor-to-canonical schema conversion
- raw response capture where permitted
- provenance and source-specific quality flags

Adapters must never classify regimes. Core modules must never branch on provider names.

Support fixture/file adapters from the beginning so tests and replay do not require live services. ThetaData or IBKR implementations are plugins behind the same ports; the core must not require either.

## 11. Application pipeline

`BuildRegimeSnapshot(as_of, universe, config_version)` should:

1. resolve the exchange session and trading calendar;
2. request the minimum necessary point-in-time data through ports;
3. normalize symbols, timestamps, units, and corporate/session conventions;
4. validate completeness, freshness, monotonicity, and plausibility;
5. calculate registered features;
6. calculate the point-in-time cycle composite and its effective weight;
7. calculate axis scores and classifications;
8. estimate transition state and probability;
9. adjust confidence using data quality and contradictions;
10. generate policy hints from a versioned mapping;
11. create an immutable snapshot and input fingerprint;
12. persist and publish the snapshot idempotently;
13. render reports as downstream views of that saved snapshot.

Never let report generation rerun inference or fetch fresh data.

## 12. Strategy-engine integration contract

Expose snapshots through at least one synchronous read boundary (library/API) and optionally an event publisher. JSON is the canonical cross-process format.

Policy hints describe suitability, never orders. Examples:

- `MEAN_REVERSION_FAVORED`
- `DIRECTIONAL_BULLISH_STRUCTURES_FAVORED`
- `DIRECTIONAL_BEARISH_STRUCTURES_FAVORED`
- `SHORT_GAMMA_REDUCE_OR_AVOID`
- `BREAKOUT_RISK_HIGH`
- `NO_TRADE_DATA_QUALITY`

The strategy engine must independently check:

- supported `schema_version`
- snapshot age and intended decision horizon
- confidence and data-quality limits
- permitted composite/axis states
- portfolio, event, liquidity, and risk constraints

Suggested consumer behavior:

```python
snapshot = regime_client.latest("SPX", as_of=decision_time)
if snapshot.is_stale(max_age_seconds=90):
    return NoTrade("stale regime")
if snapshot.confidence < configured_minimum:
    return NoTrade("low regime confidence")
decision = strategy_policy.evaluate(snapshot, market_state, portfolio_state)
```

Do not embed strategy names, strikes, deltas, wing widths, profit targets, stop losses, or capital sizing inside the classifier. Put mappings from regime to allowed strategy families in a separate, versioned policy package owned by the strategy side.

## 13. Reporting and outputs

All outputs are projections of a persisted `RegimeSnapshot`:

- JSON for machines
- concise text/Markdown for operators
- optional dashboard/API response
- audit record with feature and model versions

A concise report should show:

```text
10:45 ET | SPX | RANGE / POSITIVE GAMMA
Trend: -8/100 | Volatility: 31/100 | Confidence: 0.84
Cycle: +14/100 | Confidence: 0.42 | Horizon: 5d | Effective weight: 3%
Transition risk (60–90m): 0.18
For: positive GEX; VIX curve in contango; price inside opening range
Against: price slightly below VWAP
Data: complete; oldest critical input 28s
Hint: mean reversion favored; monitor opening-range break
```

Reports must state `as_of`, timezone, source freshness, and unknown/degraded fields. They must not imply that a policy hint is financial advice or an executable instruction.

## 14. Backtesting and evaluation

Freeze all inputs at each decision timestamp, especially 10:45 ET. Enforce `available_time <= as_of`. Account for exchange calendars, DST, early closes, late/missing vendor records, and revisions.

Evaluate the regime engine independently of any single strategy:

- direction: forward-return sign/magnitude and path
- volatility: subsequent realized range/variance
- range/trend: efficiency ratio, VWAP behavior, and excursion metrics
- transition: Brier score, calibration curve, precision/recall by horizon
- cycles: walk-forward expected-return calibration, directional hit rate, information coefficient, stability by era, and incremental value over a no-cycle baseline
- confidence: reliability diagrams and performance by confidence bucket
- operational: coverage, staleness, latency, and unknown rate

Use walk-forward or purged time-series validation. Fit thresholds and weights only on training periods. Keep final holdout periods untouched. Report performance by volatility era and exceptional-event periods, not only aggregate accuracy.

For cycle evaluation, avoid random cross-validation and full-history seasonal averages in historical predictions. At every replay date, rebuild the composite using only earlier observations. Compare against simple baselines, report transaction-free forecasting metrics before strategy P&L, and test whether each cycle component adds incremental out-of-sample value after controlling for trend and volatility. Disable components that are unstable across eras or depend on too few independent cycles.

A strategy P&L backtest is a downstream integration test, not the sole measure of classifier quality.

## 15. Testing requirements

Every change must include appropriate tests:

- pure unit tests for each feature and policy boundary
- contract tests for every provider adapter
- golden snapshot tests for stable serialized output
- replay tests for known range, trend, expansion, stress, and transition sessions
- property tests for score bounds, timestamp rules, and deterministic output
- missing/stale/conflicting-data tests
- timezone, DST, holiday, and early-close tests
- cycle-boundary tests for trading-year alignment, presidential term mapping, decennial digits, option-expiration calendars, and leap years
- tests proving cycle estimates at a historical `as_of` cannot access later returns or full-sample statistics
- tests for component correlation caps, small-sample shrinkage, contradictory components, and stress-regime suppression
- integration tests proving report generation does not fetch or mutate data
- architectural test preventing imports from execution/broker-order packages

Tests must not call paid/live data services by default. Live-provider tests require an explicit marker and environment configuration.

## 16. Observability and operations

Emit structured logs and metrics keyed by `snapshot_id`, never secrets or full credentials. Track:

- provider latency/error/rate-limit counts
- observation freshness and missingness by family
- feature calculation failures
- snapshot generation latency and idempotency conflicts
- label, confidence, and transition distributions
- drift and out-of-distribution indicators
- strategy-consumer schema rejections

Health endpoints must distinguish process health from data readiness. A running service with stale critical data is not ready.

## 17. Security and configuration

- Never commit credentials, tokens, account identifiers, or licensed raw datasets.
- Read secrets only in infrastructure/adapters through the configured secret mechanism.
- Provide `.env.example` with names, never values.
- Validate configuration at startup and fail with actionable messages.
- Record the resolved non-secret configuration version in every snapshot.
- Respect provider licensing, redistribution, and retention terms.

## 18. Agent workflow

Before coding:

1. inspect the repository, current contracts, tests, and local `AGENTS.md` files;
2. identify the bounded context and dependency direction affected;
3. state acceptance criteria and point-in-time assumptions;
4. prefer extending an existing port/model over coupling a core module to a vendor.

While coding:

- make the smallest coherent change;
- preserve public contracts unless a versioned migration is part of the task;
- keep calculations pure and side effects at adapters/application edges;
- add provenance and explicit null/quality behavior for new data;
- never silently substitute a proxy symbol or fallback source;
- do not mix unrelated refactors with feature work.

Before finishing:

1. run targeted tests, then the relevant full suite;
2. run formatting, linting, static typing, and schema validation;
3. verify deterministic replay with fixed fixtures;
4. document new configuration, formulas, and contracts;
5. summarize changed behavior, evidence, limitations, and migration needs.

If requirements are ambiguous in a way that changes data licensing, execution risk, public contracts, or look-ahead assumptions, stop and ask. Otherwise choose the simplest design consistent with this guide and document the assumption.

## 19. Definition of done

A change is done only when:

- domain boundaries remain intact;
- point-in-time behavior is explicit and tested;
- output is typed, versioned, explainable, and deterministic;
- missing/stale inputs have safe behavior;
- provider-specific code is isolated behind a port;
- tests and replay fixtures cover normal and failure paths;
- documentation describes formulas and configuration;
- the strategy engine can consume the snapshot without importing regime internals;
- no execution side effect exists in the regime engine.

## 20. Initial delivery sequence

Unless the repository already dictates another order, implement vertical slices:

1. canonical models, clock/calendar, JSON schema, and fixture adapter;
2. normalized SPX bars plus trend/intraday features;
3. VIX-family data plus volatility and term-structure features;
4. point-in-time seasonal/presidential/decennial cycle composite with walk-forward fixtures;
5. rule-based axis scores, confidence, and composite classifier;
6. persisted snapshot, JSON output, and concise report;
7. historical replay with strict `available_time` checks;
8. options-positioning plugin and transition detector;
9. strategy-client contract and no-trade safety checks;
10. monitoring, calibration, and optional ML challenger model.

At every stage, produce a usable end-to-end snapshot rather than building isolated layers with no executable path.
