# SPX 0DTE Options Backtesting Engine — Codex Instructions

## 1. Project Objective

Build a production-quality Python backtesting framework for **SPX 0DTE options strategies**.

The primary goal is to determine which market factors and market regimes produce positive risk-adjusted expectancy for different 0DTE strategies.

The system must support:

1. Historical SPX 5-minute price data
2. Historical VIX data
3. Historical SPX 0DTE option-chain data
4. Intraday market-regime classification
5. Strategy generation
6. Realistic option execution simulation
7. Stop-loss and profit-taking logic
8. Trade-level P&L analysis
9. Factor/regime attribution
10. Out-of-sample and walk-forward testing

Do **not** optimize parameters blindly. The system must be designed to minimize look-ahead bias and overfitting.

---

# 2. Initial Strategy

Implement the first strategy as a **0DTE SPX Iron Condor**.

Trading window:

* First entry: 10:45 AM ET
* Last entry: 3:15 PM ET
* New trade every 15 minutes
* No new trades after 3:15 PM ET
* All positions must be closed before the market closes

Example entry times:

```text
10:45
11:00
11:15
11:30
...
15:15
```

Use America/New_York timezone internally.

Do not use Australian local time for calculations.

---

# 3. Initial Iron Condor Rules

At every eligible entry timestamp:

### Underlying

Use the current SPX index price.

Select:

* Short put: approximately 8–15 delta
* Short call: approximately 8–15 delta
* Long put: 30 SPX points below short put
* Long call: 30 SPX points above short call

The resulting structure is:

```text
Long Put
   |
   | 30 points
   |
Short Put
   |
   |------ SPX ------|
   |
Short Call
   |
   | 30 points
   |
Long Call
```

Record the actual delta of each selected option.

Do not assume that the strike corresponding to a target delta is exactly the same on every date.

---

# 4. Entry Credit

Calculate the theoretical executable credit using historical bid/ask data.

Default execution assumption:

```text
Short options → bid
Long options  → ask
```

Therefore:

```text
IC credit =
short_put_bid
- long_put_ask
+ short_call_bid
- long_call_ask
```

Also calculate a mid-price version for comparison.

Do NOT use midpoint fills as the only backtest result.

Store both:

```text
theoretical_mid_credit
conservative_executable_credit
```

Allow configurable slippage.

---

# 5. Credit Filter

The initial target is:

```text
$1.00–$2.50 SPX points
```

Make this configurable:

```python
MIN_CREDIT = 1.0
MAX_CREDIT = 2.5
```

Do not force a trade if the credit is outside the configured range.

Record the reason when a trade is rejected.

Examples:

```text
NO_OPTION_DATA
NO_VALID_STRIKE
CREDIT_TOO_LOW
CREDIT_TOO_HIGH
MISSING_GREEKS
OUTSIDE_TRADING_WINDOW
REGIME_FILTER
```

---

# 6. Stop-Loss

Initial stop-loss rule:

```text
stop_loss = 0.90 * initial_credit
```

Interpret carefully.

For an iron condor sold for:

```text
credit = $2.00
```

the position is considered stopped when the cost to close reaches:

```text
$2.00 + $1.80 = $3.80
```

Do NOT incorrectly interpret "0.9 × credit" as a maximum loss of only $1.80.

The implementation must clearly distinguish:

```text
initial_credit
current_mark_to_close
profit_loss
stop_loss_price
```

Make the stop multiplier configurable.

Test:

```text
0.50
0.60
0.75
0.90
1.00
1.25
1.50
```

But parameter optimization must only be performed on training data.

---

# 7. Expiration Exit

If the position has not been stopped out:

* Close it before market close.
* Do not hold the SPX 0DTE position overnight.

Prefer an explicit configurable closing time, e.g.:

```python
FORCED_EXIT_TIME = "15:55"
```

Do not assume that the last available historical bar represents an executable closing price.

---

# 8. Market Factors

Build a feature engine that calculates the following factors at every 5-minute timestamp.

## Factor 1 — Market Regime / Trend

Calculate:

* SPX vs VWAP
* VWAP slope
* 5-minute return
* 15-minute return
* 30-minute return
* 60-minute return
* EMA slope
* ADX
* Opening-range breakout
* Distance from VWAP

Classify:

```text
STRONG_BULL
BULL
NEUTRAL
BEAR
STRONG_BEAR
```

---

## Factor 2 — Volatility

Calculate:

* VIX
* VIX percentage change
* 5-minute realized volatility
* 30-minute realized volatility
* 60-minute realized volatility
* ATR
* Intraday range
* Range expansion
* IV if available
* IV/RV relationship

Calculate approximate expected move:

```python
expected_move = SPX * VIX / sqrt(252)
```

Also record:

```text
distance_to_expected_move
short_strike_distance / expected_move
```

---

## Factor 3 — Key Price Levels

Calculate:

* Previous-day high
* Previous-day low
* Previous-day close
* Overnight high
* Overnight low
* Current-day open
* Opening-range high
* Opening-range low
* VWAP
* Recent swing highs
* Recent swing lows

For every short strike calculate:

```text
distance_to_previous_high
distance_to_previous_low
distance_to_VWAP
distance_to_opening_range
distance_to_overnight_level
```

---

## Factor 4 — Intraday Momentum

Calculate:

* 5-minute return
* 15-minute return
* 30-minute return
* 60-minute return
* ATR-normalized return
* Consecutive up/down bars
* Range expansion
* VWAP distance
* Opening-range breakout status

Classify momentum:

```text
STRONG_UP
UP
NEUTRAL
DOWN
STRONG_DOWN
```

---

## Factor 5 — Options Positioning

If historical option-chain information is available, calculate:

* Put/call volume
* Put/call open interest
* Strike-level volume
* Strike-level open interest
* Delta
* Gamma
* Theta
* Vega
* Implied volatility
* IV skew
* ATM IV
* OTM put IV
* OTM call IV

Where reliable option-positioning data exists, calculate an estimated gamma-exposure metric.

Do not pretend that dealer gamma is directly observable unless the data actually supports the calculation.

Clearly label estimated metrics as:

```text
ESTIMATED_GAMMA_EXPOSURE
```

---

# 9. Regime Engine

Create a separate module:

```text
regime_engine.py
```

It must classify the market using only information available **at that timestamp**.

Initial regimes:

```text
TREND_UP
TREND_DOWN
RANGE
HIGH_VOL_EXPANSION
LOW_VOL
UNKNOWN
```

Example:

```python
if strong_positive_momentum and price_above_vwap:
    regime = "TREND_UP"

elif strong_negative_momentum and price_below_vwap:
    regime = "TREND_DOWN"

elif low_adx and near_vwap:
    regime = "RANGE"

elif volatility_expansion:
    regime = "HIGH_VOL_EXPANSION"
```

Do not hard-code these definitions permanently.

Make thresholds configurable.

---

# 10. Strategy Selection

Create:

```text
strategy_engine.py
```

Initially support:

```text
IRON_CONDOR
```

Design the architecture so that later strategies can be added:

```text
BULL_PUT_SPREAD
BEAR_CALL_SPREAD
IRON_BUTTERFLY
BUTTERFLY
DOUBLE_CALENDAR
```

Each strategy should expose a common interface:

```python
generate_trade(context) -> TradeCandidate
```

The strategy must receive a `MarketContext` containing only information known at the entry time.

---

# 11. Critical Requirement — No Look-Ahead Bias

This is one of the most important requirements.

At timestamp `T`, the strategy may only use:

```text
SPX data <= T
VIX data <= T
option-chain data <= T
market factors calculated <= T
```

Never use:

```text
future day's high/low
future SPX movement
future option Greeks
future IV
future VIX
future VWAP
future regime
```

Do not calculate a daily statistic using the full day's data and then use it for an intraday trade.

For example, this is INVALID:

```python
day_high = df["high"].max()
```

if the trade occurs before the day's high has occurred.

Use only information available up to the current timestamp.

---

# 12. Historical Option Data

The backtester should use historical option-chain snapshots whenever possible.

Minimum required fields:

```text
timestamp
expiration
strike
option_type
bid
ask
last
volume
open_interest
delta
gamma
theta
vega
implied_volatility
```

For every trade, save the exact option quotes used for entry and exit.

If historical Greeks are unavailable, calculate them only if the required inputs are available.

Do not silently substitute current Greeks for historical Greeks.

---

# 13. Data Quality

Create a dedicated:

```text
data_validation.py
```

Validate:

* Missing timestamps
* Duplicate timestamps
* Missing bid/ask
* Crossed markets
* Zero bid
* Negative prices
* Invalid strikes
* Missing Greeks
* Expired contracts
* Incorrect expiration
* Timestamp timezone problems
* Market holidays
* Early-close days

Every rejected observation should have a reason.

---

# 14. Execution Simulator

Create:

```text
execution_engine.py
```

Support:

```text
MID
BID_ASK
BID_ASK_PLUS_SLIPPAGE
```

Default should be conservative.

For a multi-leg strategy, model each leg separately.

Do not simply multiply the theoretical mid-price by the number of contracts.

Allow:

```python
SLIPPAGE_POINTS = configurable
```

Also account for commissions and exchange fees through a configurable cost model.

---

# 15. Position Lifecycle

Each trade must have:

```text
CREATED
ENTERED
MONITORING
STOPPED
PROFIT_TARGET
FORCED_EXIT
EXPIRED
```

Store:

```text
entry_timestamp
exit_timestamp
entry_credit
exit_debit
gross_pnl
commission
fees
net_pnl
exit_reason
```

---

# 16. Trade Database

Use SQLite initially.

Suggested tables:

```text
market_bars
option_quotes
market_features
trades
trade_legs
daily_summary
backtest_runs
```

Every backtest run should have a unique:

```text
run_id
```

Store the complete parameter configuration for reproducibility.

---

# 17. Trade Record

Each trade should contain at least:

```text
run_id
date
entry_time
exit_time

SPX_entry
SPX_exit

VIX
VIX_change
expected_move

VWAP
opening_range_high
opening_range_low

regime
trend_score
volatility_score
sr_score
momentum_score
gamma_score

short_put
long_put
short_call
long_call

short_put_delta
short_call_delta

put_credit
call_credit
total_credit

entry_value
exit_value

gross_pnl
commission
fees
net_pnl

max_profit
max_loss

exit_reason
```

---

# 18. Factor Scoring

Create a standardized score:

```text
trend_score       = -2 to +2
volatility_score  = -2 to +2
sr_score          = -2 to +2
momentum_score    = -2 to +2
gamma_score       = -2 to +2
```

Total:

```python
factor_score = trend_score + volatility_score + sr_score + momentum_score + gamma_score
```

Do not assume that a high score means bullish.

For the iron condor, the score should represent **trade suitability**, not directional prediction.

Eventually support:

```text
IC suitability score
Bullish strategy score
Bearish strategy score
Butterfly score
```

---

# 19. Backtest Metrics

At minimum calculate:

### Overall

```text
Total trades
Winning trades
Losing trades
Win rate
Gross P&L
Net P&L
Average trade
Median trade
Profit factor
Expectancy
Sharpe ratio
Sortino ratio
Maximum drawdown
Maximum drawdown duration
```

### Risk

```text
Worst trade
Worst day
Worst week
Worst month
Largest consecutive loss
Largest consecutive losing days
95% VaR
95% CVaR
```

### Breakdown

Group results by:

```text
Entry time
Weekday
VIX regime
SPX regime
Factor score
Delta
Credit
Expected-move distance
Volatility regime
Trend regime
```

---

# 20. Entry-Time Analysis

Produce a table like:

```text
Entry Time | Trades | Win Rate | Avg P&L | PF | Max DD
--------------------------------------------------------
10:45      |
11:00      |
11:15      |
...
15:15      |
```

This is particularly important for 0DTE because gamma/theta characteristics change substantially throughout the session.

---

# 21. Delta Analysis

Group trades by actual short-leg delta:

```text
0.05–0.08
0.08–0.10
0.10–0.12
0.12–0.15
0.15–0.20
```

Compare:

```text
win rate
average P&L
expected value
max drawdown
tail loss
```

Do not assume 10-delta is optimal.

---

# 22. Expected-Move Analysis

Group trades according to short-strike distance:

```text
< 0.50 EM
0.50–0.75 EM
0.75–1.00 EM
1.00–1.25 EM
1.25–1.50 EM
> 1.50 EM
```

Determine whether expected-move normalization improves strategy selection.

---

# 23. Walk-Forward Testing

Do NOT optimize parameters on the entire historical dataset.

Implement:

```text
Training period
Validation period
Out-of-sample period
```

Example:

```text
2021–2023 → Training
2024      → Validation
2025–2026 → Out-of-sample
```

Then implement rolling walk-forward testing.

Example:

```text
Train 12 months
Test 3 months
Roll forward 3 months
```

The final performance report must clearly distinguish:

```text
IN-SAMPLE
VALIDATION
OUT-OF-SAMPLE
```

---

# 24. Parameter Optimization

Eventually optimize:

```text
short delta
wing width
minimum credit
maximum credit
stop multiplier
entry time
forced exit time
regime filters
```

But avoid brute-force optimization across dozens of parameters simultaneously.

Start with one-dimensional and two-dimensional sensitivity analysis.

Example:

```text
Delta × Stop Multiplier
Credit × Stop Multiplier
Entry Time × Delta
VIX Regime × Delta
```

Report the entire surface, not just the best parameter.

If one parameter combination is dramatically better than neighboring combinations, flag it as possible overfitting.

---

# 25. Benchmark

Every strategy must be compared against simple baselines.

Examples:

```text
Always trade IC
No regime filter
10-delta IC
15-delta IC
Fixed credit
Fixed stop
```

The factor-enhanced strategy must demonstrate improvement over these baselines.

---

# 26. Statistical Robustness

Calculate confidence intervals where practical.

Use bootstrap analysis to estimate:

```text
expectancy confidence interval
mean P&L confidence interval
Sharpe confidence interval
maximum drawdown distribution
```

Also perform:

```text
trade-order randomization
Monte Carlo trade sequence simulation
```

This helps determine whether the observed performance is robust or simply dependent on a favorable sequence of trades.

---

# 27. Visualizations

Generate:

### Equity curve

```text
Cumulative net P&L
```

### Drawdown curve

```text
Peak-to-trough drawdown
```

### P&L by entry time

### P&L by VIX regime

### P&L by delta

### P&L by expected-move distance

### P&L distribution

### Factor-score performance

### Monthly P&L heatmap

### Entry/exit examples

Charts should be generated automatically for every major backtest.

---

# 28. Project Structure

Use a modular structure similar to:

```text
spx_0dte_backtest/
│
├── config/
│   └── settings.yaml
│
├── data/
│   ├── loaders/
│   ├── validators/
│   └── schemas/
│
├── features/
│   ├── trend.py
│   ├── volatility.py
│   ├── support_resistance.py
│   ├── momentum.py
│   └── options_positioning.py
│
├── regimes/
│   └── regime_engine.py
│
├── strategies/
│   ├── base.py
│   └── iron_condor.py
│
├── execution/
│   └── execution_engine.py
│
├── risk/
│   └── risk_manager.py
│
├── backtest/
│   ├── engine.py
│   ├── portfolio.py
│   └── metrics.py
│
├── analysis/
│   ├── factor_analysis.py
│   ├── sensitivity.py
│   └── walk_forward.py
│
├── visualization/
│   └── charts.py
│
├── database/
│   └── sqlite.py
│
├── tests/
│
├── scripts/
│   ├── download_data.py
│   ├── run_backtest.py
│   └── analyze_results.py
│
└── README.md
```

---

# 29. Engineering Requirements

Use:

* Python 3.11+
* pandas
* numpy
* scipy
* pyarrow
* SQLAlchemy or sqlite3
* matplotlib
* pytest
* pydantic/dataclasses where appropriate

Prefer clean typed Python.

Use:

```python
from dataclasses import dataclass
```

for immutable configuration/context objects where appropriate.

Avoid global mutable state.

Do not create a monolithic Python script.

---

# 30. Testing Requirements

Every major component must have unit tests.

At minimum test:

### Strike selection

```text
Correct delta selection
Correct 30-point wings
Correct expiration
```

### Credit calculation

```text
Bid/ask calculation
Mid-price calculation
Slippage
```

### Stop-loss

```text
Correct stop threshold
Correct P&L
Correct exit reason
```

### Time handling

```text
NYSE timezone
DST
market holidays
early closes
```

### Look-ahead protection

Create explicit tests proving that future bars cannot influence current features.

### Option expiration

Verify that only same-day expiration contracts are selected.

---

# 31. Reproducibility

Every backtest must be reproducible from:

```text
data version
code version
configuration
strategy version
parameter set
```

Save configuration as JSON/YAML with every run.

Example:

```text
results/
    2026-08-12_run_001/
        config.yaml
        trades.parquet
        daily_pnl.parquet
        metrics.json
        charts/
```

---

# 32. CLI

Create a simple command-line interface.

Examples:

```bash
python -m scripts.run_backtest \
    --start 2021-01-01 \
    --end 2026-06-30 \
    --strategy iron_condor
```

and:

```bash
python -m scripts.analyze_results \
    --run-id 20260812_001
```

and:

```bash
python -m scripts.walk_forward \
    --strategy iron_condor
```

---

# 33. Development Sequence

Do NOT attempt to build everything at once.

Implement in this order:

### Phase 1

Data schemas + SPX 5-minute loader + timezone handling.

### Phase 2

Feature engine.

### Phase 3

Historical option-chain loader.

### Phase 4

Option strike selection.

### Phase 5

Iron-condor trade construction.

### Phase 6

Execution simulator.

### Phase 7

Backtest engine.

### Phase 8

P&L and risk metrics.

### Phase 9

Regime engine.

### Phase 10

Factor analysis.

### Phase 11

Walk-forward testing.

### Phase 12

Parameter sensitivity/optimization.

### Phase 13

Visualization/reporting.

Do not proceed to the next phase until the current phase has passing tests.

---

# 34. Important Data Limitation

If suitable historical SPX 0DTE option-chain data is not available locally, do NOT fabricate or approximate it without clearly labeling the approximation.

Instead:

1. Build the backtester interfaces.
2. Create a synthetic/mock option dataset for unit tests only.
3. Clearly separate mock data from real historical data.
4. Document the exact fields required from a real data provider.
5. Make it easy to replace the mock data with real data.

Never present a backtest using synthetic option prices as a realistic historical performance result.

---

# 35. First Deliverable

The first Codex deliverable should NOT be the complete system.

Build **Phase 1 only**:

1. Project structure
2. Configuration system
3. SPX 5-minute data schema
4. Data loader interface
5. Timezone/session handling
6. Data validation
7. Unit tests
8. README
9. Example dataset
10. Working CLI

Then stop and report:

```text
Files created
Tests passed
Known limitations
Next recommended phase
```

Do not silently implement later phases.

---

# 36. Coding Principles

Prioritize:

```text
Correctness
Reproducibility
No look-ahead bias
Realistic execution
Testability
Modularity
Observability
```

over:

```text
maximum speed
maximum parameter optimization
complex AI/ML models
```

Do not introduce machine learning until the rule-based baseline is fully working and validated.

The purpose of the project is to discover whether the proposed SPX 0DTE factors genuinely improve risk-adjusted expectancy, not to manufacture a high backtest return.

Before implementing any component, identify assumptions that could introduce survivorship bias, look-ahead bias, selection bias, execution bias, or overfitting.

