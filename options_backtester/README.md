# options_backtester

A decoupled, provider-agnostic backtesting engine for multi-leg option strategies,
built to the specification in the repository's [`CLAUDE.md`](../CLAUDE.md). It evaluates
parameterized strategies (DTE, delta targets, wing widths, profit targets, stops) across
market regimes without touching engine code when the data provider or strategy changes.

```sh
uv sync --extra dev --extra yfinance
uv run pytest

# One backtest of the `base` strategy in config/strategy_grid.yaml -> results/<name>/
uv run options-backtester run

# Real SPY and VIX closes from Yahoo; option prices are still modelled (see below)
uv run options-backtester run --provider synthetic-yfinance --vix-symbol ^VIX \
    --start 2019-01-02 --end 2025-12-31

# Parameter sweep -> sensitivity matrices and heatmaps
uv run options-backtester grid --provider synthetic-yfinance --vix-symbol ^VIX
```

## Read this before trusting a number

**The default providers model option prices; they do not observe them.** `synthetic` and
`synthetic-yfinance` price every chain with Black-Scholes from trailing realized volatility
times a fixed premium, plus a fixed skew. That is good for validating the engine and for
seeing how parameters interact with slippage and costs. It is not evidence that a strategy
made or lost money, because the volatility risk premium a short-premium strategy harvests
is an input assumption here. Reports from these providers carry a `SYNTHETIC OPTION PRICES`
banner. A real result needs real chains: ThetaData history, or snapshots you accumulate
daily from Yahoo, Schwab or IBKR.

## Layers

| Layer | Package | What it does |
| --- | --- | --- |
| 1. Data | `data/` | `DataProvider` base class, canonical `OptionChainSnapshot` / bar frames, Parquet cache, vectorized BSM and Barone-Adesi-Whaley IV/Greeks fallback |
| 2. Regimes | `regimes/` | Trend (SMA + ADX), volatility level (IV rank, VIX levels), volatility dynamics, VIX term structure; `RegimeTimeline.at(ts)` |
| 3. Strategies | `strategies/` | Pydantic `StrategyConfig`, `LegBuilder` (delta / % OTM / std-dev strikes; dollar / delta / % wings), entry and exit rule evaluators, templates |
| 4. Engine | `engine/` | Per-leg fills with slippage and costs, margin, portfolio Greeks, expiration settlement, the event loop |
| 5. Analytics | `analytics/` | Metrics, regime attribution, Strategy Blueprint (Markdown + JSON), trade logs, sensitivity matrices |

Layers only meet through the canonical schema, `RegimeState`, `TradeCandidate` and
`BacktestResult`. A new provider implements two methods (`_fetch_bars`, `_fetch_chain`);
a new strategy implements one (`select_legs`).

### Data providers

| Provider | Underlying bars | Option chains | Greeks |
| --- | --- | --- | --- |
| `synthetic` | Seeded stochastic-volatility path | Modelled | Modelled |
| `synthetic-yfinance` | Yahoo daily history | Modelled | Modelled |
| `yfinance` | Yahoo daily history | Today's chain only | Fallback engine |
| `thetadata` | Theta Terminal v2 EOD | Historical EOD, per expiration | ThetaData's, fallback where absent |
| `schwab` | Price history | Today's chain only | Schwab's, fallback where absent |
| `ibkr` | `reqHistoricalData` | Today's chain only | IBKR model Greeks, fallback where absent |

Yahoo, Schwab and IBKR serve no historical chains. Their adapters refuse a past date
unless it is already cached, so a daily job that requests today's chain builds a real
history under `data_cache/{provider}/{symbol}/{date}.parquet`. Every later backtest and
sweep reads that cache and runs offline.

**The ThetaData, Schwab and IBKR adapters have only been tested against fake transports.**
Request shapes, pagination, token refresh, sentinel values, rate limiting and pacing
retries are covered by tests; nothing has been run against a live terminal, gateway or
account. Expect to adjust field names on first contact. Schwab reads
`SCHWAB_CLIENT_ID` / `SCHWAB_CLIENT_SECRET` from the environment and a token file holding
a refresh token from Schwab's one-time browser authorization.

### Strategy configuration

```yaml
name: iron_condor_16d_45dte
template: iron_condor            # iron_condor | vertical | strangle | calendar
min_dte: 30
max_dte: 45
strike: {method: delta, value: 0.16}        # delta | pct_otm | stdev
wing:   {method: dollar, value: 10}         # dollar | delta | pct_underlying
entry:
  allowed_trend: [SIDEWAYS, BULLISH]        # BULLISH | BEARISH | SIDEWAYS
  allowed_volatility: [HIGH_IV, NORMAL_IV]  # LOW_IV | NORMAL_IV | HIGH_IV
  min_iv_rank: 30
  max_spread_pct: 0.5
  min_open_interest: 100
  min_credit: 1.0
  max_open_positions: 3
  min_days_between_entries: 7
exit:
  profit_target_pct: 0.5         # of credit received / debit paid
  stop_loss_multiple: 2.0        # loss as a multiple of credit / debit
  exit_dte: 21
  underlying_breach: false
  max_position_delta: null
  trailing_stop_pct: null
  exit_on_regime_shift: false
```

Unknown keys are rejected. `config/strategy_grid.yaml` holds the `base` strategy, the
`sweep` (dotted paths such as `strike.value`) and the two `matrix` axes.

## Modelling decisions

- **Timestamps are known-at times.** A daily bar is stamped at its 16:00 New York close.
  The engine, regime lookup and synthetic pricer use only rows stamped at or before the
  current step. Tests truncate future data and assert the past is unchanged.
- **Trade at the snapshot.** With end-of-day data, signals and fills both use the closing
  snapshot. This assumes you can trade at the close you observed.
- **Triggers on the mark, fills at executable prices.** Profit targets and stops are
  evaluated on mid; the exit then fills at the configured fill model. The trade log keeps
  both (`exit_trigger_pnl`, `exit_mid_value`, `exit_value`, `slippage_paid`).
- **Exit priority** when several rules fire on one bar: expiration, stop loss, underlying
  breach, delta breach, trailing stop, profit target, DTE exit, max hold, regime shift.
- **Stop multiple is a loss.** A 2.0 stop on a 1.50 credit fires at a 3.00 loss, which is
  a 4.50 cost to close.
- **Missing quotes do not abort a run.** A mark with an unquoted leg is skipped and
  counted (`skipped_marks`); a forced exit without a quote uses the last complete mark
  and is flagged `exit_estimated`.
- **Credit and debit filters use the executable fill**, not mid.
- **Fills.** `MID`, `NATURAL` (buy ask / sell bid) or `SLIPPAGE`: mid plus a fraction of
  the spread that grows toward the natural price as contract volume thins. Combos are
  atomic; each leg pays commission and fees.
- **Margin.** Defined-risk structures reserve their maximum loss. Net-short-call
  structures use a Reg-T style estimate. New entries are refused beyond
  `max_margin_utilization` of equity.
- **Expiration.** Legs settle at intrinsic value on the last close at or before
  expiration, with no exit commission. `settlement: physical` adds an assignment fee per
  in-the-money contract and counts assigned contracts; it does not model holding the
  resulting shares over the weekend.
- **Early assignment** is flagged, not simulated: a short in-the-money leg whose
  extrinsic value is below a threshold (or below an upcoming dividend, for calls)
  increments `assignment_risk_flags`.
- **Regime attribution** groups trades by the regime known at entry. A regime is "ideal"
  only with at least `attribution_min_trades` trades and a t-statistic of mean trade P&L
  of at least 1.0. Regimes are tested many at a time, so treat a single high t-statistic
  as a hypothesis to test out of sample, not a finding.

## Not implemented

- Intraday bars. The loop is bar-agnostic and snapshots carry their own timestamps, but
  every adapter currently produces daily data.
- Butterfly and iron-butterfly templates, rolling and adjustments.
- Walk-forward / out-of-sample splitting and bootstrap confidence intervals. The older
  [`optionbacktest`](../optionbacktest) package has both for its 0DTE engine; the sweep
  here reports the full in-sample surface only.
- A holiday calendar. Trading days are whatever the bar provider returns.
- Dividend-aware pricing beyond a constant yield; BAW Greeks use finite differences.

## Relationship to the other packages

`optionbacktest` (intraday SPX 0DTE iron condors) and `marketregime` are unchanged. This
package reuses their ideas rather than their code, because both are tied to five-minute
SPX sessions and a four-leg condor. The findings in [`REVIEW.md`](../REVIEW.md) about
look-ahead at entry, aborting on a missing quote, stops on worst-case fills, pre-slippage
credit filters and absent portfolio limits are designed out here and covered by tests.
