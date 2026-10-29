# Code review: 2026-10-29

This review covers the code imported from the NAS `projects/` folder into this repository.
At the time of review:

- `optionbacktest`: 223 tests pass and `ruff` is clean. The run logs 464 deprecation warnings.
- `marketregime`: 92 of 93 tests pass. `ruff`, `ruff format` and strict `mypy` are clean.

The code is imported unchanged. Findings are listed most severe first.

## High

### 1. Look-ahead bias at entry (`optionbacktest`)

`config/settings.yaml` declares `timestamp_semantics: bar_start`, so the bar stamped
10:45 covers 10:45–10:50, and its `close` is known only at 10:50. The code ignores this in
two places:

- `FeatureEngine` sets `data_cutoff_timestamp = timestamp`
  ([`features/engine.py:51`](optionbacktest/src/spx_0dte_backtest/features/engine.py)).
  This understates the cutoff by one bar, so the causality guards in `strategies/base.py`
  and `regimes/regime_engine.py` pass when they should fail.
- The workflow enters a trade on the 10:45 option snapshot, but uses that bar's close and
  features to pick strikes
  ([`orchestration/backtest_workflow.py:113-117`](optionbacktest/src/spx_0dte_backtest/orchestration/backtest_workflow.py)).
  Every trade therefore sees SPX five minutes into the future.

Reproduced: the row stamped 09:55 reports cutoff 09:55, but its close is only known at 10:00.

**Fix:** For `bar_start`, set `data_cutoff_timestamp = timestamp + frequency`. Then pair
each feature row with the option snapshot at its cutoff time, so the 10:45 bar is used for
a 10:50 entry. Add a test that fails whenever the feature cutoff is later than the quote
timestamp.

### 2. One missing quote aborts the whole backtest (`optionbacktest`)

Every monitoring snapshot calls `execute_exit`, which raises if any of the four legs has
no quote ([`execution/execution_engine.py:131,139`](optionbacktest/src/spx_0dte_backtest/execution/execution_engine.py)).
The workflow turns that into a `WorkflowError` for the entire run. Real ThetaData chains
routinely have gaps in individual strikes, so a single gap discards months of results.

**Fix:** Skip incomplete intermediate snapshots and record them as unmarked. Require a
complete quote set only at the forced exit, and report how many snapshots were skipped.

### 3. Stops trigger on worst-case fill prices, not marks (`optionbacktest`)

`BacktestEngine` decides whether to stop out using the executable debit, which is the ask
price plus slippage on all four legs
([`backtest/engine.py:76-82`](optionbacktest/src/spx_0dte_backtest/backtest/engine.py)).
On wide 0DTE spreads, a momentary widening triggers a stop that a broker's mark-based stop
would not, which biases results pessimistically in a way that depends on parameters.

**Fix:** Decide on the mark (mid, or a configurable price) and fill at the executable price.
Report both values in the trade record.

## Medium

4. **The entry schedule is defined twice.** The workflow hard-codes 10:45–15:15 every 15 minutes
   ([`backtest_workflow.py:192`](optionbacktest/src/spx_0dte_backtest/orchestration/backtest_workflow.py)),
   duplicating `IronCondorSettings.first_entry/last_entry/entry_interval_minutes`. If you
   change the settings, the workflow ignores them. Derive the schedule from the strategy settings.
5. **There are no portfolio-level risk limits.** A new iron condor can open every 15 minutes,
   so up to 19 can be open at once. There is no cap on open positions, no daily loss limit and
   no capital or buying-power model. Metrics add up the P&L of overlapping trades, so the
   reported drawdown understates intraday exposure. Add `max_open_positions`, a daily stop
   and a per-trade margin of `width − credit`.
6. **The credit filter uses pre-slippage prices.**
   ([`strategies/iron_condor.py:79`](optionbacktest/src/spx_0dte_backtest/strategies/iron_condor.py))
   With the default 0.05 slippage on each of 4 legs, the actual entry credit can be up to 0.20
   below `min_credit`. Filter on the same execution model that fills the trade.
7. **Monitoring snapshots are built from the entire option chain.**
   ([`backtest_workflow.py:127-131`](optionbacktest/src/spx_0dte_backtest/orchestration/backtest_workflow.py))
   The workflow builds a Python `OptionQuote` for every strike in the chain, for every
   snapshot, for every candidate. That is fine for the 5-row example file, but a real SPXW
   chain has thousands of strikes. Filter to the four legs' contracts first, and group by
   timestamp once per session.
8. **A failing `marketregime` contract test.** `test_filters_by_symbol_and_date_range`
   ([`tests/contract/test_fixture_price_bars_contract.py:72`](marketregime/tests/contract/test_fixture_price_bars_contract.py))
   passes `end=2024-01-12 00:00 UTC`. Daily bars are stamped at the 16:00 New York close
   (21:00 UTC), so the 12 January bar is excluded. The adapter is correct; the test's bound
   is wrong. Also decide whether `get_bars` should take session dates for daily bars, because
   this mistake is easy for any caller to make.

## Low / housekeeping

9. **No CI.** `marketregime/.github/workflows/` is empty. Add one workflow that runs `uv run pytest`
   and `ruff` for both packages, plus `mypy` for `marketregime`.
10. **The `marketregime` docs have drifted.** The README links to `docs/architecture.md` and a
    `RegimeSnapshot` JSON Schema, and neither exists (`docs/`, `schemas/` and `scripts/` are
    empty). Around 30 modules are empty stubs, which makes the reported 99% coverage misleading.
11. **The two projects overlap.** Both have feature and regime engines
    (`spx_0dte_backtest.regimes` and `regime_engine`). Decide whether the backtester will
    consume `RegimeSnapshot`, and align Python versions (`>=3.11` vs `>=3.12`).
12. **Devcontainer problems.** `optionbacktest/.devcontainer` runs as `root` and mounts
    `/volume1/docker/optionbacktest`, but the project now lives at
    `/volume1/docker/projects/optionbacktest`. Running as root is why many NAS files are
    root-owned. Use a non-root `remoteUser` and fix the mount path. A Synology Drive conflict
    copy of the Dockerfile was not imported.
13. **Deprecation warnings.** NumPy's "generic timedelta unit" warning appears 464 times in
    the tests and will become an error. Use explicit units such as
    `pd.Timedelta(minutes=5)` and `Timedelta(hours=2)`.
14. **Data provenance.** `sample_data.zip` (3.6 GB on the NAS) is intentionally not committed.
    Document its source, licence and checksum in `optionbacktest/data/README.md`.
15. **No LICENSE.** The repository is public but has no licence, which means all rights are reserved.
