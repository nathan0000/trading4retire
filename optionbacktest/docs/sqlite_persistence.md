# SQLite run persistence

SQLite persistence is optional and complements the file-based report bundle.
Enable it with:

```bash
uv run spx-0dte run-backtest \
  --spx path/to/spx.parquet \
  --options path/to/options.parquet \
  --output results/run_001 \
  --run-id run_001 \
  --database results/research.sqlite
```

Every `run_id` is immutable and unique. An existing ID is rejected rather than
updated or overwritten.

## Schema version 1

The database contains:

- `backtest_runs`: status, versions, configuration, metadata, and report path;
- `market_bars`: canonical SPX observations used by the run;
- `option_quotes`: accepted historical option snapshots used as run input;
- `market_features`: complete point-in-time feature/regime payloads;
- `trade_candidates`: accepted candidates and every scheduled rejection;
- `trades`: typed lifecycle economics plus the normalized trade payload;
- `trade_legs`: role, quantity, contract identity, and exact entry/exit quotes;
- `daily_summary`: run-level daily P&L and costs.

All child tables reference `backtest_runs` with foreign keys. Frequently queried
option snapshot fields are indexed. Timestamps are stored as ISO 8601 strings
with their UTC offsets; local session meaning is therefore retained.

Feature, context, quote, and normalized-trade JSON preserves fields that evolve
across research phases while important identifiers and P&L columns remain typed
and directly queryable.

## Transactions and compatibility

A complete run is inserted in one transaction. Any failure rolls back the run
header and every child row. Foreign keys are enabled on every connection, and a
30-second busy timeout supports ordinary single-host research workloads.

The schema uses SQLite `PRAGMA user_version`. Databases newer than the supported
version are rejected and never downgraded. Future schema changes must use an
explicit forward migration.

## Read API

`SQLiteStore.list_runs()` returns run summaries. `load_trades(run_id)` returns
typed trade-table rows as a DataFrame. Direct SQL remains available for factor,
candidate-rejection, quote, and leg-level research.

SQLite is not used as a substitute for source files: source paths and SHA-256
hashes remain in run metadata, and the accepted input rows are persisted for
auditability.
