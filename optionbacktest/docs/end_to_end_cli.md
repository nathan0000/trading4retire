# End-to-end backtest CLI

The production orchestration milestone connects the completed Phase 1–13
components without introducing a new strategy or synthetic pricing model.

## Command

Use the installed entry point:

```bash
uv run spx-0dte run-backtest \
  --spx /path/to/spx_5min.parquet \
  --options /path/to/spx_0dte_options.parquet \
  --vix /path/to/vix.parquet \
  --config config/settings.yaml \
  --output results/run_001 \
  --run-id run_001 \
  --database results/research.sqlite \
  --json
```

The VIX input is optional. The module entry point accepts the same arguments
without the subcommand:

```bash
uv run python -m scripts.run_backtest --spx ... --options ... --output ...
```

## Workflow

`BacktestWorkflow` performs these operations in order:

1. Load and validate SPX bars and historical option snapshots.
2. Stop when strict validation rejects either required source.
3. Compute causal features and point-in-time regimes.
4. Visit the initial strategy schedule from 10:45 through 15:15 ET.
5. Generate an iron-condor candidate using only the exact entry snapshot.
6. Record every accepted candidate and rejection reason.
7. Add suitability scores using actual selected short strikes.
8. Execute and monitor each accepted position using chronological snapshots.
9. Require an exact executable forced-exit snapshot at 15:55 ET.
10. Normalize completed trades and attach SPX at the exact exit timestamp.
11. Generate metrics, charts, Parquet outputs, and audit manifests.

Accepted candidates with malformed lifecycle quotes, or an incomplete 15:55 forced-exit
snapshot, fail the run. Intermediate monitoring snapshots missing a leg are skipped and
counted in `skipped_marks`. They
are never silently omitted from performance. A valid run with no accepted trades
still produces zero-trade metrics and explicit skipped-chart reasons.

## Artifacts and reproducibility

In addition to the Phase 13 report bundle, orchestration writes:

- `candidates.parquet`, including every scheduled rejection;
- `workflow_summary.json`, containing candidate and completed-trade counts.

Manifest metadata includes the run ID, package/code version, strategy version,
full application configuration, execution costs and slippage, risk settings,
forced-exit settings, resolved source paths, SHA-256 source hashes, and whether
source labels identify the data as synthetic.

The command rejects a non-empty destination directory. Synthetic source labels
are retained and classified as `SYNTHETIC`; such output must not be presented as
historical performance.

## Current limitations

- Only `IRON_CONDOR` is orchestrated.
- Entry and lifecycle defaults are the currently implemented strategy/backtest
  settings; they are recorded in metadata but are not yet modeled in YAML.
- VIX ingestion is a small validated adapter, not a provider-specific loader.
- SQLite persistence is optional; omit `--database` for file-only reports.
