# Phase 13 visualization and reporting

Phase 13 generates headless, deterministic charts and a reproducible report
bundle from completed trade records. It does not alter trades, select parameters,
or treat synthetic data as historical evidence.

## Generated charts

`ChartGenerator.generate_all()` creates the following when required fields exist:

- equity and drawdown curves;
- P&L by entry time, VIX regime, actual delta, and expected-move distance;
- P&L distribution and factor-score performance;
- monthly P&L heatmap and SPX entry/exit examples;
- one- or two-dimensional training sensitivity surface.

Matplotlib uses the non-interactive `Agg` backend. PNG is the default, with SVG
also supported. Input frames are defensively copied and figures are closed after
saving.

## Missing information

Core completed-trade fields are required. Optional charts are skipped when their
inputs are absent or unusable. `ChartManifest` records generated paths and exact
skip reasons. VIX regimes are not inferred from VIX values, and SPX exits are not
reconstructed from unrelated bars.

## Report bundle

`ReportGenerator` writes:

```text
run_directory/
  trades.parquet
  daily_pnl.parquet
  metrics.json
  manifest.json
  summary.md
  charts/
```

The manifest includes caller-supplied reproducibility metadata, portable chart
paths, and skipped-chart reasons. Metadata must be JSON serializable. Undefined
floating-point metrics become JSON `null`. Generation rejects a non-empty
destination directory to preserve prior results.

Callers should provide run ID, data version, code version, configuration, and
strategy version in metadata once those identifiers are available from run
persistence.

## Interpretation constraints

- Sensitivity charts show the full training surface, not only its winner.
- Charts are descriptive and do not establish statistical significance.
- Missing historical values are never fabricated to complete a chart.
- Mock-data reports must never be presented as historical performance.
