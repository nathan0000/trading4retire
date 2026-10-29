# SPX 0DTE backtesting framework

This repository contains Phases 1–13 of a point-in-time-safe SPX 0DTE research
platform: validated market data, causal features and regimes, option-chain and
strike selection, iron-condor construction and execution, position backtesting,
risk metrics, factor attribution, walk-forward testing, bounded sensitivity
analysis, and automated reporting. No historical performance claim is included.

## Setup

Python 3.11 or newer and `uv` are recommended:

```bash
uv sync --extra dev
```

## Validate the example

The bundled input is explicitly synthetic:

```bash
uv run spx-0dte validate-spx \
  --input data/examples/spx_5min_example.csv \
  --config config/settings.yaml
```

Use `--json` for machine-readable output. Strict validation returns exit code `1` for
data-quality failures and exit code `2` for configuration or input errors.

## Collect recent one-minute SPX bars

Run this after the US cash session to capture Yahoo Finance recent `^GSPC` one-minute bars in the same canonical SPX schema used by the ThetaData adapter:

```bash
uv run spx-0dte download-data --provider yfinance --interval 1 --start 2026-08-31 --end 2026-08-31 --output data/vendor/yfinance/2026-08-31
```

The output includes `spx_1min.parquet`, an empty canonical options file, and a checksummed manifest. Yahoo intraday retention is limited and its feed is not exchange-grade; use it for underlying bars, not as a substitute for historical option-chain data.

## Run an end-to-end backtest

```bash
uv run spx-0dte run-backtest \
  --spx path/to/spx_5min.parquet \
  --options path/to/spx_0dte_options.parquet \
  --output results/run_001 \
  --run-id run_001 \
  --database results/research.sqlite
```

Historical option snapshots must include exact entry, monitoring, and 15:55 ET
forced-exit quotes. See [the CLI workflow](docs/end_to_end_cli.md).

## Test

```bash
uv run pytest
uv run ruff check .
```

See [the architecture decisions](docs/architecture.md) and
[the data contract](docs/data_contracts.md). The development sequence and constraints
are defined in [AGENTS.md](AGENTS.md).

Phase-specific decisions are documented through `docs/phase13_reporting.md`.
