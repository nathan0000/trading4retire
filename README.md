# trading4retire

Research tooling for systematic SPX 0DTE options trading.

| Project | What it is | Status |
| --- | --- | --- |
| [`optionbacktest/`](optionbacktest/) | Point-in-time SPX 0DTE iron-condor backtesting framework (`spx-0dte-backtest`): data validation, causal features and regimes, strike selection, per-leg execution, lifecycle, metrics, walk-forward, sensitivity and reporting. | Phases 1–13 implemented |
| [`marketregime/`](marketregime/) | Market regime engine (`market-regime-engine`) that turns observations into an explainable, versioned `RegimeSnapshot` for a strategy engine. Makes no trading decisions. | Phase 1: domain models, ports, fixture adapters |

Each project is an independent [`uv`](https://docs.astral.sh/uv/) package with its own
`README.md`, `AGENTS.md` engineering spec and lock file:

```sh
cd optionbacktest && uv sync --extra dev && uv run pytest
cd marketregime && make install && make check
```

Bundled datasets are synthetic or small fixtures. Historical vendor data is not
committed. Nothing here is investment advice, and no historical performance is claimed.

See [`REVIEW.md`](REVIEW.md) for the current code review and recommended improvements.
