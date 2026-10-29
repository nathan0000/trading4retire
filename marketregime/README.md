# Market Regime Engine

A point-in-time-correct market regime engine for SPX and 0DTE options decision
support. It converts market observations into an explainable, versioned
`RegimeSnapshot` that a separate strategy engine consumes; this package makes
no trading decisions and has no execution side effects.

See `CLAUDE.md` / `AGENTS.md` for the full engineering specification and
`docs/architecture.md` for the bounded-context / dependency-flow overview.

## Status

Phase 1 of the delivery sequence in `CLAUDE.md` section 20: canonical domain
models, the clock/calendar ports, a fixture NYSE-calendar adapter, and the
`RegimeSnapshot` JSON Schema. Feature calculation, inference, and the
application pipeline are not implemented yet — see `docs/architecture.md`
for what's real vs. placeholder.

## Development

Requires Python >=3.12 and [`uv`](https://docs.astral.sh/uv/).

```sh
make install   # uv sync
make check     # lint + format-check + typecheck + test
```

Individual targets: `make lint`, `make format`, `make format-check`,
`make typecheck`, `make test`.
