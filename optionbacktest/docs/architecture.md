# Architecture and design decisions

## Phase 1 boundaries

Phase 1 ends at validated, canonical SPX five-minute bars. It deliberately contains
no indicators, option pricing, strategy selection, execution simulation, or P&L.
This is the first test gate required by `AGENTS.md`.

The loader owns source-format parsing. The validator owns quality decisions. The
session service owns exchange schedules. Downstream components will therefore consume
one canonical representation rather than repeat vendor-specific assumptions.

## Time and sessions

- Canonical timestamps are timezone-aware and converted to `America/New_York`.
- A source containing naive timestamps is rejected unless `source_timezone` is set.
- Bar timestamps must explicitly mean either bar start or bar end.
- XNYS schedules come from `exchange-calendars`, including holidays and early closes.
- Missing-bar checks cover gaps between observed bars. They do not claim that bars
  outside an intentionally supplied partial range are missing.
- Date cutoffs are inclusive and must themselves be timezone-aware.

These decisions prevent host timezone, daylight-saving changes, or a full-day aggregate
from changing information available at a timestamp.

## Validation

Validation produces stable reason codes and returns accepted rows separately from its
report. Invalid rows are never silently filled, deduplicated, or coerced into plausible
market observations. An unsorted input can be normalized for consumers but is still
reported as a source-quality error.

## Known data assumption

SPX index feeds frequently lack meaningful traded volume. Volume is therefore nullable.
No VWAP will be implemented until a defensible volume-bearing source or explicitly
labeled proxy is selected.

