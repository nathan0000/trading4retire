# Phase 3 historical option data

Phase 3 defines provider-neutral historical SPX option snapshots. CSV and Parquet
loaders normalize source records into the same contract and never calculate, forward
fill, or substitute Greeks. Current Greeks must never replace historical values.

Required fields are `timestamp`, `expiration`, `strike`, `option_type`, `bid`, `ask`,
`last`, `volume`, `open_interest`, `delta`, `gamma`, `theta`, `vega`, and
`implied_volatility`. `contract_symbol` and `source` are strongly recommended for
lineage and unambiguous quote identity.

The validator requires timezone-aware cash-session snapshots, same-day expiration,
positive strikes, canonical option types, non-crossed bid/ask markets, and complete
finite Greeks and IV. Zero bids are explicitly rejected as non-executable. Duplicate
contract snapshots and expired or incorrectly dated contracts retain stable rejection
codes. Inclusive timestamp cutoffs cannot admit future snapshots.

No strike selection, quote matching, pricing, or strategy construction is part of
Phase 3. The bundled option file is synthetic test/demo input, not historical data, and
must never support a performance claim.
