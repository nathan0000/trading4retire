# Phase 4 strike selection

Phase 4 selects option strikes from already validated historical snapshots. It does
not create a trade, calculate credit, apply entry filters, or simulate fills.

The default immutable `StrikeSelectionSettings` are:

```text
target absolute delta: 0.10
minimum absolute delta: 0.08
maximum absolute delta: 0.15
wing width:             30 SPX points
```

Selection rules:

1. Require the requested expiration to equal the New York snapshot date.
2. Filter quotes to the exact timestamp and expiration. There is no last-known,
   nearest-time, or future-quote fallback.
3. Require signed OTM candidates: negative-delta puts below SPX and positive-delta
   calls above SPX.
4. Rank candidates by absolute distance from target delta.
5. Resolve equal delta distances deterministically by preferring the more OTM strike,
   then strike and contract identity. Delta distances are rounded only for stable
   comparison of mathematically equal decimal distances; stored deltas are unchanged.
6. Require a same-snapshot, same-expiration wing exactly 30 points farther OTM. A
   candidate without its exact wing is skipped in favor of the next ranked valid
   candidate. Approximate wing widths are never accepted.

Failures are returned as structured values: `NO_OPTION_DATA`, `NO_VALID_STRIKE`,
`NO_VALID_WING`, or `INCORRECT_EXPIRATION`. A failed result contains no partial legs.

The selector preserves the actual signed deltas and exact quote records chosen. These
records will be consumed by trade construction in Phase 5.
