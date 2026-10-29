# Phase 6 execution simulation

Phase 6 converts an accepted candidate or an existing set of legs into exact per-leg
fills. It does not manage positions, evaluate stops, choose an exit time, or calculate
complete trade P&L.

## Modes

- `MID`: buys and sells fill at each leg's midpoint. This is a comparison mode only.
- `BID_ASK`: buys fill at ask and sells at bid.
- `BID_ASK_PLUS_SLIPPAGE`: buys fill at ask plus configured slippage and sells at bid
  minus configured slippage, floored at zero.

The default is `BID_ASK_PLUS_SLIPPAGE` with 0.05 SPX points per leg. Slippage is
applied to each contract independently; it is never applied once to a net package
price.

## Signed cash flow

Each fill records role, side, quantity, historical quote, fill price, and signed point
cash flow. Sells are positive cash flow and buys are negative. Entry results expose a
positive `entry_credit_points` when the structure is sold for a net credit. Exit
results expose `exit_debit_points` as the negation of exit cash flow.

Exits reverse each original leg and require exactly one quote for every contract, all
from one snapshot later than entry. Missing, duplicate, mixed-timestamp, and non-later
exit quotes are rejected rather than approximated.

## Costs

The configurable cost model charges commission and exchange fees per contract per
leg. Point cash flow is converted with the configurable SPX contract multiplier,
default 100. Costs are reported separately and deducted once from gross dollar cash
flow. Later lifecycle logic must combine entry and exit results, including costs from
both sides, to obtain trade P&L.

This phase stores no claim of real execution. Its fills are deterministic simulations
under the selected assumptions.
