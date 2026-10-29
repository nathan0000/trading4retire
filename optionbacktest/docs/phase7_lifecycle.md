# Phase 7 backtest lifecycle

Phase 7 coordinates entry, discrete quote monitoring, risk triggers, and mandatory
same-session exits. It produces immutable trade records but does not calculate
portfolio statistics or performance metrics.

Every entered position records `CREATED → ENTERED → MONITORING`, then terminates as
`STOPPED`, `PROFIT_TARGET`, or `FORCED_EXIT`. `EXPIRED` is reserved explicitly, but
normal 0DTE operation must force an earlier exit. Terminal positions cannot be marked
again.

## Stop interpretation

```text
stop_loss_amount = initial_credit * stop_multiplier
stop_loss_price  = initial_credit + stop_loss_amount
```

Thus a 2.00 credit with a 0.90 multiplier stops when the close mark reaches
3.80. Entry credit, close mark, P&L, loss amount, and stop price remain separate.
Tests cover multipliers 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, and 1.50. Optional profit
taking is expressed as entry-credit fraction captured and is disabled by default.

## Monitoring and forced exit

Stops and profit targets are evaluated on `BacktestSettings.stop_trigger`. The default,
`MID`, is the per-leg mid-price cost to close, like a broker's mark-based stop, so a
momentary bid/ask widening does not stop a position out. `EXECUTABLE` evaluates on the
worst-case fill price instead. Either way, the exit fills through the per-leg
execution simulation, and the trade record keeps both `exit_trigger_mark` and
`exit_debit`. Triggers are evaluated only at supplied snapshots; intrabar crossings are
not inferred. The first observed breach closes at that snapshot's debit, not the
theoretical threshold.

A monitoring snapshot missing a quote for any open leg cannot be marked. It is skipped
and counted in the trade's `skipped_marks`, and monitoring continues.

Forced exit defaults to exactly 15:55 America/New_York. A 15:50 or last-known quote is
never treated as a 15:55 fill. A missing or incomplete 15:55 snapshot raises
`MissingForcedExitQuotes`;
next-day observations are rejected. This prevents silent overnight positions and
imaginary closing prices.

Gross P&L is `(entry_credit - exit_debit) * contract_multiplier`. Entry and exit costs
are summed for net P&L. Records retain both executions, timestamps, state history, and
exit reason. `Portfolio` is only a completed-record ledger; capital, margin, and
metrics are deferred.
