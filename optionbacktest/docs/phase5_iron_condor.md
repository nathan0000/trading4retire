# Phase 5 iron-condor construction

Phase 5 turns a point-in-time `MarketContext` into an immutable `TradeCandidate`.
It does not enter or monitor a position and does not claim that a quoted value was
filled.

## Point-in-time context

`MarketContext` contains the decision timestamp, same-day expiration, current SPX
price, validated option snapshots, and feature metadata known at that timestamp. It:

- requires timezone-aware timestamps;
- rejects option quotes later than the decision timestamp;
- rejects feature cutoffs later than the decision timestamp;
- defensively copies option data and freezes feature mappings.

## Entry schedule

Eligible timestamps are every 15 minutes from 10:45 through 15:15 New York time,
inclusive. UTC or other aware timestamps are converted before schedule evaluation.
Non-grid timestamps return `OUTSIDE_TRADING_WINDOW` without selecting legs.

## Quoted credit

The candidate records both required credit views in SPX points:

```text
conservative = short put bid - long put ask
             + short call bid - long call ask

mid = short put midpoint - long put midpoint
    + short call midpoint - long call midpoint
```

The conservative value is used for the inclusive default credit filter of 1.00–2.50.
Rejected credit candidates retain both credit calculations but no partial legs. An
accepted candidate records all four exact historical quotes and actual short deltas.

`max_profit_points` equals conservative quoted credit and `max_loss_points` equals
wing width minus that credit. These are structural point values, not realized dollar
P&L. Contract multipliers, fills, slippage, commissions, and fees belong to Phase 6.

## Rejections and extension

Selection failures map to `NO_OPTION_DATA`, `NO_VALID_STRIKE`, or
`INCORRECT_EXPIRATION`. Credit failures map to `CREDIT_TOO_LOW` and
`CREDIT_TOO_HIGH`. Failed candidates contain no legs, preventing accidental execution
of incomplete structures.

`StrategyEngine` dispatches through the common `generate_trade(context)` protocol and
currently registers only `IRON_CONDOR`. Later strategies can be registered without
putting their rules into the dispatcher.
