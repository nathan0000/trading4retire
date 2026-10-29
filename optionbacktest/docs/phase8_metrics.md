# Phase 8 P&L and risk metrics

Phase 8 consumes completed `TradeRecord` objects or an equivalent typed DataFrame. It
does not tune parameters, classify regimes, or infer unavailable market factors.

Net P&L drives win rate, expectancy, equity, drawdown, streaks, Sharpe, Sortino, VaR,
CVaR, and grouped results. Gross P&L remains separately reported. Breakeven trades are
neither wins nor losses. Profit factor is positive P&L divided by absolute negative
P&L; it is infinity for a positive sample without losses and `NaN` when undefined.

Sharpe and Sortino use daily dollar P&L and annualize by `sqrt(252)`. They describe P&L
stability, not capital returns. Undefined or zero-variance cases remain explicit as
`NaN` or infinity.

Trades are ordered by exit timestamp with entry timestamp as a stable tie-breaker.
Equity is cumulative net P&L with its high-water mark anchored at zero. Maximum
drawdown is reported as a positive magnitude. Duration is the maximum number of
consecutive completed trades below the prior high-water mark, not calendar time.

Daily summaries use New York exit dates. Worst week/month use those local dates.
Consecutive losing trades and consecutive losing days are separate measures.

Historical 95% VaR is the positive loss magnitude at the fifth percentile of trade
P&L. CVaR is the positive magnitude of mean P&L at or below that threshold. No
distribution is assumed.

Entry-time analysis emits the full 10:45–15:15 schedule. Delta analysis uses actual
short-leg absolute deltas, never target delta. Expected-move buckets require an
explicit normalized distance. Generic grouping supports later VIX/regime fields only
when provided; missing factors are never fabricated.
