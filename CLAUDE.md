# Project Specification: Decoupled Options Backtesting Engine

## 1. Objective & Core Philosophy
Build a modular, event-driven (or vectorized-hybrid), decoupled options backtesting engine in Python. The architecture must strictly separate **Data Ingestion**, **Market Regime Analysis**, **Strategy Logic**, **Execution/Portfolio Accounting**, and **Analytics/Reporting**. 

The primary goal is to evaluate multi-leg option strategies across varying market regimes with high parameter flexibility (DTE, delta targets, wing widths, profit targets, stop losses) without rewriting core engine logic when switching data providers or adding new strategies.

---

## 2. System Architecture (Decoupled Layers)

### Layer 1: Unified Data Ingestion & Normalization (`data/`)
Create an abstract base class `DataProvider` that normalizes disparate data sources into a canonical internal format (e.g., Parquet/Polars DataFrames or standardized Pydantic/Dataclass tick/bar events).

*   **Canonical Schema (`OptionChainSnapshot` & `UnderlyingBar`):**
    *   `timestamp` (UTC `datetime`)
    *   `underlying_symbol` (str), `underlying_price` (float), `iv_rank` / `iv_percentile` (optional float)
    *   `expiration` (date), `dte` (int), `strike` (float), `option_type` (`CALL` | `PUT`)
    *   `bid` (float), `ask` (float), `mid` (float), `last` (float), `volume` (int), `open_interest` (int)
    *   `implied_volatility` (float), `delta` (float), `gamma` (float), `theta` (float), `vega` (float)
*   **Required Adapters:**
    1.  **ThetaData Adapter:** Historical tick/1-min/EOD option chain & Greeks ingestion via REST/local terminal API.
    2.  **IBKR (`ib_insync` / `ibapi`) Adapter:** Historical bars and option chain snapshots; include rate-limiting/pacing violation handlers.
    3.  **Charles Schwab API Adapter:** OAuth2 token refresh handling + historical price/option chain ingestion.
    4.  **`yfinance` Adapter:** Fallback/baseline adapter for underlying OHLCV and current option chains (note: must gracefully handle missing historical Greeks by triggering the internal Greeks fallback calculator).
*   **Greeks & IV Fallback Engine:** If a provider (like `yfinance`) lacks historical Greeks or IV, automatically compute them using Black-Scholes-Merton / Barone-Adesi-Whaley (for American options) via `py_vollib` or `scipy`.

### Layer 2: Market Regime Classifier (`regimes/`)
A standalone module that tags historical timestamps with market states before or during the backtest. Strategies can query this state to filter entries or adjust sizing.

*   **Regime Dimensions:**
    *   **Trend Regime:** Bullish, Bearish, Sideways/Choppy (e.g., via SMA crossovers, ADX thresholds, or Hurst exponent).
    *   **Volatility Regime:** Low IV, Normal IV, High IV, Volatility Expansion/Contraction (e.g., VIX levels, VIX term structure contango/backwardation, or IV Rank).
*   **Output:** A time-indexed `RegimeState` object passed to the strategy engine at each evaluation step.

### Layer 3: Flexible Strategy Definition (`strategies/`)
Strategies must be declarative and parameterized, inheriting from a base `OptionStrategy` class. No hardcoded strikes or dates.

*   **Configurable Parameter Space (JSON/YAML/Dict driven):**
    *   **Time Parameters:** Target entry DTE range (e.g., `min_dte=30, max_dte=45`), entry time-of-day, max hold duration, exit DTE threshold (e.g., close at 21 DTE).
    *   **Strike Selection:** Selectable by `Delta` (e.g., short leg at `0.16 delta`), `Percentage OTM`, or `Standard Deviation`.
    *   **Wing / Structure Parameters:** Wing width defined by fixed dollar width (e.g., `$5` wide), `Delta` differential, or percentage of underlying price (for Iron Condors, Verticals, Butterflies, Calendars, Strangles).
    *   **Entry Conditions:** Allowed `RegimeState` tags, minimum IV Rank, minimum liquidity filters (max bid-ask spread, min open interest).
    *   **Exit Conditions:** Take-profit (% of max credit or % of debit paid), Stop-loss (multiple of credit received, underlying price breach, or portfolio delta breach), Trailing stops, Time-based exits.

### Layer 4: Execution, Portfolio & Risk Engine (`engine/`)
Simulates realistic fills and tracks margin/Greeks at the portfolio level.

*   **Execution & Slippage Model:** Configurable fill assumptions (Mid-price, Bid/Ask natural, or dynamic slippage model based on bid-ask spread width and volume).
*   **Transaction Costs:** Per-contract commissions + exchange/regulatory fees.
*   **Lifecycle Management:** Handle expiration settlement (cash vs. physical assignment simulation), early assignment risk flags (e.g., deep ITM short calls near ex-dividend), and multi-leg combo order atomicity.
*   **Portfolio Greeks:** Aggregate net Delta, Gamma, Theta, and Vega across all open positions at each timestamp.

### Layer 5: Output, Diagnostics & Strategy Discovery (`analytics/`)
Generate structured outputs detailing strategy performance and regime attribution.

*   **Required Outputs:**
    1.  **Strategy Blueprint Report:** Explicitly outputs the tested strategy rules, exact entry/exit triggers, parameter set, and **Ideal Market Regimes** (derived from regime-segmented Sharpe/Win-Rate attribution).
    2.  **Trade Log (CSV/Parquet):** Leg-by-leg entry/exit timestamps, prices, Greeks at entry/exit, slippage paid, PnL, and exit reason (`PROFIT_TARGET`, `STOP_LOSS`, `DTE_EXIT`, `REGIME_SHIFT`).
    3.  **Performance Metrics:** CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, Average Days in Trade, Expectancy, and Tail Risk (VaR/CVaR).
    4.  **Parameter Sensitivity Matrix:** Heatmaps/tables showing performance across parameter sweeps (e.g., `Delta` vs. `Wing Width`, or `Entry DTE` vs. `Profit Target %`).

---

## 3. Recommended Directory Structure

```text
options_backtester/
├── config/
│   ├── settings.yaml            # API keys, cache paths, default commission/slippage
│   └── strategy_grid.yaml       # Parameter sweep configurations
├── data/
│   ├── base.py                  # Abstract DataProvider & Canonical Schemas
│   ├── cache.py                 # Local Parquet caching layer to minimize API calls
│   ├── providers/
│   │   ├── thetadata.py
│   │   ├── ibkr.py
│   │   ├── schwab.py
│   │   └── yfinance_provider.py
│   └── greeks_engine.py         # Vectorized IV & Greeks fallback solver
├── regimes/
│   ├── classifier.py            # Volatility & Trend regime tagging
│   └── indicators.py            # IVR, ATR, ADX, VIX term structure helpers
├── strategies/
│   ├── base.py                  # Abstract OptionStrategy & LegBuilder
│   ├── conditions.py            # Modular Entry/Exit rule evaluators
│   └── templates/               # IronCondor, VerticalSpread, Calendar, Strangle
├── engine/
│   ├── backtester.py            # Main event loop
│   ├── execution.py             # Slippage, fill logic, commissions
│   └── portfolio.py             # Position tracking, margin, portfolio Greeks
├── analytics/
│   ├── metrics.py               # Risk/return calculations
│   ├── regime_attribution.py    # Identifies ideal market regimes for the strategy
│   └── reporter.py              # Generates Markdown/JSON strategy summaries & plots
├── tests/                       # Unit tests (Greeks accuracy, PnL math, leg selection)
└── main.py                      # CLI / Entry point for running single or grid backtests
```

---

## 4. Coding & Implementation Standards for Claude

1.  **Strict Type Hinting & Validation:** Use Python `typing` and `Pydantic` (or `dataclasses`) for all configuration schemas, order objects, and canonical data models.
2.  **Vectorization Where Possible:** Use `Polars` or `NumPy`/`Pandas` for filtering option chains and calculating Greeks. Avoid slow nested Python loops over option chains.
3.  **Local Caching First:** Every data adapter must write normalized data to a local Parquet store (`data_cache/{provider}/{symbol}/{date}.parquet`) so subsequent backtests and parameter sweeps run offline at maximum speed.
4.  **No Lookahead Bias:** Ensure regime indicators and option chain snapshots strictly use data available at or before `current_timestamp`.
5.  **Testability:** Write unit tests verifying multi-leg PnL calculations (credit vs. debit spreads), delta-based strike selection edge cases, and exit trigger priority (e.g., if both stop-loss and DTE exit trigger on the same bar).

---

## 5. Step-by-Step Implementation Plan
Please implement this codebase incrementally in the following order:
1.  **Phase 1:** Core canonical schemas (`data/base.py`), local Parquet cache, `yfinance` provider, and the Black-Scholes Greeks fallback engine.
2.  **Phase 2:** Strategy builder (`strategies/base.py`), dynamic leg selector (by Delta, DTE, and Wing width), and modular entry/exit condition evaluators.
3.  **Phase 3:** Execution simulator, portfolio tracker, and main backtest loop (`engine/`).
4.  **Phase 4:** Market regime classifier (`regimes/`) and regime-attribution reporting (`analytics/regime_attribution.py`) that outputs ideal market regimes and strategy rules.
5.  **Phase 5:** Production data adapters for **ThetaData**, **IBKR**, and **Schwab** with rate-limiting and retry logic.