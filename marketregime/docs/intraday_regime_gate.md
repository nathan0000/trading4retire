# Intraday regime gate (`MarketRegimeSnapshot`)

A five-indicator, single-session regime label for intraday SPX trend-following
consumers. It is narrower than `RegimeSnapshot`: it reads only today's regular-session
bars and does not produce axis scores or confidence. The multi-axis snapshot is unchanged.

```python
from regime_engine.application.intraday_regime import MarketRegimeSnapshot
from regime_engine.domain.intraday import IntradayRegimeConfig

engine = MarketRegimeSnapshot(IntradayRegimeConfig(volume_column="spy_volume"))
payload = engine.evaluate(bars, as_of)  # bars: dict of columns or a pandas DataFrame
```

| Module | Contents |
| --- | --- |
| `domain/intraday.py` | enums, `IntradayRegimeConfig`, `RegimeSnapshotPayload` |
| `features/trend.py` | `ema_last`, `rsi_last` |
| `features/intraday_structure.py` | `session_vwap` |
| `features/realized_vol.py` | `true_range`, `atr_last` |
| `inference/intraday_gate.py` | `classify_intraday`, the pure decision matrix |
| `application/intraday_regime.py` | `MarketRegimeSnapshot`: session slicing, logging, thread safety |

## Input

| Column | Required | Meaning |
| --- | --- | --- |
| `timestamp` | yes | tz-aware datetime; bar start by default (`timestamp_semantics`) |
| `open`, `high`, `low`, `close` | yes | SPX OHLC |
| `volume_column` (default `volume`) | for VWAP | volume weighting VWAP |
| each of `proxy_columns` (default `spy_close`, `qqq_close`) | no | proxy closes on the same timestamps |

## Point-in-time rules

- Only bars dated on `as_of`'s exchange day and starting at or after 09:30 ET are used.
  Every indicator, VWAP included, therefore anchors at the open. Prior sessions and
  pre-market bars are ignored.
- A bar is used only after it has completed: `bar_start + bar_minutes <= as_of`.
  The bar still forming at `as_of`, and every later row in the table, are ignored.
  Tests overwrite those rows with garbage and assert the payload is identical.
- The following data problems yield `UNKNOWN` with the reason in `reasons`, rather
  than a guess or an exception:
  - no 09:30 bar, or a gap between bars;
  - a non-finite or inconsistent OHLC value;
  - a naive timestamp;
  - a missing column.

  A naive `as_of` is a caller bug and raises `ValueError`.

## Formulas and warm-up

| Indicator | Formula | First available (1-min bars) |
| --- | --- | --- |
| EMA 20 / 40 | seeded with the SMA of the first N closes, then α = 2/(N+1) | 09:50 / 10:10 |
| VWAP | Σ(typical × volume) / Σ volume, typical = (H+L+C)/3; `None` if Σ volume = 0 | 09:31 |
| RSI 14 | Wilder: seed with the mean of the first 14 gains and losses, then α = 1/14 | 09:45 |
| ATR 14 | Wilder over true range; the first bar's TR is H−L | 09:44 |
| Opening baseline | mean true range of the bars in [09:30, 09:30 + `baseline_minutes`) | 10:00 |
| Volatility ratio | ATR / baseline: > 1.5 is `CRITICAL`, > 1.2 is `EXPANDING`, otherwise `NORMAL` | 10:00 |
| Proxy bias | (EMA20 − EMA40) / EMA40 × 10⁴, in basis points, per proxy | 10:10 |

With 5-minute bars the 40 EMA needs 200 minutes, so it is first available at 12:50 ET.
Use 1-minute bars unless a late start is intended.

## Decision matrix

Rules are applied in order, and the first match wins:

1. Any indicator unavailable: `UNKNOWN`.
2. Volatility `CRITICAL`: `EXTREME_VOLATILITY`.
3. EMA20 = EMA40: `WHIPSAW_NEUTRAL`.
4. Every proxy is on the opposite side by at least `proxy_conflict_bps` (3 bp): `WHIPSAW_NEUTRAL`.
5. Bullish EMAs with close > VWAP and RSI < 70: `BULL_TREND`. Bearish EMAs with close < VWAP and RSI > 30: `BEAR_TREND`.
6. Otherwise, the EMAs are crossed but VWAP or RSI disagrees: `WHIPSAW_NEUTRAL`.

The two derived booleans:

- `ema_gate_open` is true exactly when the label is `BULL_TREND` or `BEAR_TREND`.
- `internal_bias_match` is true when at least one proxy is present and every proxy's
  EMA spread has the same sign as SPX's.

## Assumptions and deviations from the original brief

- **Added `UNKNOWN` members.** Every enum has one, and `rsi_status` and `spx_spot` are
  optional. Warm-up and bad data are therefore explicit (CLAUDE.md 2.6) rather than
  showing up as a fake neutral reading.
- **Cash SPX usually has no volume.** Set `volume_column` to a real volume series such
  as SPY or ES. Without one, VWAP and the label are `UNKNOWN`. Nothing is substituted
  silently.
- **The volatility measure is ATR**, not Bollinger Band Width: the brief allows either,
  and only one is used.
- **Mild proxy disagreement does not veto a trend.** Only a unanimous opposition of at
  least 3 bp does; mild disagreement sets `internal_bias_match = false`. The brief says
  "heavily bearish" without giving a number, so 3 bp is a placeholder pending calibration.
- **The `EXPANDING` threshold of 1.2× is a placeholder.** The brief defines only the
  1.5× critical threshold.
- **The payload is a frozen `dataclass(slots=True)`** as the brief asks, not a pydantic
  model. It is not yet part of the JSON-schema contract.
