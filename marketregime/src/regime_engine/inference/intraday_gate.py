"""Decision matrix for the intraday regime gate: five indicators in, one label out.

Precedence, first match wins:

1. any indicator not yet available           -> UNKNOWN
2. ATR / opening baseline above critical      -> EXTREME_VOLATILITY
3. 20 EMA == 40 EMA                           -> WHIPSAW_NEUTRAL
4. every proxy heavily opposes the SPX EMAs   -> WHIPSAW_NEUTRAL
5. bullish EMAs, close > VWAP, RSI < high     -> BULL_TREND
   bearish EMAs, close < VWAP, RSI > low      -> BEAR_TREND
6. EMAs crossed but VWAP or RSI disagrees     -> WHIPSAW_NEUTRAL

Pure function: no logging, clock, or I/O, so every rule boundary is unit-testable.
"""

from dataclasses import dataclass

from regime_engine.domain.intraday import IntradayRegimeConfig, PrimaryRegime, VolatilityState


@dataclass(frozen=True, slots=True)
class GateInputs:
    close: float
    ema_fast: float | None
    ema_slow: float | None
    vwap: float | None
    rsi: float | None
    volatility_state: VolatilityState
    volatility_ratio: float | None
    proxy_bias_bps: tuple[tuple[str, float], ...]  # (column, EMA spread in bp)


@dataclass(frozen=True, slots=True)
class GateDecision:
    regime: PrimaryRegime
    internal_bias_match: bool
    reasons: tuple[str, ...]


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _rejections(inputs: GateInputs, trend: int, config: IntradayRegimeConfig) -> list[str]:
    """Why VWAP or RSI disagree with the EMA direction; empty when both confirm."""
    assert inputs.vwap is not None and inputs.rsi is not None
    rejections: list[str] = []
    if trend > 0:
        if not inputs.close > inputs.vwap:
            rejections.append("below-VWAP structural price action")
        if not inputs.rsi < config.rsi_overbought:
            rejections.append(
                f"exhausted momentum (RSI {inputs.rsi:.1f} >= {config.rsi_overbought:g})"
            )
    else:
        if not inputs.close < inputs.vwap:
            rejections.append("above-VWAP structural price action")
        if not inputs.rsi > config.rsi_oversold:
            rejections.append(
                f"overextended momentum (RSI {inputs.rsi:.1f} <= {config.rsi_oversold:g})"
            )
    return rejections


def classify_intraday(inputs: GateInputs, config: IntradayRegimeConfig) -> GateDecision:
    pending = [
        name
        for name, value in (
            (f"{config.ema_fast} EMA", inputs.ema_fast),
            (f"{config.ema_slow} EMA", inputs.ema_slow),
            ("VWAP", inputs.vwap),
            (f"RSI({config.rsi_period})", inputs.rsi),
        )
        if value is None
    ]
    if inputs.volatility_state is VolatilityState.UNKNOWN:
        pending.append("ATR baseline")
    if pending:
        return GateDecision(
            PrimaryRegime.UNKNOWN, False, (f"Indicators not available: {', '.join(pending)}",)
        )
    # Narrowed by the pending check above; asserts keep mypy honest without casts.
    assert inputs.ema_fast is not None and inputs.ema_slow is not None
    assert inputs.vwap is not None and inputs.rsi is not None

    trend = _sign(inputs.ema_fast - inputs.ema_slow)
    proxies = inputs.proxy_bias_bps
    bias_match = bool(proxies) and trend != 0 and all(_sign(bps) == trend for _, bps in proxies)
    proxy_text = ", ".join(f"{name} {bps:+.1f}bp" for name, bps in proxies)

    if inputs.volatility_state is VolatilityState.CRITICAL:
        return GateDecision(
            PrimaryRegime.EXTREME_VOLATILITY,
            bias_match,
            (
                f"ATR at {inputs.volatility_ratio:.2f}x the opening baseline exceeds "
                f"{config.critical_ratio:.2f}x; short-premium gamma risk is outsized",
            ),
        )
    if trend == 0:
        return GateDecision(
            PrimaryRegime.WHIPSAW_NEUTRAL, False, ("20/40 EMA flat; no trend direction",)
        )

    word = "Bullish" if trend > 0 else "Bearish"
    if proxies and all(
        _sign(bps) == -trend and abs(bps) >= config.proxy_conflict_bps for _, bps in proxies
    ):
        opposite = "bearish" if trend > 0 else "bullish"
        return GateDecision(
            PrimaryRegime.WHIPSAW_NEUTRAL,
            False,
            (f"{word} EMA detected but rejected by {opposite} proxy internals ({proxy_text})",),
        )

    rejections = _rejections(inputs, trend, config)
    if rejections:
        return GateDecision(
            PrimaryRegime.WHIPSAW_NEUTRAL,
            bias_match,
            (f"{word} EMA detected but rejected by {' and '.join(rejections)}",),
        )

    if not proxies:
        internals = "no proxy data, internals unconfirmed"
    elif bias_match:
        internals = f"proxies confirm ({proxy_text})"
    else:
        internals = f"proxies mixed ({proxy_text})"
    side = "above" if trend > 0 else "below"
    return GateDecision(
        PrimaryRegime.BULL_TREND if trend > 0 else PrimaryRegime.BEAR_TREND,
        bias_match,
        (
            f"{word} EMA confirmed {side} VWAP, RSI {inputs.rsi:.1f}, "
            f"volatility {inputs.volatility_state.value}; {internals}",
        ),
    )
