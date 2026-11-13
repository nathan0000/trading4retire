"""Vectorized IV and Greeks fallback: Black-Scholes-Merton and Barone-Adesi-Whaley.

All functions broadcast over NumPy arrays. ``theta`` is per calendar day and
``vega`` is per one volatility point (0.01), matching common vendor conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import ndtr

from options_backtester.data.base import GREEK_COLUMNS, MARKET_CLOSE, MARKET_TZ

Model = Literal["bsm", "baw"]
_SECONDS_PER_YEAR = 365.0 * 24 * 3600
_BISECTIONS = 60


@dataclass(frozen=True, slots=True)
class GreeksConfig:
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    #: "bsm" for European-style (index) options, "baw" for American-style (equity/ETF).
    model: Model = "bsm"
    min_volatility: float = 1e-3
    max_volatility: float = 5.0

    def __post_init__(self) -> None:
        if self.model not in ("bsm", "baw"):
            raise ValueError("model must be 'bsm' or 'baw'")
        if not 0 < self.min_volatility < self.max_volatility:
            raise ValueError("volatility bounds must be positive and increasing")


def _pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _d1_d2(s, k, t, r, q, sigma):
    vol_t = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_t
    return d1, d1 - vol_t


def bsm_price(s, k, t, r, q, sigma, is_call) -> np.ndarray:
    s, k, t, sigma = (np.asarray(x, dtype=float) for x in (s, k, t, sigma))
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    ds, dk = s * np.exp(-q * t), k * np.exp(-r * t)
    call = ds * ndtr(d1) - dk * ndtr(d2)
    return np.where(is_call, call, call - ds + dk)


def bsm_greeks(s, k, t, r, q, sigma, is_call) -> dict[str, np.ndarray]:
    s, k, t, sigma = (np.asarray(x, dtype=float) for x in (s, k, t, sigma))
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    root_t = np.sqrt(t)
    eq, er, density = np.exp(-q * t), np.exp(-r * t), _pdf(d1)
    decay = -(s * eq * density * sigma) / (2.0 * root_t)
    call_theta = decay - r * k * er * ndtr(d2) + q * s * eq * ndtr(d1)
    put_theta = decay + r * k * er * ndtr(-d2) - q * s * eq * ndtr(-d1)
    return {
        "delta": np.where(is_call, eq * ndtr(d1), eq * (ndtr(d1) - 1.0)),
        "gamma": eq * density / (s * sigma * root_t),
        "theta": np.where(is_call, call_theta, put_theta) / 365.0,
        "vega": s * eq * density * root_t / 100.0,
    }


def _bisect_log(f, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Root of ``f`` (negative at ``lo``, positive at ``hi``) by bisection in log space."""
    lo, hi = np.log(lo), np.log(hi)
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        positive = f(np.exp(mid)) > 0
        hi = np.where(positive, mid, hi)
        lo = np.where(positive, lo, mid)
    return np.exp(0.5 * (lo + hi))


def baw_price(s, k, t, r, q, sigma, is_call) -> np.ndarray:
    """Barone-Adesi-Whaley (1987) quadratic approximation for American options."""
    s, k, t, sigma, is_call = np.broadcast_arrays(
        np.asarray(s, dtype=float),
        np.asarray(k, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(is_call, dtype=bool),
    )
    out = np.array(bsm_price(s, k, t, r, q, sigma, is_call), dtype=float)
    if r <= 0:  # no early-exercise premium on puts; calls handled by q below
        if q <= 0:
            return out
    b = r - q
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        m = 2.0 * r / sigma**2
        n = 2.0 * b / sigma**2
        kc = 1.0 - np.exp(-r * t)
        root = np.sqrt((n - 1.0) ** 2 + 4.0 * m / kc)
        q2 = (-(n - 1.0) + root) / 2.0
        q1 = (-(n - 1.0) - root) / 2.0
        carry = np.exp((b - r) * t)

        calls = is_call & (q > 0) & (r > 0)
        if calls.any():
            kk, tt, ss, sg, qq = k[calls], t[calls], s[calls], sigma[calls], q2[calls]
            cc = carry[calls]

            def f_call(x):
                d1, _ = _d1_d2(x, kk, tt, r, q, sg)
                euro = bsm_price(x, kk, tt, r, q, sg, True)
                return x - kk - euro - (1.0 - cc * ndtr(d1)) * x / qq

            hi = kk * 1e3
            has_boundary = f_call(hi) > 0
            star = _bisect_log(f_call, kk, hi)
            d1, _ = _d1_d2(star, kk, tt, r, q, sg)
            a2 = (star / qq) * (1.0 - cc * ndtr(d1))
            american = np.where(ss < star, out[calls] + a2 * (ss / star) ** qq, ss - kk)
            out[calls] = np.where(has_boundary, american, out[calls])

        puts = ~is_call & (r > 0)
        if puts.any():
            kk, tt, ss, sg, qq = k[puts], t[puts], s[puts], sigma[puts], q1[puts]
            cc = carry[puts]

            def f_put(x):  # increasing in x: positive near K, negative near 0
                d1, _ = _d1_d2(x, kk, tt, r, q, sg)
                euro = bsm_price(x, kk, tt, r, q, sg, False)
                return -(kk - x - euro + (1.0 - cc * ndtr(-d1)) * x / qq)

            star = _bisect_log(f_put, kk * 1e-6, kk)
            d1, _ = _d1_d2(star, kk, tt, r, q, sg)
            a1 = -(star / qq) * (1.0 - cc * ndtr(-d1))
            out[puts] = np.where(ss > star, out[puts] + a1 * (ss / star) ** qq, kk - ss)
    return out


def price(s, k, t, r, q, sigma, is_call, model: Model = "bsm") -> np.ndarray:
    fn = baw_price if model == "baw" else bsm_price
    return fn(s, k, t, r, q, sigma, is_call)


def greeks(s, k, t, r, q, sigma, is_call, model: Model = "bsm") -> dict[str, np.ndarray]:
    if model == "bsm":
        return bsm_greeks(s, k, t, r, q, sigma, is_call)
    s, k, t, sigma = (np.asarray(x, dtype=float) for x in (s, k, t, sigma))
    h = s * 1e-3
    base = baw_price(s, k, t, r, q, sigma, is_call)
    up, down = (baw_price(s + d, k, t, r, q, sigma, is_call) for d in (h, -h))
    dv = 0.01
    vol_up = baw_price(s, k, t, r, q, sigma + dv, is_call)
    vol_down = baw_price(s, k, t, r, q, np.maximum(sigma - dv, 1e-4), is_call)
    dt = np.minimum(1.0 / 365.0, t / 2.0)
    later = baw_price(s, k, t - dt, r, q, sigma, is_call)
    return {
        "delta": (up - down) / (2.0 * h),
        "gamma": (up - 2.0 * base + down) / (h * h),
        "theta": (later - base) / (dt * 365.0),
        "vega": (vol_up - vol_down) / 2.0,
    }


def implied_volatility(
    market_price,
    s,
    k,
    t,
    r,
    q,
    is_call,
    model: Model = "bsm",
    min_volatility: float = 1e-3,
    max_volatility: float = 5.0,
) -> np.ndarray:
    """Solve IV by bisection; NaN where the price violates no-arbitrage bounds."""
    market_price, s, k, t, is_call = np.broadcast_arrays(
        np.asarray(market_price, dtype=float),
        np.asarray(s, dtype=float),
        np.asarray(k, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(is_call, dtype=bool),
    )
    lo = np.full(market_price.shape, min_volatility)
    hi = np.full(market_price.shape, max_volatility)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        valid = (
            (market_price > 0)
            & (t > 0)
            & (price(s, k, t, r, q, lo, is_call, model) <= market_price)
            & (price(s, k, t, r, q, hi, is_call, model) >= market_price)
        )
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            above = price(s, k, t, r, q, mid, is_call, model) > market_price
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
    return np.where(valid, 0.5 * (lo + hi), np.nan)


def years_to_expiry(timestamp: pd.Series, expiration: pd.Series) -> np.ndarray:
    """Year fraction from each UTC timestamp to the 16:00 New York close on expiration."""
    close = pd.to_timedelta(MARKET_CLOSE.hour * 3600 + MARKET_CLOSE.minute * 60, unit="s")
    expiry = (expiration + close).dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
    return ((expiry - timestamp).dt.total_seconds() / _SECONDS_PER_YEAR).to_numpy()


def fill_missing_greeks(frame: pd.DataFrame, config: GreeksConfig | None = None) -> pd.DataFrame:
    """Fill missing IV/Greeks on a canonical chain without overwriting provider values.

    Rows that are filled are labelled ``ESTIMATED_BSM`` / ``ESTIMATED_BAW`` in
    ``greeks_source``. Rows whose inputs cannot support a solution stay NaN and are
    labelled ``UNAVAILABLE``.
    """
    cfg = config or GreeksConfig()
    fields = ["implied_volatility", *GREEK_COLUMNS]
    need = frame[fields].isna().any(axis=1).to_numpy()
    if not need.any():
        return frame
    out = frame.copy()
    t = years_to_expiry(out["timestamp"], out["expiration"])
    s = out["underlying_price"].to_numpy(dtype=float)
    k = out["strike"].to_numpy(dtype=float)
    is_call = (out["option_type"] == "CALL").to_numpy()
    iv = out["implied_volatility"].to_numpy(dtype=float).copy()
    usable = need & (t > 0) & (s > 0) & (k > 0)
    solve = usable & ~(iv > 0)
    r, q = cfg.risk_free_rate, cfg.dividend_yield
    if solve.any():
        iv[solve] = implied_volatility(
            out["mid"].to_numpy(dtype=float)[solve],
            s[solve],
            k[solve],
            t[solve],
            r,
            q,
            is_call[solve],
            cfg.model,
            cfg.min_volatility,
            cfg.max_volatility,
        )
    done = usable & (iv > 0)
    source = out["greeks_source"].to_numpy(dtype=object).copy()
    # Rows the model cannot solve (e.g. mid below intrinsic) stay NaN and say so.
    source[need & ~done] = "UNAVAILABLE"
    if done.any():
        values = greeks(s[done], k[done], t[done], r, q, iv[done], is_call[done], cfg.model)
        values["implied_volatility"] = iv[done]
        rows = np.flatnonzero(done)
        for name in fields:
            column = out[name].to_numpy(dtype=float).copy()
            current = column[rows]
            column[rows] = np.where(np.isnan(current), values[name], current)
            out[name] = column
        source[rows] = f"ESTIMATED_{cfg.model.upper()}"
    out["greeks_source"] = source
    return out
