"""Opt-in Black--Scholes fallback for incomplete historical option chains."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from math import exp, isfinite, log, sqrt

import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class GreeksFallbackSettings:
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    expiration_time: time = time(16, 0)
    minimum_volatility: float = 1e-4
    maximum_volatility: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.risk_free_rate,
            self.dividend_yield,
            self.minimum_volatility,
            self.maximum_volatility,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Greeks fallback settings must be finite")
        if self.minimum_volatility <= 0 or self.maximum_volatility <= self.minimum_volatility:
            raise ValueError("volatility bounds must be positive and increasing")


def fill_missing_greeks(
    quotes: pd.DataFrame, settings: GreeksFallbackSettings | None = None
) -> pd.DataFrame:
    """Fill missing IV/Greeks from same-row historical inputs without overwriting values."""
    required = {
        "timestamp",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "underlying_price",
        "delta",
        "gamma",
        "theta",
        "vega",
        "implied_volatility",
    }
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Greeks fallback missing required columns: {sorted(missing)}")
    assumptions = settings or GreeksFallbackSettings()
    result = quotes.copy()
    estimated = pd.Series(False, index=result.index)
    for index, row in result.iterrows():
        fields = ["delta", "gamma", "theta", "vega", "implied_volatility"]
        if row[fields].notna().all():
            continue
        inputs = _inputs(row, assumptions)
        if inputs is None:
            continue
        spot, strike, years, option_type = inputs
        volatility = _number(row["implied_volatility"])
        if volatility is None or volatility <= 0:
            bid, ask = _number(row["bid"]), _number(row["ask"])
            if bid is None or ask is None:
                continue
            volatility = _solve_iv((bid + ask) / 2, spot, strike, years, option_type, assumptions)
        if volatility is None:
            continue
        values = _greeks(spot, strike, years, volatility, option_type, assumptions)
        if pd.isna(row["implied_volatility"]):
            result.at[index, "implied_volatility"] = volatility
        for field, value in values.items():
            if pd.isna(row[field]):
                result.at[index, field] = value
        estimated.at[index] = True
    if estimated.any():
        if "source" not in result:
            result["source"] = pd.NA
        result.loc[estimated, "source"] = result.loc[estimated, "source"].apply(
            lambda value: (
                "ESTIMATED_BLACK_SCHOLES_GREEKS"
                if pd.isna(value) or not str(value)
                else f"{value}|ESTIMATED_BLACK_SCHOLES_GREEKS"
            )
        )
    return result


def _inputs(row: pd.Series, settings: GreeksFallbackSettings):
    timestamp, expiration = pd.Timestamp(row["timestamp"]), pd.Timestamp(row["expiration"])
    if pd.isna(timestamp) or pd.isna(expiration) or timestamp.tzinfo is None:
        return None
    expiry = pd.Timestamp.combine(expiration.date(), settings.expiration_time).tz_localize(
        timestamp.tz
    )
    years = (expiry - timestamp).total_seconds() / (365 * 24 * 60 * 60)
    spot, strike = _number(row["underlying_price"]), _number(row["strike"])
    option_type = {"C": "CALL", "P": "PUT"}.get(
        str(row["option_type"]).upper(), str(row["option_type"]).upper()
    )
    if (
        spot is None
        or strike is None
        or spot <= 0
        or strike <= 0
        or years <= 0
        or option_type not in {"CALL", "PUT"}
    ):
        return None
    return spot, strike, years, option_type


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _price(spot, strike, years, volatility, option_type, settings):
    d1 = (
        log(spot / strike)
        + (settings.risk_free_rate - settings.dividend_yield + volatility**2 / 2) * years
    ) / (volatility * sqrt(years))
    d2 = d1 - volatility * sqrt(years)
    ds, dk = (
        spot * exp(-settings.dividend_yield * years),
        strike * exp(-settings.risk_free_rate * years),
    )
    return (
        ds * norm.cdf(d1) - dk * norm.cdf(d2)
        if option_type == "CALL"
        else dk * norm.cdf(-d2) - ds * norm.cdf(-d1)
    )


def _solve_iv(market_price, spot, strike, years, option_type, settings):
    if not isfinite(market_price) or market_price <= 0:
        return None

    def objective(volatility: float) -> float:
        return _price(spot, strike, years, volatility, option_type, settings) - market_price

    low, high = settings.minimum_volatility, settings.maximum_volatility
    if objective(low) * objective(high) > 0:
        return None
    try:
        return float(brentq(objective, low, high, xtol=1e-10, maxiter=100))
    except ValueError:
        return None


def _greeks(spot, strike, years, volatility, option_type, settings):
    root = sqrt(years)
    d1 = (
        log(spot / strike)
        + (settings.risk_free_rate - settings.dividend_yield + volatility**2 / 2) * years
    ) / (volatility * root)
    d2 = d1 - volatility * root
    ds, dk, density = (
        exp(-settings.dividend_yield * years),
        exp(-settings.risk_free_rate * years),
        norm.pdf(d1),
    )
    theta = -(spot * ds * density * volatility) / (2 * root)
    if option_type == "CALL":
        delta = ds * norm.cdf(d1)
        theta += -settings.risk_free_rate * strike * dk * norm.cdf(d2)
        theta += settings.dividend_yield * spot * ds * norm.cdf(d1)
    else:
        delta = ds * (norm.cdf(d1) - 1)
        theta += settings.risk_free_rate * strike * dk * norm.cdf(-d2)
        theta -= settings.dividend_yield * spot * ds * norm.cdf(-d1)
    return {
        "delta": float(delta),
        "gamma": float(ds * density / (spot * volatility * root)),
        "theta": float(theta / 365),
        "vega": float(spot * ds * density * root / 100),
    }
