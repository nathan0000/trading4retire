"""Strategy layer: declarative configs, leg selection and entry/exit rules."""

from __future__ import annotations

from typing import Any

from options_backtester.strategies.base import OptionStrategy, StrategyConfig
from options_backtester.strategies.templates import Calendar, IronCondor, Strangle, VerticalSpread

TEMPLATES: dict[str, type[OptionStrategy]] = {
    "iron_condor": IronCondor,
    "vertical": VerticalSpread,
    "strangle": Strangle,
    "calendar": Calendar,
}


def build_strategy(config: StrategyConfig | dict[str, Any]) -> OptionStrategy:
    parsed = config if isinstance(config, StrategyConfig) else StrategyConfig.model_validate(config)
    return TEMPLATES[parsed.template](parsed)


__all__ = ["TEMPLATES", "OptionStrategy", "StrategyConfig", "build_strategy"]
