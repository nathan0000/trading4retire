"""Factor suitability scoring and descriptive performance attribution."""

from spx_0dte_backtest.analysis.factor_analysis import factor_performance
from spx_0dte_backtest.analysis.factor_scoring import (
    FactorScore,
    FactorScoreEngine,
    FactorScoreSettings,
)
from spx_0dte_backtest.analysis.sensitivity import (
    ParameterGrid,
    SensitivityAnalyzer,
    SensitivityResult,
    SensitivitySettings,
    evaluate_selected,
)
from spx_0dte_backtest.analysis.walk_forward import (
    FixedSamples,
    FixedSplitSettings,
    SampleLabel,
    WalkForwardResult,
    WalkForwardRunner,
    WalkForwardSettings,
    WalkForwardWindow,
    split_fixed_samples,
)

__all__ = [
    "FactorScore",
    "FactorScoreEngine",
    "FactorScoreSettings",
    "factor_performance",
    "ParameterGrid",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "SensitivitySettings",
    "evaluate_selected",
    "FixedSamples",
    "FixedSplitSettings",
    "SampleLabel",
    "WalkForwardResult",
    "WalkForwardRunner",
    "WalkForwardSettings",
    "WalkForwardWindow",
    "split_fixed_samples",
]
