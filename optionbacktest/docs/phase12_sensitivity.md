# Phase 12 parameter sensitivity

Phase 12 adds controlled parameter-sensitivity analysis. Each surface is limited
to one or two dimensions and reports every combination, not only the winner.

## Grid and objective contract

`ParameterGrid` accepts one or two ordered dimensions with non-empty, unique
values. Cartesian products retain declared order, giving deterministic
tie-breaking when objective values are equal.

`SensitivitySettings` specifies the numeric objective, whether it is maximized
or minimized, and the relative threshold for an isolated-winner warning.
Evaluators receive defensive training-data copies and immutable parameter
mappings. They must return consistent, finite numeric metrics. Results retain
the complete surface and immutable selected parameters.

## Training-only selection

`SensitivityAnalyzer.analyze()` accepts Phase 11 `FixedSamples` and evaluates
only its training frame. Validation and out-of-sample rows cannot enter fixed
sample selection.

`analyze_training()` accepts an explicitly identified training window for use
inside `WalkForwardRunner.fit`. `evaluate_selected()` then applies the fixed
training-selected parameters to validation, walk-forward test, or out-of-sample
data without reselection.

## Possible-overfitting warning

The winner is compared with immediately adjacent grid points. In two
dimensions, adjacency is one grid step along one axis. A winner is flagged when
its normalized improvement over the strongest adjacent result reaches the
configured isolation threshold. Plateaus are therefore distinguished from
isolated spikes.

This flag is a warning, not proof of overfitting, and does not replace
walk-forward or out-of-sample validation.

## Bias controls and limitations

- Surfaces cannot exceed two dimensions.
- Full surfaces are returned instead of hiding neighboring results.
- Ties are deterministic and are not presented as unique optima.
- Fixed-sample selection uses training data only.
- Validation and out-of-sample results must not be used to revise the grid after
  inspection.
- Statistical simulations and charts remain outside Phase 12.
