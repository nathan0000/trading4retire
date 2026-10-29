# Phase 11 chronological and walk-forward testing

Phase 11 introduces explicit sample separation and rolling evaluation. It does
not define a parameter search space or select strategy parameters; those belong
to Phase 12.

## Fixed research samples

`FixedSplitSettings` defines four ordered calendar dates and therefore three
contiguous, half-open samples:

```text
[start, validation_start)             IN_SAMPLE
[validation_start, out_of_sample_start) VALIDATION
[out_of_sample_start, end)            OUT_OF_SAMPLE
```

Half-open intervals guarantee that a trade at a boundary belongs to exactly one
sample. Entry timestamps must be timezone-aware and sorted. Boundaries are
interpreted as midnight in `America/New_York` by default. Observations outside
the configured research interval are excluded rather than silently assigned.

## Rolling evaluation

`WalkForwardRunner` defaults to the requested 12-month training and 3-month test
schedule, rolling forward 3 months at a time. Expanding training windows are
also supported. Test-step length must equal test-window length so reported test
samples neither overlap nor leave gaps.

Each window uses:

```text
[train_start, train_end) training
[train_end, test_end)    test
```

Only complete final test windows are emitted. The data coverage end is inferred
as the beginning of the month after the final observation; callers should pass
a dataset whose start and end represent the intended complete research range.

The runner accepts two callbacks:

```python
parameters = fit(training_only)
result = evaluate(test_only, parameters)
```

Both callbacks receive defensive frame copies. Fitted parameters are copied
before evaluation and stored as an immutable mapping in `WalkForwardResult`,
along with window boundaries and row counts. This makes the sample flow
auditable and prevents an evaluator from changing the recorded training choice.

## Bias controls and limitations

- Training ends exactly where testing begins; no future test row is passed to
  `fit`.
- Input order is validated rather than silently sorted.
- Fixed validation and out-of-sample labels remain explicit.
- Callbacks are responsible for fitting transforms only on their supplied
  training frame. External/global data access cannot be prevented by this API.
- Phase 11 does not optimize parameters, combine test periods during selection,
  or promote the best out-of-sample result.
