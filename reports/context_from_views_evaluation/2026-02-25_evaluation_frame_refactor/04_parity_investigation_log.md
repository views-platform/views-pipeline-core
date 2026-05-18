# Parity Investigation Log

**Date:** 2026-02-25
**Campaign:** Green Team (Happy Path)

## Finding 1: Step-Wise Filtering Discrepancy
**Symptom**: Native evaluator returned 4 steps, legacy returned 2 steps.
**Root Cause**: Legacy `EvaluationManager` strictly respects `config['steps']` when initializing the result dictionary. Native implementation blindly grouped by all available `step_id` values.
**Resolution**: Update `NativeEvaluator` to filter by `config['steps']` during the step-wise schema execution.

## Finding 2: Positional Step Assignment
**Symptom**: `PandasAdapter` assumes step is `1..N` based on row position.
**Risk**: If a model returns predictions starting at step 3, the adapter will mislabel them as step 1.
**Legacy Behavior**: Confirmed `_split_dfs_by_step` is purely positional. This is a known technical debt/implicit contract in the legacy system that we must reproduce for parity, but flag for improvement.

## Finding 3: Step-Wise Truncation via `zip`
**Symptom**: Legacy returns `NaN` for steps where any one sequence is missing. Native returns valid metrics.
**Root Cause**: `EvaluationManager._split_dfs_by_step` uses `zip(*all_month_ids)`. If sequences have different numbers of unique months, `zip` truncates to the shortest sequence. 
**Impact**: High. A single incomplete forecast sequence suppresses step-wise metrics for all other sequences at that lead time.
**Parity Strategy**: To achieve parity, we must decide if we want to reproduce this bug or fix it and document the divergence. Given the "Total Parity" mandate, we should probably reproduce it in a "LegacyCompatibilityMode" but default to the "Fixed" behavior.

## Finding 4: Legacy Month-Wise Fragility (Float/NaN)
**Symptom**: `KeyError: 'month100.0'` when index contains floats or NaNs.
**Root Cause**: Legacy code assumes `month_id` is an integer and uses `range(min, max+1)` to build keys, but uses `str(month)` (which can be a float string or 'nan') to access them.
**Impact**: Medium. Prevents evaluation of data with non-integer time indices or missing time values.
**Parity Strategy**: We will NOT reproduce this bug. `EvaluationFrame` will treat identifiers as opaque keys for grouping. If the legacy code crashes, we will document that "Parity is impossible for this adversarial case due to legacy bugs."


