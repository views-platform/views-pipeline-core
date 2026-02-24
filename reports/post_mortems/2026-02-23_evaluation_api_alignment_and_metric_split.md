# Post-Mortem Report: Evaluation API Alignment and Point/Uncertainty Metric Split

**Date**: 2026-02-23
**Branch**: `debug/fix-evaluation-and-alerts`
**Compared to**: `main` (merged from `feature/class_and_reg_eval`)

---

## 1. What Was Done

This branch resolved a cluster of post-refactor failures and extended the evaluation
architecture introduced in `feature/class_and_reg_eval`. The work broke into four
distinct efforts.

### 1.1 Stale Test Resolution

Six tests in `tests/test_modules/test_model_check.py` were left broken by the
previous refactor. Error-message match strings and empty-list normalization
expectations were updated to match the new `validate_config` behaviour.

### 1.2 `prepare_actuals_df` Hook (ADR-029)

A Template Method hook — `prepare_actuals_df(df: pd.DataFrame) → pd.DataFrame` —
was added to `ModelManager`. It is a no-op by default and is called inside
`_evaluate_prediction_dataframe` immediately after loading the raw viewser data.
Models that need to manufacture derived or transformed targets (e.g. HydraNet's
best-model-select columns) can override this in their own manager class without
touching pipeline-core. ADR-029 and a README section document the contract.

### 1.3 Three-Tier Metric Key Hierarchy (Point/Uncertainty Split)

The most significant feature addition. A model producing 1 000 posterior samples
requires fundamentally different metrics (CRPS, MIS, Coverage) than one producing a
scalar point estimate (MSE, MSLE, AP). Previously the pipeline had no way to express
this distinction in config; the scalar gate was a hardcoded set of metric names in
`model.py` that raised `TypeError` for distribution predictions.

A three-tier key hierarchy was introduced for metrics only (target keys were already
explicit after the previous refactor):

```
Tier 1 (legacy):        targets / metrics
Tier 2 (transitional):  regression_metrics / classification_metrics
Tier 3 (explicit):      regression_point_metrics / regression_sample_metrics
                        classification_point_metrics / classification_sample_metrics
```

Backward compatibility is preserved via a full mapping chain:

```
metrics → regression_metrics → regression_point_metrics
```

Each mapping step emits a coloured deprecation warning banner. Tier 1 mixed with
Tier 2/3, or Tier 2 mixed with Tier 3, both raise `ValueError` with a clear
"Configuration Conflict" message.

Files changed:

- `views_pipeline_core/modules/validation/model/check.py` — new key sets, mutual
  exclusivity rules, backward-compat mapping, extended normalization, updated
  `all_metrics` sync-back.
- `views_pipeline_core/managers/configuration/configuration.py` — same mapping and
  normalization mirrored in both `get_combined_config()` and
  `get_combined_sweep_config()`.
- `views_pipeline_core/managers/model/model.py` — `has_metrics` gate covers all 7
  keys; evaluation tasks dict simplified to pure target lists; hardcoded scalar gate
  replaced by config-driven dispatch delegated entirely to `EvaluationManager`.
- `views_pipeline_core/templates/reports/evaluation.py` — `all_available_metrics`
  aggregates from all 4 Tier 3 keys + Tier 2 keys + legacy `metrics`.

Tests added (TDD — written before implementation):

- `tests/test_modules/test_model_check.py`: 3 new tests covering Tier 3 keys,
  Tier 2→Tier 3 transitional mapping, and the Tier 2+Tier 3 conflict rule.
- `tests/test_explicit_tasks.py`: scalar-gate tests updated to reflect new
  no-crash/warning behaviour; distribution+uncertainty test added.

### 1.4 EvaluationManager No-Args API Alignment

During integration testing, `EvaluationManager.__init__()` was found to have been
updated in `views-evaluation` to take **no arguments**. Metrics and task-type
dispatch are now resolved internally from the config dict passed to `evaluate()`.
The implementation was calling `EvaluationManager(metrics_to_use)` — one argument
too many — producing:

```
TypeError: EvaluationManager.__init__() takes 1 positional argument but 2 were given
```

The fix simplified the evaluation loop significantly:

- `EvaluationManager()` instantiated once before the target loop, no args.
- `evaluate(df_actual, df_predictions, target, self.configs)` passes the full config;
  EvaluationManager reads the relevant metric keys internally.
- The metrics-selection and warning+skip logic in `model.py` were removed entirely —
  they are now EvaluationManager's responsibility.
- `df_actual` was changed to be sliced from `regression_targets +
  classification_targets` (Tier 3 keys) instead of the legacy `"targets"` alias,
  which is cleaner and more self-documenting.

### 1.5 Prediction DataFrame Pre-Slicing Fix

After aligning with the no-args API, HydraNet integration testing revealed a second
error:

```
ValueError: Predictions[0] must contain exactly one column, but found 15:
['c_id', 'now', 'col', 'lr_sb_best', 'by_sb_best', 'pred_lr_sb_best', ...]
```

`EvaluationManager.validate_predictions()` requires each DataFrame in the predictions
list to contain **exactly one column**, named `pred_{target}`. The full wide
prediction DataFrame (all targets plus metadata columns) was being passed directly.

The fix:

- The existence check was tightened to only look for `f"pred_{target}"` (the bare
  `target` column name fallback was dropped — `validate_predictions` would reject it
  anyway).
- A list comprehension slices each prediction DataFrame to the single column before
  `evaluate()` is called.
- `df_actual` is similarly sliced to the current target column per iteration.

All test prediction DataFrames were updated to use `pred_{target}`-named columns,
reflecting what the real pipeline actually produces.

---

## 2. Why It Was Done

### Post-Refactor Debt

The `feature/class_and_reg_eval` branch merged without resolving all test failures it
introduced. Six stale tests blocked CI and masked real issues.

### Mathematical Correctness

Applying point metrics (MSE, MSLE) to distribution predictions is mathematically
wrong. The old scalar gate prevented crashes, but did so with a hardcoded metric-name
lookup — fragile, implicit, and impossible to extend. Config-driven dispatch is the
right answer: the config is the authority on what a model produces and how it should
be evaluated.

### API Contract Enforcement

The `views-evaluation` package updated its public API (no-args constructor, strict
single-column validation) as part of its own restructuring. These constraints encode
correct usage at the library boundary rather than relying on caller discipline.
Pipeline-core had to align, or it would fail at every evaluation call.

### HydraNet Multi-Target Reality

HydraNet produces a single wide DataFrame with predictions for all six targets
simultaneously (`pred_lr_sb_best`, `pred_by_sb_best`, …). Passing this full DataFrame
to a per-target `evaluate()` call was never going to work once `validate_predictions`
was enforced. Per-target slicing is the only correct pattern.

---

## 3. How It Was Done

The work followed a strict test-driven sequence throughout:

1. **Failing tests first.** For every behaviour change, tests were written to assert
   the new expected behaviour before the implementation was touched.
2. **Read the upstream source.** Before assuming the EvaluationManager API, the
   installed package source was inspected directly (`__init__`, `evaluate`,
   `validate_predictions`). This caught both the no-args constructor and the
   single-column constraint without having to run HydraNet end-to-end first.
3. **One concern at a time.** Each commit addresses a single named fix. The branch
   history is a readable sequence of decisions, not a batch of changes.
4. **No backward-compat hacks.** Every Tier 3 key is a first-class citizen with
   validation, normalization, and sync-back. Legacy and transitional keys are mapped
   forward with warnings, not silently swallowed.

### A Bug Found During Implementation: Stale Snapshot of a Mutable Dict

The Tier 2→Tier 3 mapping in `check.py` initially used `present_transitional_metrics`
— a variable computed at the top of the function, **before** the Tier 1→Tier 2
mapping ran. For legacy configs, `regression_metrics` is added to the config dict
*during* the Tier 1→Tier 2 mapping, so `present_transitional_metrics` (computed
before that point) was always an empty set for legacy configs, silently skipping the
second-level mapping.

The fix was to re-inspect `config.keys()` live at the point of the Tier 2→Tier 3
mapping rather than using the pre-computed set:

```python
# Correct: inspect config.keys() *after* Tier 1→Tier 2 mapping has run
if (transitional_metric_keys & config.keys()) and not (explicit_metric_keys & config.keys()):
```

This class of bug — a stale variable snapshot of a dict that is later mutated in the
same function — is easy to introduce and hard to catch without a test that exercises
the full legacy→Tier 3 chain end-to-end.

---

## 4. What Was Learned

### Explicit API Contracts at Library Boundaries Are Non-Negotiable

`EvaluationManager.validate_predictions` raising `ValueError` for multi-column input
is not pedantic — it is the library doing what a library should do: enforce its
contract loudly rather than silently producing wrong results. Pipeline-core must
pre-process its data to match. The lesson: whenever consuming an external library's
method, read its validation logic before writing the calling code.

### Config Is the Right Authority for Prediction Type

The hardcoded `point_metrics` set in the old scalar gate was the symptom of a deeper
problem: the pipeline was making decisions that belong in the config. The three-tier
hierarchy resolves this permanently. Any future metric type (sharpness, calibration,
energy score) gets a new key; `model.py` touches nothing.

### Stale Snapshots in Mutable Dicts Are a Silent Footgun

The `present_transitional_metrics = transitional_metric_keys & config.keys()` pattern
computed at function entry, followed by mutation of `config`, left the variable stale
for the legacy path. The rule going forward: evaluate `key_set & config.keys()` at
the point of use, not once at the top of a function that modifies `config`.

### The `pred_` Prefix Is a Pipeline API Contract, Not a Convention

`EvaluationManager.validate_predictions` validates that prediction columns are named
`pred_{target}`. This is now an explicit, documented contract. Tests must use
`pred_`-prefixed column names in prediction DataFrames. Any model that does not
produce `pred_{target}` columns must rename them upstream (in `prepare_actuals_df` or
equivalent) before the evaluation call.

### Template Method Hooks Are Cheap Insurance

The `prepare_actuals_df` hook costs almost nothing to add (a one-line no-op default)
but gives model-specific managers a clean escape valve for bespoke pre-processing.
The alternative — adding special-case parameters to `_evaluate_prediction_dataframe`
— would accumulate as a maintenance burden. Hooks should be added proactively
whenever a base-class method operates on data that a subclass might reasonably need
to transform.

---

## 5. Impact Assessment

| Area | Before | After |
|---|---|---|
| Metric configuration | Hardcoded scalar gate; no uncertainty support | Three-tier hierarchy; any point/uncertainty combination expressible in config |
| EvaluationManager compatibility | Broken (`TypeError` on construction) | Fully aligned: no-args constructor, per-target single-column slices |
| HydraNet multi-target evaluation | `ValueError` (15 columns passed) | Correctly sliced to single `pred_{target}` column per target |
| Actuals derivation | No hook; bespoke logic had to live in pipeline-core | `prepare_actuals_df` hook; model-specific logic stays in the model |
| Test suite | 6 stale failures + EvaluationManager mock mismatches | 682 passed, 1 skipped |
| Backward compatibility | Unchanged — all legacy keys worked | Unchanged — all legacy keys still work; deprecation warnings guide migration |

The branch is safe to merge. All legacy model configs (single `targets`/`metrics`
keys) continue to work without modification; they will see deprecation warning banners
that clearly explain the migration path to Tier 3 keys.
