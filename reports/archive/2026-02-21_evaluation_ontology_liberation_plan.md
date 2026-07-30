# Architectural Manifesto: Evaluation Ontology Liberation (Revised)

**Revision Date**: 2026-02-22
**Revised by**: Simon + Claude
**Status**: Supersedes the original 2026-02-21 draft — the original plan only addressed the immediate crash; this revision addresses the full scope of EvaluationManager overreach identified after reading the complete source.

---

## 1. Executive Mission: From Gatekeeper to Pure Metrics Engine

### 1.1 The Original Framing Was Too Narrow

The original plan described the goal as making EvaluationManager a "Passenger" — a data-agnostic entity that stops crashing on unrecognised column name prefixes. That framing was correct but insufficiently ambitious. After reading the full source of `evaluation_manager.py` and `metric_calculators.py`, it is clear that the immediate `ValueError` is only the most visible symptom of a broader architectural problem.

The true mission is this: **EvaluationManager must become a pure metrics engine.** Its sole responsibility is to receive pre-prepared numbers, align them by index, and compute the metrics it was asked to compute. Nothing else.

### 1.2 The Single Responsibility Defined

**EvaluationManager IS responsible for:**
- Receiving aligned, evaluation-ready actuals and predictions
- Determining whether predictions are point estimates or distributions (inferred from data shape — arrays vs scalars — not from column names)
- Aligning actuals and predictions by temporal index
- Computing the metrics it was initialised with
- Returning structured results

**EvaluationManager is NOT responsible for:**
- Transforming data in any direction (forward or inverse)
- Scaling, normalising, or otherwise manipulating values
- Inferring what space the data is in from column name prefixes (`ln_`, `lx_`, `lr_`, `by_`, or any other)
- Deciding how to binarise continuous predictions (thresholds, cutoffs)
- Making any assumption about the semantics of values beyond their Python types

This boundary, once drawn, must never be crossed again.

---

## 2. Complete Diagnosis: All Sites of Overreach

The original plan identified one offender. A full read of the source reveals four distinct sites of overreach, of varying severity.

### 2.1 `transform_data` — Primary Offence (Critical)

**Location**: `evaluation_manager.py`, method `transform_data`, called from `_process_data`

**What it does**: Inspects column name prefixes (`ln_`, `lx_`, `lr_`) and applies domain-specific inverse mathematical transformations (exp, identity) to both actuals and predictions before metric computation.

**Why it is wrong**: This is the evaluator doing the model manager's job. The evaluator has no business knowing that `ln_` signals a natural-log transformation that needs to be inverted with `exp(x) - 1`. That knowledge is a property of the model that produced the data — specifically, it belongs to the model manager that chose the transformation. By embedding this knowledge in the evaluator, we have created a closed-world assumption: any target whose prefix is not on the evaluator's internal whitelist is rejected with a `ValueError`. This is precisely what crashes HydraNet.

**The deeper flaw**: The evaluator is currently responsible for bringing data back to "raw count space" before computing metrics. This assumes that all metrics should be computed in raw count space, which is itself a domain assumption the evaluator should not be making. A model that produces calibrated probabilities (like HydraNet's binary classification) should have its predictions evaluated in probability space — not forced through an inappropriate inverse transformation.

**Note on `lx_` formula**: The current `lx_` branch computes `exp(x) - exp(100)`. Since `exp(100) ≈ 2.7 × 10^43`, this would produce astronomically large negative numbers for any realistic input. This is almost certainly a latent bug. It is not the focus of this plan but should be investigated separately once the transformation logic is moved to where it belongs (the model manager).

### 2.2 `calculate_ap` Hardcoded Threshold — Secondary Offence (Significant)

**Location**: `metric_calculators.py`, `calculate_ap`, `threshold=25` default argument

**What it does**: Converts continuous predictions to binary using a hardcoded threshold of 25, then computes Average Precision. The value 25 is calibrated for raw fatality counts — "25 deaths or more constitutes a conflict event."

**Why it is wrong**: A threshold of 25 is not a property of Average Precision as a metric — it is a domain-specific modelling decision about what constitutes a positive class in the context of raw fatality count data. Baking it into the metric function means:

1. For HydraNet's `by_sb_best` (already a binary 0/1 signal): a threshold of 25 classifies every single prediction as 0 (since all values are ≤ 1), making AP undefined or misleading.
2. For any future model operating in a different space (log counts, normalised, calibrated probabilities): the threshold is simply wrong.
3. The metric function now encodes a domain assumption that will silently produce incorrect results for any model that doesn't happen to operate in raw fatality count space.

**The correct approach**: Thresholds that convert continuous values to binary are a property of the **evaluation configuration** (defined by the model team), not of the metric function itself. The metric function should receive pre-binarised actuals and predictions, or the threshold should be passed explicitly through the config and applied upstream, before the evaluator sees the data. The evaluator should not be in the business of deciding what counts as a "positive" event.

### 2.3 `convert_to_array` Structural Coercion — Tertiary Offence (Moderate)

**Location**: `evaluation_manager.py`, method `convert_to_array`, called from `_process_data`

**What it does**: Wraps every cell value in a numpy array. Scalars become single-element arrays `np.array([x])`, lists become `np.array(x)`, existing ndarrays pass through.

**Why it is a concern**: This is a structural manipulation of the data — the evaluator is deciding what form the numbers should be in before metric computation. The metric functions then assume this array-per-cell structure (they use `np.concatenate(matched_actual[target].values)` etc.). This creates a tight coupling between the input format and the internal metric computation format.

**However — this is the least urgent issue**. The array-per-cell structure is the evaluator's internal representation and is not exposed externally. The concern is more about clarity of contract: callers should know exactly what format is expected. The current implicit coercion hides this. The right fix here is documentation and, eventually, moving to explicit format validation rather than silent coercion.

### 2.4 The `pred_` Naming Convention — Structural Contract (Mild, Keep with Documentation)

**Location**: `validate_predictions`, `_match_actual_pred`, and **every single function in `metric_calculators.py`** — all hardcode `f"pred_{target}"` to locate the prediction column.

**What it does**: Establishes a naming convention: actuals live in column `{target}`, predictions live in column `pred_{target}`.

**Is this overreach?** This is a genuinely difficult question. The user is right to be on the fence. There is a meaningful distinction between:

- **Semantic inference** from column names (e.g. "this column starts with `ln_`, therefore it is log-transformed") — this is wrong, it makes the evaluator a domain expert
- **Structural identification** via naming convention (e.g. "predictions are in the column prefixed with `pred_`") — this is a contract, not domain inference

The `pred_` convention is structural identification. It is more akin to a function parameter naming convention than to domain-specific knowledge. **The recommendation is to keep it, but to make it explicit and documented as the API contract rather than silent magic.** The metric functions should clearly state in their docstrings that `pred_{target}` is the expected prediction column name. If in future the codebase migrates to passing explicit Series/arrays instead of named DataFrame columns, that is a reasonable refactor — but it would require changing every metric function and every caller simultaneously. The cost exceeds the benefit at this stage.

**The line we draw**: The naming convention is acceptable. Semantic inference from content of names is not.

---

## 3. The Responsibility Boundary: A Formal Statement

To prevent future violations, the boundary must be stated formally and enforced through code review.

```
┌──────────────────────────────────────────────────────────────────┐
│                    EVALUATION MANAGER BOUNDARY                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INSIDE (EvaluationManager's responsibility):                    │
│  ✓ Index alignment of actuals and predictions                    │
│  ✓ Point vs. uncertainty detection (from array shape, not names) │
│  ✓ Step-wise / time-series / month-wise aggregation structure    │
│  ✓ Dispatching to metric functions                               │
│  ✓ Returning structured result dictionaries and DataFrames       │
│                                                                  │
│  OUTSIDE (caller's responsibility, never EvaluationManager's):   │
│  ✗ Inverse transformations (exp, log, scale inversion)           │
│  ✗ Forward transformations of any kind                           │
│  ✗ Binarisation / thresholding                                   │
│  ✗ Knowing what a column name prefix means semantically          │
│  ✗ Deciding what "evaluation space" a target should be in        │
│  ✗ Converting prediction formats (that is the model manager's job)│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

The entity responsible for ensuring data is in evaluation-ready form when it arrives at EvaluationManager is **the model manager in the model repo**, assisted by the hooks in `views-pipeline-core` (`prepare_actuals_df` for actuals).

---

## 4. Implementation Plan

The implementation is split into two phases. Phase 1 is tactical and immediate — it unblocks HydraNet without touching architecture. Phase 2 is the structural correction that removes the overreach permanently.

### 4.1 Phase 1: Tactical Unblock (Immediate)

**Target file**: `views-evaluation/views_evaluation/evaluation/evaluation_manager.py`
**Method**: `transform_data`
**Change**: Replace `else: raise ValueError` with identity pass-through and a structured warning.

**Before**:
```python
else:
    raise ValueError(f"Target {t} is not a valid target")
```

**After**:
```python
else:
    logger.warning(
        f"transform_data: unrecognised prefix for target '{t}'. "
        "Applying identity (no transformation). "
        "If this target requires inverse transformation, it must be "
        "applied by the model manager before calling evaluate(). "
        "This fallback will be removed when transform_data is deprecated."
    )
    df[[t]] = df[[t]].applymap(lambda x: x)
```

**Why a warning, not silence**: Silent pass-through would mask typos. A developer who misnames `ln_ged_sb` as `1n_ged_sb` would get silently wrong metrics (computed on log-scale values instead of raw counts) with no indication anything went wrong. The warning surfaces this immediately.

**Risk**: Zero. The `ln_` and `lx_` and `lr_` branches are completely unchanged. Only targets with unknown prefixes are affected, and they were crashing before. A warning is strictly better than a crash.

**This phase buys time** for Phase 2 without creating permanent technical debt, because the warning itself explicitly states it is a temporary fallback.

### 4.2 Phase 2: Structural Correction (Planned)

Phase 2 has three parallel tracks. They should be implemented together or in close sequence, not piecemeal.

#### Track A: Remove `transform_data` from `_process_data`

**Target file**: `views-evaluation/views_evaluation/evaluation/evaluation_manager.py`

The `_process_data` method currently applies `convert_to_array` and then `transform_data` to both actuals and predictions. The `transform_data` call must be removed. The `convert_to_array` call should remain for now (it is internal structural normalisation), but should be documented explicitly as the input format contract.

`transform_data` should be **deprecated** (not deleted immediately) — marked with a deprecation warning if called directly — so that any external callers who depend on it are informed. It can be deleted once no callers remain.

The `evaluate()` method signature does not need to change. The data simply arrives pre-transformed.

#### Track B: Add `prepare_predictions_for_evaluation` Hook to `views-pipeline-core`

**Target file**: `views_pipeline_core/managers/model/model.py`
**Class**: `ForecastingModelManager`

We already have `prepare_actuals_df` for actuals. We need the symmetric hook for predictions. Before the predictions list is passed into `evaluation_manager.evaluate()`, the model manager should have the opportunity to transform them into evaluation-ready form.

This hook mirrors the exact pattern of `prepare_actuals_df`:

```python
def prepare_predictions_for_evaluation(
    self, predictions: list[pd.DataFrame]
) -> list[pd.DataFrame]:
    """
    Hook for model-specific preparation of prediction DataFrames
    before evaluation metrics are computed.

    By default this is a no-op. Subclasses that produce transformed
    predictions (e.g. log-scale outputs that need inverting before
    computing metrics on raw counts) must override this method.

    Args:
        predictions: List of prediction DataFrames as produced by
            _evaluate_model_artifact. May contain transformed values.

    Returns:
        List of DataFrames with values in evaluation-ready form.
    """
    return predictions
```

This hook is called in `_evaluate_prediction_dataframe` immediately before `evaluation_manager.evaluate()` is called, just as `prepare_actuals_df` is called before slicing actuals.

#### Track C: Migrate Legacy Models

For legacy models that currently rely on `transform_data` to invert `ln_` transformations:

The inverse transformation must move into those models' `prepare_predictions_for_evaluation` overrides. For example, a legacy model producing `ln_ged_sb` predictions would implement:

```python
def prepare_predictions_for_evaluation(self, predictions):
    for df in predictions:
        if "pred_ln_ged_sb" in df.columns:
            df["pred_ln_ged_sb"] = np.exp(df["pred_ln_ged_sb"]) - 1
    return predictions
```

This is exactly where this logic belongs — in the model repo, beside the forward transformation that was applied at training time.

**Note on `calculate_ap` threshold**: Once Track B is in place, the threshold binarisation that currently lives in `calculate_ap` should be moved upstream. The model config should specify a threshold per target, and the `prepare_actuals_df` / `prepare_predictions_for_evaluation` hooks should apply it before the evaluator sees the data. For already-binary targets (like `by_sb_best`), no thresholding is applied — the data is already in the right form. This makes the threshold an explicit modelling decision rather than an implicit metric function default.

---

## 5. What EvaluationManager Will Look Like After Phase 2

`_process_data` will simplify to:

```python
def _process_data(self, actual, predictions, target):
    actual = EvaluationManager.convert_to_array(actual, target)
    predictions = [
        EvaluationManager.convert_to_array(pred, f"pred_{target}")
        for pred in predictions
    ]
    return actual, predictions
```

No transformations. No prefix inspection. No domain knowledge. Pure structural normalisation into the array-per-cell format that the metric functions expect.

`transform_data` will carry a deprecation warning and eventually be removed entirely in a future minor version.

---

## 6. Risk Matrix

| Risk | Severity | Mitigation |
|---|---|---|
| Legacy models produce wrong metrics after Phase 2 (they relied on `transform_data` to invert `ln_`) | High | Legacy models must implement `prepare_predictions_for_evaluation`. Tracked via deprecation warning on `transform_data`. Full regression test suite run after each model migrates. |
| Developer forgets to override hook and gets silently wrong metrics | Medium | The Phase 1 warning is the safety net during transition. Post-Phase-2, wrong metrics will be obviously wrong (log-scale numbers vs raw counts) rather than silently wrong. |
| `calculate_ap` threshold issue causes wrong AP scores for binary targets immediately | Medium | HydraNet's `by_sb_best` is already 0/1, so `threshold=25` produces all-zero predictions — AP will be 0 or undefined. This must be addressed in Track C alongside the threshold migration. |
| `transform_data` removed too soon before all models migrated | Low | Keep `transform_data` in the class (deprecated) until all callers have migrated. Delete only when `grep transform_data` returns no external callers. |

---

## 7. Success Definition

### Phase 1 Success
- HydraNet evaluation runs without crashing.
- A warning is logged for each unrecognised prefix.
- All existing models continue to produce identical metric values (the `ln_`, `lx_`, `lr_` branches are unchanged).

### Phase 2 Success
- `transform_data` is not called from `_process_data`.
- `EvaluationManager.evaluate()` receives pre-prepared data from all callers.
- No transformation logic of any kind exists inside `EvaluationManager` that is called during a normal evaluation run.
- The `pred_` naming convention is explicitly documented as the API contract.
- Metric values for all models are numerically identical to pre-Phase-2 values (verified by regression tests).
- The `calculate_ap` threshold decision has been moved to the model config and applied upstream.

### The Definition of Done (Permanent)
**EvaluationManager calculates metrics on the numbers it is given. It does not transform, scale, threshold, or infer anything from column name content. The model manager is the sole authority on what form data takes when it enters the evaluation pipeline.**
