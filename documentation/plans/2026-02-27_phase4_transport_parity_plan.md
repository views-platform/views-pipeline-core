# Phase 4: Unified Evaluation Entrance (Transport Parity)

> **Archived 2026-02-27.** This plan was superseded by the emergency base_origin
> fix (step_mapping ↔ prediction temporal mismatch). See the implementation notes
> in `feature/samples_for_fao`.

## Context

Phase 1–3 are complete. The "Transition Hole" is that `_evaluate_prediction_dataframe`
still only accepts `pd.DataFrame` as input and does Pandas column surgery inside the
Manager. If HydraNet returns a `PredictionFrame` directly, the orchestrator crashes
before reaching the parity audit because it tries `.columns` on a PF object.

**Goal**: Make the evaluation entrance framework-agnostic so that both `pd.DataFrame`
(legacy) and `PredictionFrame` (native) arrive cleanly at the adapter without Pandas
surgery in the Manager.

---

## Current State (The Hole)

**`_evaluate_prediction_dataframe`** — `model.py:2659`
- Signature: `(self, df_predictions, eval_type, ensemble=False)` — no type hint
- Line 2735: `first_df.columns` check → crashes if `df_predictions` is `PredictionFrame`
- Line 2747–2748: `pred_slices = [df[[f"pred_{target}"]] for df in raw_preds]` — Pandas surgery in Manager
- Line 2755: only calls `PandasAdapter.from_dataframes()`, never `from_prediction_frame()`

**`_evaluate_model_artifact`** — `model.py:1497`
- Return type annotated as `Union[Dict, pd.DataFrame]` (incorrect even for current usage)
- Docstring contract says "return list of prediction DataFrames" — no mention of PF

**`PandasAdapter.from_prediction_frame`** — `adapter.py:199`
- Already implemented and tested
- Takes `(actual, prediction_frame, target, step_mapping)` — no column slicing needed

---

## Critical Files

| File | Role |
|------|------|
| `views_pipeline_core/managers/model/model.py` | Main changes (lines ~1497, ~2659–2760) |
| `views_pipeline_core/modules/validation/adapter.py` | Already complete — reference only |
| `views_pipeline_core/data/prediction_frame.py` | Already complete — reference only |
| `tests/test_explicit_tasks.py` | Add PF path test |
| `tests/test_modules/test_evaluation_adapter.py` | Already covers adapter — reference |

---

## Task 4.1 — Unified Entrance

**File**: `model.py` (~line 2659)

Update `_evaluate_prediction_dataframe` to detect input type and route accordingly.

**Signature update** (add type hint):
```python
def _evaluate_prediction_dataframe(
    self,
    df_predictions: Union[pd.DataFrame, List[pd.DataFrame], "PredictionFrame"],
    eval_type: str,
    ensemble: bool = False,
) -> None:
```

**Per-target dispatch block** — replace the current column-check + pred_slices block:
```python
from views_pipeline_core.data.prediction_frame import PredictionFrame

# ...inside the for target loop...

if isinstance(df_predictions, PredictionFrame):
    # Native path: PredictionFrame bypasses Pandas entirely.
    step_mapping_single = self._get_evaluation_step_mappings(n_sequences=1)[0]
    ef = PandasAdapter.from_prediction_frame(
        actual=actual_slice,
        prediction_frame=df_predictions,
        target=target,
        step_mapping=step_mapping_single,
    )
    eval_result_dict = evaluation_manager.evaluate(
        actual_slice, [], target, self.configs, ef=ef
    )
else:
    # Legacy path: DataFrames — dual-track parity audit preserved.
    if f"pred_{target}" not in (
        df_predictions[0] if isinstance(df_predictions, list) else df_predictions
    ).columns:
        logger.warning(f"Column pred_{target} not found. Skipping.")
        continue

    raw_preds = df_predictions if isinstance(df_predictions, list) else [df_predictions]
    pred_slices = [df[[f"pred_{target}"]] for df in raw_preds]
    step_mappings = self._get_evaluation_step_mappings(n_sequences=len(pred_slices))
    ef = PandasAdapter.from_dataframes(
        actual=actual_slice,
        predictions=pred_slices,
        target=target,
        step_mapping=step_mappings,
    )
    legacy_result = evaluation_manager.evaluate(
        actual_slice, pred_slices, target, self.configs, ef=None
    )
    shadow_result = evaluation_manager.evaluate(
        actual_slice, pred_slices, target, self.configs, ef=ef
    )
    self._audit_parity(legacy_result, shadow_result, target)
    eval_result_dict = shadow_result
```

---

## Task 4.2 — Remove Column Surgery from Manager

The Manager currently pre-slices DataFrames before passing them to the adapter.
`from_dataframes()` already does `pred_col = f"pred_{target}"` internally.

**Change**: Pass `raw_preds` (full DataFrames) directly:

```python
# Before
pred_slices = [df[[f"pred_{target}"]] for df in raw_preds]
ef = PandasAdapter.from_dataframes(
    actual=actual_slice, predictions=pred_slices, target=target, ...
)

# After
ef = PandasAdapter.from_dataframes(
    actual=actual_slice, predictions=raw_preds, target=target, ...
)
```

---

## Task 4.2b — Update Abstract Contract

Update return type annotation and docstring on `_evaluate_model_artifact`:

```python
@abstractmethod
def _evaluate_model_artifact(
    self, eval_type: str, artifact_name: str
) -> Union[List[pd.DataFrame], "PredictionFrame"]:
    """
    Returns:
        List[pd.DataFrame]: One DataFrame per evaluation sequence (rolling-origin legacy).
        PredictionFrame: Direct NumPy output for tensor-based models (native path).
    """
```

---

## Task 4.3 — Tests

**New test in `tests/test_explicit_tasks.py`** — proves PF path routes correctly:
```python
def test_prediction_frame_path_no_crash(mock_read, ...):
    """PredictionFrame input routes to from_prediction_frame(), no Pandas surgery crash."""
    ...
    manager._evaluate_prediction_dataframe(pf, eval_type="standard")
```

---

## What Does NOT Change

- `PredictionFrame` class — already correct
- `PandasAdapter.from_prediction_frame()` — already correct
- `PandasAdapter.from_dataframes()` — already correct
- Legacy DataFrame path, parity audit, `_audit_parity()` — unchanged
- `_evaluate_model_artifact()` in subclasses — unchanged (still returns DataFrames today)

---

## Verification

```bash
conda run -n views_pipeline pytest tests/ --tb=short -q
conda run -n views_pipeline pytest tests/test_explicit_tasks.py tests/test_audit_security_robustness.py -v -k "prediction_frame"
conda run -n views_pipeline ruff check views_pipeline_core/managers/model/model.py
```
