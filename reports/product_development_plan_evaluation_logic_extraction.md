# Product Development Plan: Evaluation Logic Extraction (Option 2)

**Date:** 2026-03-17
**Prerequisite:** Option 1 (Prediction I/O extraction) must be completed first.
**Prerequisite doc:** `reports/rd_roadmap_evaluation_logic_extraction.md`

---

## Task List

### Task 1: Create `EvaluationContext` dataclass

**Effort:** Small
**Risk:** Low

**What:** Bundle the 6 context values that `_evaluate_prediction_dataframe()` needs into a clean data object, avoiding a 6-parameter constructor.

```python
# New: views_pipeline_core/managers/prediction/evaluation_context.py

@dataclass(frozen=True)
class EvaluationContext:
    configs: dict
    partition_dict: dict
    run_type: str
    model_path: ModelPathManager
    steps: list
    month_last: Optional[int]  # from _data_loader.month_last (forecasting only)
```

**Files to create:**
- `views_pipeline_core/managers/prediction/evaluation_context.py`

---

### Task 2: Create `EvaluationOrchestrator` class

**Effort:** Large
**Risk:** Medium

**What:** Move these methods verbatim into the new class:
- `_evaluate_prediction_dataframe()` (model.py:2864-3057) → `evaluate_predictions()`
- `_get_evaluation_step_mappings()` (model.py:3059-3114) → `get_step_mappings()`
- `_assert_predictions_in_step_window()` (model.py:3161-3258) → `assert_predictions_in_step_window()`

**Constructor:**
```python
class EvaluationOrchestrator:
    def __init__(self, io_manager: PredictionIOManager, wandb_module: WandBModule):
        self._io = io_manager
        self._wandb = wandb_module

    def evaluate_predictions(self, predictions, eval_type, actuals_df,
                             context: EvaluationContext, eval_mode: str):
        # Current _evaluate_prediction_dataframe() logic
        # Uses context instead of self.configs, self._partition_dict, etc.
        # Calls self._io.save_evaluations() instead of self._save_evaluations()
        ...
```

**Critical design:** `prepare_actuals_df()` is called by the ForecastingModelManager BEFORE passing actuals to the evaluator. The evaluator receives pre-prepared actuals — it never calls back into the manager.

**Files to create:**
- `views_pipeline_core/managers/prediction/evaluation_orchestrator.py`

**Files to modify:**
- `views_pipeline_core/managers/model/model.py`:
  - Create `self._evaluator = EvaluationOrchestrator(self._io, self._wandb_module)` in `__init__()`
  - 3 call sites updated to delegate to `self._evaluator`
  - Remove 3 method definitions (~350 lines)

---

### Task 3: Update callers and tests

**Effort:** Medium
**Risk:** Medium

**Call sites to update:**
1. `_execute_model_evaluation()` line 2226: `self._evaluator.evaluate_predictions(...)`
2. `_execute_model_evaluation()` line 2232: same
3. `_execute_model_sweeping()` line 2448: same
4. `_execute_model_evaluation()` line 2163: `self._evaluator.assert_predictions_in_step_window(...)`

**EnsembleManager impact:**
- EnsembleManager calls `_evaluate_prediction_dataframe()` via inheritance
- Keep thin delegation method on ForecastingModelManager:
  ```python
  def _evaluate_prediction_dataframe(self, predictions, eval_type):
      actuals = self._load_and_prepare_actuals()
      ctx = self._build_evaluation_context()
      self._evaluator.evaluate_predictions(predictions, eval_type, actuals, ctx, ...)
  ```

**Test files to update:**
- Tests mocking `_evaluate_prediction_dataframe` → mock `self._evaluator.evaluate_predictions`

---

## Dependency Graph

```
Option 1 (PredictionIOManager) ← MUST be done first
  └── Task 1 (EvaluationContext) — independent
      └── Task 2 (EvaluationOrchestrator) — depends on Task 1
          └── Task 3 (Wire + update tests) — depends on Task 2
```

---

## Verification

```bash
conda run -n views_pipeline pytest tests/ --tb=short -q
conda run -n views_pipeline ruff check views_pipeline_core/managers/prediction/
wc -l views_pipeline_core/managers/model/model.py  # Target: ~2850 (from ~3200 after Option 1)
```

---

## Scope Boundaries

**DO:** Move methods verbatim. Pass context explicitly. Keep delegation methods for backward compat.
**DO NOT:** Refactor evaluation logic. Change metric computation. Modify EvaluationAdapter/EvaluationManager interfaces. Touch `prepare_actuals_df()`.
