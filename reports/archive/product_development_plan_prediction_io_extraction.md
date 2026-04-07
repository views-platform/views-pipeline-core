# Product Development Plan: Prediction I/O Extraction (Option 1)

**Date:** 2026-03-17
**Prerequisite:** Read `reports/rd_roadmap_prediction_io_extraction.md`

---

## Task List

### Task 1: Create `PredictionIOManager` class

**Effort:** Medium
**Risk:** Low

**What:**
- Create `views_pipeline_core/managers/prediction/io.py`
- Move these methods verbatim (no logic changes):
  - `_save_predictions()` (model.py:2766-2862) → `save_predictions()`
  - `_save_evaluations()` (model.py:2666-2764) → `save_evaluations()`
  - `_save_model_artifact()` (model.py:2583-2626) → `save_model_artifact()`
  - `_generate_evaluation_table()` (model.py:3260-3297) → `generate_evaluation_table()`

**Constructor:**
```python
class PredictionIOManager:
    def __init__(self, model_path, wandb_module, datastore=None,
                 pred_store_name=None, use_prediction_store=False):
        self._model_path = model_path
        self._wandb_module = wandb_module
        self._datastore = datastore
        self._pred_store_name = pred_store_name
        self._use_prediction_store = use_prediction_store
```

**Critical detail:** `_save_predictions()` currently accesses `self.configs` and `self.args` for naming. These become explicit parameters on the extracted method signature:
```python
def save_predictions(self, data, path_generated, run_type, timestamp,
                     level, targets, origin_idx=None, send_alert=True):
```

**Files to create:**
- `views_pipeline_core/managers/prediction/io.py`

**Test strategy:**
- Unit tests for each extracted method with synthetic data + tmp_path
- Verify parquet write (DataFrame + Arrow Table paths)
- Verify prediction store mock calls

---

### Task 2: Wire `PredictionIOManager` into `ForecastingModelManager`

**Effort:** Small
**Risk:** Low

**What:**
- In `ForecastingModelManager.__init__()`, create `self._io = PredictionIOManager(...)`
- Replace all 7 call sites:
  - `self._save_predictions(...)` → `self._io.save_predictions(...)`
  - `self._save_evaluations(...)` → `self._io.save_evaluations(...)`
  - `self._save_model_artifact(...)` → `self._io.save_model_artifact(...)`
  - `self._generate_evaluation_table(...)` → `self._io.generate_evaluation_table(...)`
- Remove the 4 method definitions from model.py
- Optionally: keep thin delegation methods for backward compat with EnsembleManager

**Files to modify:**
- `views_pipeline_core/managers/model/model.py`

**Test strategy:**
- All 899 tests pass
- No behavioral change — same code, different location

---

### Task 3: Update test mocks

**Effort:** Small
**Risk:** Low

**What:**
- Tests that patch `_save_predictions` on ForecastingModelManager must patch `self._io.save_predictions` instead
- Grep for all mock targets referencing the moved methods

**Files to modify:**
- `tests/test_managers/test_model_manager_prediction_format.py`
- `tests/test_managers/test_ensemble_manager.py` (if it mocks save methods)
- Any other test files that mock the moved methods

**Verification:**
```bash
conda run -n views_pipeline pytest tests/ --tb=short -q
grep -rn "_save_predictions\|_save_evaluations\|_save_model_artifact\|_generate_evaluation_table" tests/
```

---

## Dependency Graph

```
Task 1 (Create PredictionIOManager) — independent
Task 2 (Wire into ForecastingModelManager) — depends on Task 1
Task 3 (Update test mocks) — depends on Task 2
```

**Execute sequentially:** 1 → 2 → 3

---

## Verification (after all tasks)

```bash
conda run -n views_pipeline pytest tests/ --tb=short -q  # 899 pass
conda run -n views_pipeline ruff check views_pipeline_core/managers/prediction/io.py views_pipeline_core/managers/model/model.py
wc -l views_pipeline_core/managers/model/model.py  # Target: ~3200 (from 3436)
wc -l views_pipeline_core/managers/prediction/io.py  # Target: ~280 (236 + imports/class def)
```

---

## Scope Boundaries

**DO:** Move methods verbatim. Wire via constructor injection. Update mocks.
**DO NOT:** Change method behavior. Refactor the moved methods. Add new features. Touch orchestration logic.
