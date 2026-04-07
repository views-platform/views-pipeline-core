# Product Development Plan: Reporting Extraction (Option 3)

**Date:** 2026-03-17
**Status:** DEFERRED — safe to do, but doesn't unblock any planned work
**Prerequisite doc:** `reports/rd_roadmap_reporting_extraction.md`

---

## Task List

### Task 1: Create `ReportingOrchestrator` class

**Effort:** Medium
**Risk:** Low

**What:** Move these methods verbatim into a new class:
- `_execute_forecast_reporting()` (model.py:2454-2581) → `generate_forecast_report()`
- `_execute_evaluation_reporting()` (model.py:3299-3369) → `generate_evaluation_report()`

```python
# New: views_pipeline_core/managers/prediction/reporting.py

class ReportingOrchestrator:
    def __init__(self, wandb_module: WandBModule):
        self._wandb = wandb_module

    def generate_forecast_report(self, configs, model_path, args):
        # Current _execute_forecast_reporting() logic, unchanged
        ...

    def generate_evaluation_report(self, configs, model_path, args):
        # Current _execute_evaluation_reporting() logic, unchanged
        ...
```

**Files to create:**
- `views_pipeline_core/managers/prediction/reporting.py`

---

### Task 2: Wire into ForecastingModelManager

**Effort:** Small
**Risk:** Low

**What:**
- Create `self._reporter = ReportingOrchestrator(self._wandb_module)` in `__init__()`
- Replace 2 call sites in `_execute_model_tasks()`:
  ```python
  # Was: self._execute_forecast_reporting()
  # Now: self._reporter.generate_forecast_report(self.configs, self._model_path, self.args)
  ```
- Remove 2 method definitions (~199 lines) from model.py

**Files to modify:**
- `views_pipeline_core/managers/model/model.py`

---

### Task 3: Update tests

**Effort:** Small
**Risk:** Low

**What:** Update any tests that mock `_execute_forecast_reporting` or `_execute_evaluation_reporting`.

---

## Dependency Graph

```
Task 1 (Create ReportingOrchestrator) — independent of Options 1 & 2
Task 2 (Wire into ForecastingModelManager) — depends on Task 1
Task 3 (Update tests) — depends on Task 2
```

---

## Why Deferred

1. **No project unblocked.** Neither forecast shipping nor ensemble refactoring touches reporting.
2. **Low duplication pressure.** Each method is called once — no DRY violation.
3. **Low cognitive load.** Reporting methods are self-contained; they don't make model.py harder to understand in the way that I/O and evaluation mixing does.
4. **Better timing:** Execute when reporting needs to change (new report type, new visualization framework, or FAO-specific reports).

---

## Verification (when executed)

```bash
conda run -n views_pipeline pytest tests/ --tb=short -q
wc -l views_pipeline_core/managers/model/model.py  # Target: ~200 fewer lines
```
