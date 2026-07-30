# R&D Roadmap: Reporting Extraction (Option 3)

**Date:** 2026-03-17
**Scope:** Extract `_execute_forecast_reporting` and `_execute_evaluation_reporting` from `ForecastingModelManager` into `ReportingOrchestrator`
**Lines moved:** ~199 of 2,077 (ForecastingModelManager)

---

## 1. Current State

Two reporting methods generate interactive HTML reports. Each is called from exactly one place (`_execute_model_tasks()`). They're self-contained: load data from disk, generate HTML via template classes, send WandB alerts.

| Method | Lines | Fan-in | Purpose |
|--------|-------|--------|---------|
| `_execute_forecast_reporting()` | 2454-2581 (128) | 1 call site | Maps + time-series graphs for forecasts |
| `_execute_evaluation_reporting()` | 3299-3369 (71) | 1 call site | Metrics tables + baseline comparisons |

**Dependencies consumed:** `self.configs`, `self._model_path`, `self.args`, `self._wandb_module`

**Notable complexity:** `_execute_forecast_reporting()` contains a 40-line if/elif block (lines 2506-2546) that handles ensemble vs. single model data loading differently. This is the most complex part and the main reason this method is 128 lines.

---

## 2. Target State

```
ForecastingModelManager (orchestration)
  └── self._reporter: ReportingOrchestrator (injected)
        ├── generate_forecast_report(configs, model_path, args, wandb_module)
        └── generate_evaluation_report(configs, model_path, args, wandb_module)
```

`ReportingOrchestrator` lives in `views_pipeline_core/managers/prediction/reporting.py`. It reuses existing template classes (`ForecastReportTemplate`, `EvaluationReportTemplate`).

---

## 3. Independence from Other Options

This extraction is **fully independent** of Options 1 and 2. Reporting methods don't call `_save_predictions()`, `_save_evaluations()`, or `_evaluate_prediction_dataframe()`. They can be extracted in any order.

However, **the value is lower** than Options 1 and 2 because:
- Reporting is not in the critical path of either planned project (forecast shipping or ensemble)
- The methods are only called once each — they don't create duplication pressure
- The complexity is self-contained (no cross-method coupling to untangle)

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Ensemble vs. model branching in forecast reporting | Low | Medium | Move the branching logic as-is; don't redesign |
| Template class imports become wrong | Low | Low | Use same import paths; templates are in `views_pipeline_core/templates/reports/` |
| WandB run context leaks across methods | Low | Medium | Each report method creates and finishes its own WandB run |

---

## 5. Recommendation

**Defer this extraction.** It's safe and clean, but it doesn't unblock any planned work. The 199 lines it removes from model.py are welcome but not critical.

**When to do it:** After Options 1 and 2 are stable. Or when reporting needs to change (new report type, new visualization backend).
