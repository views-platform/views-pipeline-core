# R&D Roadmap: Evaluation Logic Extraction (Option 2)

**Date:** 2026-03-17
**Scope:** Extract `_evaluate_prediction_dataframe`, `_get_evaluation_step_mappings`, `_assert_predictions_in_step_window` from `ForecastingModelManager` into `EvaluationOrchestrator`
**Lines moved:** ~350 of 2,077 (ForecastingModelManager)

---

## 1. Current State

Three evaluation-related methods embed metric computation, step mapping, and validation logic inside the orchestration class. Unlike the I/O methods (Option 1), these have **moderate coupling** to the manager's state.

| Method | Lines | Fan-in | Dependencies |
|--------|-------|--------|-------------|
| `_evaluate_prediction_dataframe()` | 2864-3057 (194) | 3 call sites | configs, _partition_dict, _data_loader, _model_path, _wandb_module, prepare_actuals_df(), _io.save_evaluations(), _io.generate_evaluation_table() |
| `_get_evaluation_step_mappings()` | 3059-3114 (56) | 3 call sites | args.run_type, _data_loader.month_last, _partition_dict, configs["steps"] |
| `_assert_predictions_in_step_window()` | 3161-3258 (98) | 1 call site | _get_evaluation_step_mappings(), _partition_dict |

**Why this is harder than Option 1:** `_evaluate_prediction_dataframe()` accesses 6 manager attributes and calls `prepare_actuals_df()` — a method that subclasses may override (hydranet does). This creates a coupling that requires careful design.

---

## 2. Target State

```
ForecastingModelManager (orchestration)
  └── self._evaluator: EvaluationOrchestrator (injected)
        ├── evaluate_predictions(predictions, eval_type, actuals_df, configs, ...)
        ├── get_step_mappings(n_sequences, run_type, partition_dict, ...)
        └── assert_predictions_in_step_window(predictions, ...)
```

**Key design decision:** `prepare_actuals_df()` stays on ForecastingModelManager (it's an override hook for subclasses). The orchestrator receives the prepared actuals as a parameter.

---

## 3. Dependency on Option 1

This extraction **depends on Option 1 being completed first.** `_evaluate_prediction_dataframe()` calls `_save_evaluations()` and `_generate_evaluation_table()`. If those are already extracted into `PredictionIOManager`, the evaluator can receive the I/O manager as a dependency:

```python
class EvaluationOrchestrator:
    def __init__(self, io_manager: PredictionIOManager, wandb_module: WandBModule):
        self._io = io_manager
        self._wandb = wandb_module
```

Without Option 1 first, this extraction creates a circular dependency.

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `prepare_actuals_df()` override breaks | Medium | High | Pass prepared actuals as parameter; don't call from evaluator |
| 6-value context coupling creates leaky abstraction | Medium | Medium | Create `EvaluationContext` dataclass to bundle values cleanly |
| EnsembleManager calls `_evaluate_prediction_dataframe()` | High | Medium | Keep thin delegation method on ForecastingModelManager |
| Step mapping logic drift during extraction | Low | High | Copy verbatim first; refactor later |

---

## 5. What NOT to change

- `prepare_actuals_df()` hook (must stay on ForecastingModelManager for subclass overrides)
- EvaluationManager / EvaluationAdapter interfaces (external packages)
- Step mapping arithmetic (proven correct after off-by-one fix)
- WandB logging format (step-wise, time-series-wise, month-wise)
