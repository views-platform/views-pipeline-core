# Class Intent Contract: EnsembleManager

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology), ADR-004 (Evolution), ADR-006 (Intent Contracts), ADR-036 (Ensemble Reconciliation)

---

## 1. Purpose

Orchestrates ensemble forecasting by coordinating N sub-models as a single
pipeline unit. Extends `ForecastingModelManager` to provide ensemble-specific
training, evaluation, forecasting, and reconciliation. Each sub-model is
executed as a separate shell subprocess; predictions are pooled via
`AggregationManager` and optionally reconciled via `ReconciliationModule` for
hierarchical PGM-CM consistency.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** implement model-specific training or inference logic. Sub-models
  are invoked via their own `main.py` as shell subprocesses.
- Does **not** own aggregation logic. That is `AggregationManager`'s
  responsibility.
- Does **not** own reconciliation logic. That is `ReconciliationModule`'s
  responsibility.
- Does **not** validate individual sub-model predictions. Each sub-model is
  responsible for its own validation when run as a subprocess.
- Does **not** support the PredictionFrame code path for sub-model outputs.
  Sub-model predictions are always loaded as `pd.DataFrame` from disk or the
  prediction store.
- Does **not** provide fine-grained error recovery for individual sub-model
  failures. If any subprocess fails, the exception propagates and the entire
  ensemble run fails.

---

## 3. Responsibilities and Guarantees

- Guarantees that all sub-models listed in `configs["models"]` are trained,
  evaluated, or forecasted (depending on the requested stages).
- Guarantees that `AggregationManager` is used to pool sub-model predictions
  using the aggregation method declared in `configs["aggregation"]`.
- Guarantees that all sub-models return the same number of evaluation outputs;
  raises `ValueError` if any model returns fewer outputs than expected.
- Guarantees that `ReconciliationModule` is applied during forecasting when
  `self.__activate_reconciliation` is `True` and `configs["reconciliation"]`
  is `"pgm_cm_point"`.
- Guarantees that `validate_ensemble_model(configs)` is called before
  execution when `args.train` is `False` (i.e., when using existing
  artifacts).
- Guarantees that `handle_ensemble_log_creation` is called after evaluation
  and forecasting for audit trail purposes.
- Guarantees that WandB alerts are sent on stage completion and on errors,
  inheriting the alert pattern from `ForecastingModelManager`.

---

## 4. Inputs and Assumptions

- `ensemble_path: EnsemblePathManager` -- must point to a valid ensemble
  directory under `ensembles/`.
- `args: ForecastingModelArgs` -- validated CLI arguments. Must be a
  `ForecastingModelArgs` instance; raises `ValueError` otherwise.
- `configs["models"]` -- a list of sub-model names (strings). Each must be
  a valid model name resolvable by `ModelPathManager`.
- `configs["aggregation"]` -- the aggregation method (e.g., `"mean"`,
  `"median"`, `"concat"`, `"vincentization"`).
- `configs["reconciliation"]` -- optional. If `"pgm_cm_point"`, hierarchical
  reconciliation is applied during forecasting.
- `configs["reconcile_with"]` -- optional. The CM ensemble model name used
  as the reconciliation target.
- Assumes that each sub-model has already been set up with its own config
  files, artifacts directory, and `main.py` entry point.
- Assumes that sub-model `main.py` scripts accept the same CLI arguments
  as `ForecastingModelArgs.to_shell_command()` produces.

---

## 5. Outputs and Side Effects

- Executes sub-model `main.py` as shell subprocesses via `subprocess.run()`.
- Creates aggregated prediction files in `ensemble_path.data_generated/`.
- Creates ensemble log entries via `handle_ensemble_log_creation`.
- During forecasting with reconciliation: loads a C-level dataset (from
  prediction store or local path), performs PGM-CM reconciliation, and saves
  the reconciled predictions.
- Creates WandB runs for `train`, `evaluate`, `forecast`, and `report`
  stages.

---

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| `args` is not `ForecastingModelArgs` | `ValueError` immediately |
| Sub-model subprocess fails | `PipelineException` with WandB alert |
| Sub-models return different numbers of outputs | `ValueError` in `_evaluate_ensemble()` |
| No prediction files found after subprocess | `PipelineException` |
| Aggregation with zero DataFrames | `ValueError` in `_get_aggregated_df()` |
| Reconciliation returns `None` | WandB warning alert; original predictions returned unmodified |
| C dataset not found for reconciliation | `None` returned from `_load_c_dataset()`; reconciliation skipped with warning |
| Training/evaluation/forecasting exception | `PipelineException` with full traceback and WandB alert |

All failures are loud. Subprocess failures propagate via `check=True` on
`subprocess.run()`.

---

## 7. Boundaries and Interactions

```
EnsembleManager (extends ForecastingModelManager)
    |-- EnsemblePathManager        (path resolution for ensembles/)
    |-- ModelPathManager           (path resolution for each sub-model)
    |-- ForecastingModelArgs       (CLI args, converted to shell commands)
    |-- subprocess.run()           (sub-model execution)
    |-- AggregationManager         (prediction pooling)
    |-- ReconciliationModule       (PGM-CM hierarchical reconciliation)
    |-- _PGDataset / _CDataset     (dataset wrappers for reconciliation)
    |-- WandBModule                (inherited: alerts, run lifecycle)
    |-- PredictionIOManager        (inherited: save predictions)
    |-- ViewsDataLoader            (inherited: data fetching)
```

- `EnsembleManager` inherits all pipeline stage orchestration from
  `ForecastingModelManager` but overrides `execute_single_run`,
  `_execute_model_tasks`, `_execute_model_training`,
  `_execute_model_evaluation`, and `_execute_model_forecasting`.
- It does **not** call `CoreConfigSniffer` directly; the parent's
  `execute_single_run` sniffing is replaced by
  `validate_ensemble_model(configs)`.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.ensemble import EnsemblePathManager, EnsembleManager
from views_pipeline_core.cli import ForecastingModelArgs

ensemble_path = EnsemblePathManager("mighty_coalition")
manager = EnsembleManager(
    ensemble_path=ensemble_path,
    wandb_notifications=True,
    use_prediction_store=False,
)
args = ForecastingModelArgs(
    run_type="calibration",
    train=True,
    evaluate=True,
    forecast=False,
    saved=False,
    eval_type="standard",
)
manager.execute_single_run(args)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: passing ModelPathManager instead of EnsemblePathManager
manager = EnsembleManager(ensemble_path=ModelPathManager("purple_alien"))
# -> Will look in models/ instead of ensembles/

# WRONG: configs["models"] is empty
# -> _train_ensemble() iterates zero models; no training occurs; evaluation
#    will fail with missing predictions

# WRONG: sub-models not trained before evaluate-only run without validation
args = ForecastingModelArgs(run_type="calibration", train=False, evaluate=True)
manager.execute_single_run(args)
# -> validate_ensemble_model() may pass, but subprocess evaluation will fail
#    if artifacts are missing
```

---

## 10. Test Alignment

- `tests/test_managers/test_ensemble_manager.py` -- tests for ensemble
  initialization, sub-model coordination, aggregation, and reconciliation.

---

## 11. Evolution Notes

- `EnsembleManager` originally overrode the abstract methods
  `_train_model_artifact`, `_evaluate_model_artifact`, and
  `_forecast_model_artifact` from `ForecastingModelManager`. It now overrides
  the higher-level `_execute_model_*` methods instead, because ensemble
  orchestration is fundamentally different (subprocess dispatch, aggregation,
  reconciliation).
- The `_load_or_generate_prediction` helper implements a cache-first pattern:
  check prediction store or local file before invoking the subprocess.
- Weighted aggregation was added via `configs["use_weights"]` and
  `configs["weights"]` dict.

---

## 12. Known Deviations

- **Uses `models[0]` for actuals (R3):** When `_evaluate_prediction_dataframe`
  is called with `ensemble=True`, actuals are loaded from `models[0]`'s raw
  data directory. This assumes all sub-models share the same ground-truth
  data, which is true for current ensembles but is not enforced structurally.
- **Subprocess execution for sub-models:** Sub-models are invoked via
  `subprocess.run()` with `check=True`. Error propagation relies on the
  subprocess exit code; stderr output is not captured or parsed. A sub-model
  that fails silently (exit code 0 but produces garbage) will not be detected
  until aggregation or evaluation.
- **No `CoreConfigSniffer` call:** `EnsembleManager.execute_single_run()`
  replaces the parent's pre-flight sniffing with
  `validate_ensemble_model(configs)`, which has a different (narrower) scope.
- **Reconciliation is forecasting-only:** The `__activate_reconciliation`
  flag is checked only in `_forecast_ensemble()`. Reconciliation is not
  applied during evaluation, so evaluation metrics reflect un-reconciled
  predictions.

---

## End of Contract

This document defines the **intended meaning** of `EnsembleManager`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
