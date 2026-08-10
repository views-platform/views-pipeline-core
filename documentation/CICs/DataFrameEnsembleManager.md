# Class Intent Contract: DataFrameEnsembleManager

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-05-18
**Related ADRs:** ADR-045 (Stage Pattern), ADR-051 (Composition-Based Ensemble Architecture)

---

## 1. Purpose

Composition-based ensemble orchestrator for the DataFrame prediction path. Proves
the ADR-045 stage composition pattern for ensembles without inheriting from
`ForecastingModelManager` or `ModelManager`. Fixes C-55 by including
`CoreConfigSniffer.sniff_all()` in `execute_single_run()` before any side effects.

All ensemble-specific business logic (aggregation, reconciliation, subprocess
delegation) is intentionally copied from `EnsembleManager` (WET-before-DRY). The
architectural difference is HOW infrastructure is accessed: composition of explicit
collaborators vs. inheritance chain.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** inherit from `ForecastingModelManager`, `ModelManager`, or any base
  class. This is a deliberate architectural choice, not an omission.
- Does **not** replace `EnsembleManager`. Both classes coexist. Existing ensembles
  continue to use `EnsembleManager`; this class is a proving ground for composition.
- Does **not** support the PredictionFrame code path. That is reserved for the
  future `PredictionFrameEnsembleManager`.
- Does **not** implement model-specific training or inference. Sub-models are
  invoked via shell subprocesses.
- Does **not** own aggregation logic (`AggregationModule`) or reconciliation
  logic (`ReconciliationModule`).
- Does **not** manage WandB run lifecycle inside stages. The `_execute_*` methods
  are the lifecycle boundary; stages contain only business logic.

---

## 3. Responsibilities and Guarantees

- Guarantees that `CoreConfigSniffer.sniff_all()` runs before WandB login or any
  model execution (C-55 fix). This is the primary behavioral difference from
  `EnsembleManager`.
- Guarantees that an immutable `EnsembleContext` (frozen dataclass,
  `managers/ensemble/context.py`) is built once in `execute_single_run()` and threaded
  to every method, via the shared `EnsembleContext.from_config()` (#432). No method
  reads mutable `self` state for config, args, or run-type during execution.
  `prediction_format` is read from config with a `"dataframe"` fallback;
  `expected_samples_per_model` is not passed on this path.
- Guarantees that `EvaluationStage`, `PredictionIOManager`, and `ReportingStage`
  are used via composition (injected at construction), not via inheritance.
- Guarantees that all sub-models in `configs["models"]` are trained, evaluated,
  or forecasted depending on requested stages.
- Guarantees that `AggregationModule` pools sub-model predictions using
  `configs["aggregation"]` with optional per-model weights.
- Guarantees that all sub-models return the same number of evaluation outputs;
  raises `ValueError` if any model returns fewer.
- Guarantees that `ReconciliationModule` is applied during forecasting when
  `ctx.reconciliation == "pgm_cm_point"`.
- Guarantees that `validate_ensemble_model(configs, saved=args.saved)` is called
  before execution when `args.train is False`. Data freshness checks (Conditions
  2+3) are only enforced for forecasting runs with non-saved data (ADR-018
  amendment).
- Guarantees that `_create_model_args()` forces `saved=True` for all
  non-training subprocess dispatch, independent of the caller's `saved` flag.
- Guarantees that subprocess execution uses `timeout=7200` seconds.

---

## 4. Inputs and Assumptions

- `ensemble_path: EnsemblePathManager` -- must point to a valid ensemble
  directory under `ensembles/`.
- `wandb_notifications: bool` -- gate for WandB alerts.
- `use_prediction_store: bool` -- enables prediction store upload via
  `PredictionIOManager`.
- `execute_single_run(args: ForecastingModelArgs)` -- must receive a
  `ForecastingModelArgs` instance; raises `ValueError` otherwise.
- Config files (`config_deployment.py`, `config_hyperparameters.py`,
  `config_meta.py`, `config_modelset.py`, `config_partitions.py`) are loaded
  via `importlib` from `ensemble_path.get_scripts()`.
- `config_modelset.py` -- optional. When present, its keys are merged into
  `config_meta` (modelset values take precedence). Collision warning logged.
  Contains the `"models"` list for ensemble constituent models.
- `configs["models"]` -- list of sub-model names, each resolvable by
  `ModelPathManager`.
- `configs["aggregation"]` -- aggregation method (e.g., `"mean"`, `"median"`,
  `"concat"`, `"vincentization"`).
- `configs["regression_targets"]` and/or `configs["classification_targets"]` -- lists of
  target names. The pooled target list is **derived** from both by `combined_targets`
  (`managers/configuration/configuration.py`), regression first, then classification —
  identical to `PredictionFrameEnsembleManager`. A legacy `configs["targets"]` key is
  **retired** (#380) and `combined_targets` raises `ValueError` on it (C-132, #422).
- `configs["reconciliation"]` -- optional. `"pgm_cm_point"` enables hierarchical
  reconciliation during forecasting.
- `configs["reconcile_with"]` -- optional. CM model name for reconciliation target.
- Assumes sub-model `main.py` scripts accept `ForecastingModelArgs.to_shell_command()`
  arguments.

---

## 5. Outputs and Side Effects

- Executes sub-model `main.py` as shell subprocesses via `subprocess.run()`.
- Creates aggregated prediction files in `ensemble_path.data_generated/`.
- Creates ensemble log entries via `handle_ensemble_log_creation`.
- During forecasting with reconciliation: loads a C-level dataset, performs PGM-CM
  reconciliation, and saves reconciled predictions.
- Creates WandB runs for `train`, `evaluate`, `forecast`, and `report` stages.
- Delegates evaluation to `EvaluationStage.evaluate()` with `ensemble=True` and
  `prediction_format="dataframe"`.
- Delegates reporting to `ReportingStage.generate_forecast_report()` and
  `generate_evaluation_report()`.

---

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| `args` is not `ForecastingModelArgs` | `ValueError` immediately |
| `CoreConfigSniffer.sniff_all()` fails | Exception propagates; no WandB login or execution |
| Sub-model subprocess fails | `PipelineException` with WandB alert |
| Sub-model subprocess times out (>7200s) | `PipelineException` with timeout message |
| Sub-models return different output counts | `ValueError` in `_evaluate_ensemble()` |
| No prediction files found after subprocess | `PipelineException` |
| Aggregation with zero DataFrames | `ValueError` in `_get_aggregated_df()` |
| `AggregationModule.prediction_type` is `None` | `RuntimeError` |
| Reconciliation returns `None` | `PipelineException` with WandB `ERROR` alert; unreconciled predictions never published |
| C dataset not found for reconciliation | `None` from `_load_c_dataset()`; `_apply_reconciliation()` raises `PipelineException` |
| Training/evaluation/forecasting exception | `PipelineException` with traceback and WandB alert |

All failures are loud. No silent fallbacks.

---

## 7. Boundaries and Interactions

```
DataFrameEnsembleManager (composition, no inheritance)
    |-- EnsemblePathManager        (path resolution for ensembles/)
    |-- ModelPathManager           (path resolution for each sub-model)
    |-- ForecastingModelArgs       (CLI args, converted to shell commands)
    |-- subprocess.run()           (sub-model execution, timeout=7200)
    |-- CoreConfigSniffer          (pre-flight config validation, C-55 fix)
    |-- ConfigurationManager       (config merging and WandB integration)
    |-- AggregationModule         (prediction pooling)
    |-- ReconciliationModule       (PGM-CM hierarchical reconciliation)
    |-- _PGDataset / _CDataset     (dataset wrappers for reconciliation)
    |
    |-- Composed stages (ADR-045):
    |   |-- EvaluationStage        (metric computation, ensemble=True)
    |   |-- PredictionIOManager    (prediction/evaluation persistence)
    |   |-- ReportingStage         (HTML report generation)
    |
    |-- WandBModule                (alerts, run lifecycle)
    |-- LoggingModule              (logging setup)
```

- **Depends on:** `ForecastingModelManager._resolve_evaluation_sequence_number()`
  (static method) for determining evaluation sequence count.
- **Does not depend on:** `ForecastingModelManager` instance state, `ViewsDataLoader`,
  `ForecastingStage`, or `TrainingStage`.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.ensemble import DataFrameEnsembleManager, EnsemblePathManager
from views_pipeline_core.cli.args import ForecastingModelArgs

ensemble_path = EnsemblePathManager("mighty_coalition")
manager = DataFrameEnsembleManager(
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
manager = DataFrameEnsembleManager(ensemble_path=ModelPathManager("purple_alien"))
# -> Will look in models/ instead of ensembles/

# WRONG: using for PredictionFrame-based models
# -> DataFrameEnsembleManager only handles pd.DataFrame predictions.
#    Use PredictionFrameEnsembleManager (future) for sample-based outputs.

# WRONG: calling methods directly without execute_single_run()
manager._execute_model_training(ctx)
# -> EnsembleContext is not built; _args is None; CoreConfigSniffer has not run.
```

---

## 10. Test Alignment

- `tests/test_managers/test_dataframe_ensemble_manager.py` -- 45 characterization
  tests across 13 test classes:
  - `TestNoInheritance` -- verifies no inheritance from ForecastingModelManager
    or ModelManager, and that composed stages exist as attributes.
  - `TestEnsembleContext` -- verifies immutability, field population, and
    BaseStageContext ancestry.
  - `TestCoreConfigSnifferIntegration` -- verifies sniff_all() runs before WandB
    login, and that sniffer failure prevents all execution (C-55 fix).
  - `TestAggregationDelegation` -- verifies AggregationModule receives correct
    add_model calls, weight passthrough, and empty-input rejection.
  - `TestEvaluationDelegation` -- verifies EvaluationStage.evaluate() is called
    with `ensemble=True`, `prediction_format="dataframe"`, and `data_loader=None`.
  - `TestReportingDelegation` -- verifies ReportingStage delegation for forecast
    and evaluation reports.
  - `TestExecutionFlow` -- verifies args validation, WandB login, task dispatch,
    and ensemble validation gating.
  - `TestCreateModelArgs` -- verifies ForecastingModelArgs construction for
    training, evaluation, and forecasting modes.
  - `TestSubprocessTimeout` -- verifies timeout=7200 passed to subprocess.run().
  - `TestFailureModes` -- verifies PipelineException on subprocess failure,
    training failure, forecasting failure, and mismatched model output counts.
  - `TestReconciliation` -- verifies reconciliation dispatch, pgm_cm_point
    application, and fallback when reconciler returns None.
  - `TestEntityRenameDataFrameEnsemble` -- verifies ADR-034
    `priogrid_gid` → `priogrid_id` rename in aggregation dispatch.
  - `TestConfigModelsetMerge` -- verifies config_modelset → effective_meta
    merge precedence, collision warning, copy semantics (original
    config_meta unchanged), and no-op when config_modelset is absent.

---

## 11. Evolution Notes

- This class is a proving ground for composition-based ensemble orchestration.
  If successful, the pattern will be used for `PredictionFrameEnsembleManager`
  (HydraNet production ensembles with 64 posterior samples).
- Business logic is intentionally WET (copied from `EnsembleManager`). Once both
  composition-based managers are proven, shared abstractions may be extracted.
- The dependency on `ForecastingModelManager._resolve_evaluation_sequence_number()`
  is a static method call, not an instance dependency. It may be relocated to a
  utility module in the future.

---

## 12. Known Deviations

- **Uses `models[0]` for actuals:** When `EvaluationStage.evaluate()` is called
  with `ensemble=True`, actuals are loaded from `models[0]`'s raw data directory.
  This assumes all sub-models share the same ground-truth data.
- **Subprocess stderr not captured:** Sub-models invoked via `subprocess.run()`
  with `check=True`. A sub-model that fails silently (exit code 0, garbage output)
  is not detected until aggregation or evaluation.
- **Reconciliation is forecasting-only:** Reconciliation is applied only in
  `_forecast_ensemble()`, not during evaluation.
- **`wandb.AlertLevel.WARN` usage:** The reconciliation None-result path uses
  `wandb.AlertLevel.WARN` (the old `EnsembleManager` used the non-existent
  `AlertLevel.WARNING`; this class corrects that).

---

## End of Contract

This document defines the **intended meaning** of `DataFrameEnsembleManager`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
