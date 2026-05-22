# Class Intent Contract: PredictionFrameEnsembleManager

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-05-21
**Related ADRs:** ADR-042 (PredictionFrame), ADR-045 (Stage Pattern), ADR-051 (Composition-Based Ensemble Architecture)

---

## 1. Purpose

Composition-based ensemble orchestrator for the PredictionFrame (numpy) prediction
path. ADR-051 Phase 2. Resolves C-66 (OOM on DataFrame ensemble aggregation with
sample-based models) by keeping predictions in numpy format throughout — no
conversion to DataFrame or parquet at any point.

Handles HydraNet-class ensembles where each sub-model produces `(N, S)` posterior
sample arrays. Aggregation is pure numpy: `np.concatenate` (stacking samples) or
`np.mean(np.stack(...))` (averaging samples).

All ensemble-specific business logic is intentionally copied from
`DataFrameEnsembleManager` (WET-before-DRY). The architectural difference is the
data format: `PredictionFrame` numpy arrays instead of `pd.DataFrame`.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** inherit from `ForecastingModelManager`, `ModelManager`, or any base
  class. This is a deliberate architectural choice, not an omission.
- Does **not** replace `EnsembleManager` or `DataFrameEnsembleManager`. All three
  classes coexist (Strangler Fig pattern). Existing ensembles continue unchanged.
- Does **not** support the DataFrame/parquet code path. That belongs to
  `DataFrameEnsembleManager`.
- Does **not** implement reconciliation. Reconciliation is a point-prediction
  operation; PredictionFrame carries posterior samples where reconciliation has
  no defined semantics.
- Does **not** implement model-specific training or inference. Sub-models are
  invoked via shell subprocesses.
- Does **not** own complex aggregation logic (`AggregationModule`). Uses pure
  numpy functions for concat and arithmetic mean.
- Does **not** manage WandB run lifecycle inside stages. The `_execute_*` methods
  are the lifecycle boundary; stages contain only business logic.
- Does **not** support prediction store upload (`use_prediction_store` is always
  `False` for sub-model dispatch). No prediction store API exists for numpy format.

---

## 3. Responsibilities and Guarantees

- Guarantees that predictions **never** convert to DataFrame or parquet. The
  entire pipeline is numpy: `y_pred.npy` + `identifiers.npz` via
  `PredictionFrame.save()/load()`.
- Guarantees that `CoreConfigSniffer.sniff_all()` runs before WandB login or any
  model execution.
- Guarantees that an immutable `EnsembleContext` (frozen dataclass, imported from
  `dataframe_ensemble.py`) is built once in `execute_single_run()` and threaded to
  every method. Context always has `prediction_format="prediction_frame"` and
  `reconciliation=None`.
- Guarantees that `EvaluationStage` and `ReportingStage` are used via composition
  (injected at construction), not via inheritance.
- Guarantees that all sub-models in `configs["models"]` are trained, evaluated,
  or forecasted depending on requested stages.
- Guarantees that `_aggregate_prediction_frames()` validates input before
  aggregation: non-empty list, identical `n_rows`, identical identifiers (time/unit
  arrays must be `np.array_equal`), supported method.
- Guarantees that all sub-models return the same number of evaluation outputs per
  target; raises `ValueError` if any model returns fewer.
- Guarantees that `validate_ensemble_model(configs)` is called before execution
  when `args.train is False`.
- Guarantees that subprocess execution uses `timeout=7200` seconds.
- Guarantees cache-first loading: `_load_or_generate_pf()` checks for existing
  `y_pred.npy` before dispatching a subprocess.
- Guarantees that aggregated PFs are saved to disk via `PredictionFrame.save()`
  before evaluation or reporting.

---

## 4. Inputs and Assumptions

- `ensemble_path: EnsemblePathManager` -- must point to a valid ensemble
  directory under `ensembles/`.
- `wandb_notifications: bool` -- gate for WandB alerts.
- `use_prediction_store: bool` -- accepted for API symmetry with
  `DataFrameEnsembleManager` but functionally unused (sub-models always get
  `prediction_store=False`).
- `execute_single_run(args: ForecastingModelArgs)` -- must receive a
  `ForecastingModelArgs` instance; raises `ValueError` otherwise.
- Config files (`config_deployment.py`, `config_hyperparameters.py`,
  `config_meta.py`, `config_partitions.py`) are loaded via `importlib` from
  `ensemble_path.get_scripts()`.
- `configs["models"]` -- list of sub-model names, each resolvable by
  `ModelPathManager`.
- `configs["aggregation"]` -- must be in `SUPPORTED_PF_AGGREGATION_METHODS`:
  `{"concat", "arithmetic_mean"}`.
- `configs["targets"]` or `configs["regression_targets"]` -- list of target names.
- Assumes sub-model `main.py` scripts accept `ForecastingModelArgs.to_shell_command()`
  arguments and produce PredictionFrame outputs at:
  - Evaluation: `data_generated/predictions_{run_type}_{ts}/origin_{i}/{target}/y_pred.npy`
  - Forecast: `data_generated/predictions_{run_type}_{ts}/{target}/y_pred.npy`

---

## 5. Outputs and Side Effects

- Executes sub-model `main.py` as shell subprocesses via `subprocess.run()`.
- Creates aggregated PredictionFrame directories in `ensemble_path.data_generated/`:
  - Evaluation: `predictions_{run_type}_{ts}/origin_{i}/{target}/y_pred.npy`
  - Forecast: `predictions_{run_type}_{ts}/{target}/y_pred.npy`
- Creates ensemble log entries via `handle_ensemble_log_creation`.
- Creates WandB runs for `train`, `evaluate`, `forecast`, and `report` stages.
- Delegates evaluation to `EvaluationStage.evaluate()` with `ensemble=True` and
  `prediction_format="prediction_frame"`.
- Delegates reporting to `ReportingStage.generate_forecast_report()` and
  `generate_evaluation_report()`.
- Uses `mmap=True` when loading sub-model PFs during evaluation (memory-bounded
  sequential access).

---

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| `args` is not `ForecastingModelArgs` | `ValueError` immediately |
| `CoreConfigSniffer.sniff_all()` fails | Exception propagates; no WandB login or execution |
| `wandb_module.login()` fails | `RuntimeError` propagates (C-85 coverage) |
| Sub-model subprocess fails | `PipelineException` with WandB alert |
| Sub-model subprocess times out (>7200s) | `PipelineException` with timeout message |
| Sub-models return different output counts | `ValueError` in `_evaluate_ensemble()` |
| No PredictionFrame output found after subprocess | `PipelineException` with `logger.error` (ADR-008 compliant) |
| `_aggregate_prediction_frames([])` | `ValueError`: "requires at least one PredictionFrame" |
| Mismatched `n_rows` between PFs | `ValueError` identifying which frame mismatches |
| Mismatched identifiers between PFs | `ValueError` identifying which key differs |
| Unsupported aggregation method | `ValueError` listing supported methods |
| Sub-model did not produce forecast for target | `ValueError` in `_forecast_ensemble()` |
| Training/evaluation/forecasting exception | `PipelineException` with traceback and WandB alert |

All failures are loud. No silent fallbacks.

---

## 7. Boundaries and Interactions

```
PredictionFrameEnsembleManager (composition, no inheritance)
    |-- EnsemblePathManager        (path resolution for ensembles/)
    |-- ModelPathManager           (path resolution for each sub-model)
    |-- ForecastingModelArgs       (CLI args, converted to shell commands)
    |-- subprocess.run()           (sub-model execution, timeout=7200)
    |-- CoreConfigSniffer          (pre-flight config validation)
    |-- ConfigurationManager       (config merging and WandB integration)
    |-- _aggregate_prediction_frames()  (pure numpy aggregation)
    |-- PredictionFrame.save/load  (numpy persistence, mmap support)
    |
    |-- Composed stages (ADR-045):
    |   |-- EvaluationStage        (metric computation, ensemble=True)
    |   |-- ReportingStage         (HTML report generation)
    |
    |-- WandBModule                (alerts, run lifecycle)
    |-- LoggingModule              (logging setup)
```

- **Depends on:** `ForecastingModelManager._resolve_evaluation_sequence_number()`
  (static method) for determining evaluation sequence count.
- **Depends on:** `EnsembleContext` (frozen dataclass from `dataframe_ensemble.py`).
- **Does not depend on:** `ForecastingModelManager` instance state, `ViewsDataLoader`,
  `ForecastingStage`, `PredictionIOManager`, `AggregationModule`,
  `ReconciliationModule`, or any DataFrame conversion.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.ensemble import PredictionFrameEnsembleManager, EnsemblePathManager
from views_pipeline_core.cli.args import ForecastingModelArgs

ensemble_path = EnsemblePathManager("synthetic_chorus")
manager = PredictionFrameEnsembleManager(
    ensemble_path=ensemble_path,
    wandb_notifications=True,
)
args = ForecastingModelArgs(
    run_type="calibration",
    train=False,
    evaluate=True,
    forecast=True,
    saved=True,
    eval_type="standard",
)
manager.execute_single_run(args)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: passing ModelPathManager instead of EnsemblePathManager
manager = PredictionFrameEnsembleManager(ensemble_path=ModelPathManager("purple_alien"))
# -> Will look in models/ instead of ensembles/

# WRONG: using for DataFrame-based models (parquet predictions)
# -> PredictionFrameEnsembleManager only handles PredictionFrame (numpy) outputs.
#    Use DataFrameEnsembleManager for point-prediction ensembles.

# WRONG: expecting reconciliation to work
# -> reconciliation is always None. PredictionFrame carries posterior samples;
#    hierarchical reconciliation has no defined semantics for sample arrays.

# WRONG: calling methods directly without execute_single_run()
manager._evaluate_ensemble(ctx)
# -> EnsembleContext is not built; _args is None; CoreConfigSniffer has not run.

# WRONG: using aggregation methods other than concat/arithmetic_mean
# -> configs["aggregation"] = "median" will raise ValueError
```

---

## 10. Test Alignment

- `tests/test_managers/test_prediction_frame_ensemble_manager.py` -- 43
  characterization tests across 8 test classes:
  - `TestPredictionFrameAggregation` (9 tests) -- verifies concat stacks samples
    axis, arithmetic mean averages correctly, identifiers are preserved, mismatches
    raise, empty list raises, unsupported method raises.
  - `TestPredictionFrameEnsembleConstants` (4 tests) -- verifies
    `SUPPORTED_PF_AGGREGATION_METHODS` is frozenset, EnsembleContext accepts
    `prediction_format="prediction_frame"`, context is frozen.
  - `TestPredictionFrameEnsembleConstruction` (6 tests) -- verifies no inheritance
    from ModelManager/ForecastingModelManager, composed stages exist, path stored.
  - `TestPredictionFrameLoading` (5 tests) -- verifies cache-first loading, generate
    when missing, exception when no output, mmap flag passthrough.
  - `TestPredictionFrameEvaluationFlow` (2 tests) -- verifies aggregation per
    sequence and prediction_format propagation to EvaluationStage.
  - `TestPredictionFrameForecastingFlow` (4 tests) -- verifies aggregation across
    models, PF save, no reconciliation method, return type is Dict[str, PF].
  - `TestPredictionFrameTrainingFlow` (2 tests) -- verifies subprocess dispatch per
    model and train flag passthrough.
  - `TestPredictionFrameEntryPoint` (11 tests) -- verifies args validation, sniffer
    call, wandb login (and failure propagation), task dispatch, subprocess timeout/
    failure → PipelineException, validation gating.

---

## 11. Evolution Notes

- This class proves the PredictionFrame path for ensemble orchestration, enabling
  HydraNet production ensembles with 64 posterior samples per observation.
- Business logic is intentionally WET (copied from `DataFrameEnsembleManager`).
  Once both composition-based managers are proven in production, shared abstractions
  may be extracted.
- The dependency on `ForecastingModelManager._resolve_evaluation_sequence_number()`
  is a static method call. It may be relocated to a utility module.
- Future work: `prediction_store` support for numpy format (blocked on views-forecasts
  package numpy API). When available, `_create_model_args` can pass
  `prediction_store=True`.
- Sub-model numpy directory discovery relies on explicit path construction from the
  model artifact timestamp. The utility method `_get_generated_pf_prediction_paths()`
  on `ModelPathManager` provides an alternative lookup mechanism.

---

## 12. Known Deviations

- **No reconciliation:** Unlike `DataFrameEnsembleManager`, this class has no
  reconciliation path. The `reconciliation` field in `EnsembleContext` is always
  set to `None`.
- **No `PredictionIOManager`:** `EvaluationStage` is constructed with
  `io_manager=None`. All persistence is via `PredictionFrame.save()` directly.
- **Subprocess stderr not captured:** Sub-models invoked via `subprocess.run()`
  with `check=True`. A sub-model that fails silently (exit code 0, garbage output)
  is not detected until aggregation or loading.
- **`handle_ensemble_log_creation` called in both eval and forecast
  `_execute_*` methods:** This matches the DataFrameEnsembleManager pattern.
  Log creation is an orchestration-level side effect, not a business-logic concern.
- **`EnsembleContext` imported from `dataframe_ensemble.py`:** Shared frozen
  dataclass. If the two managers diverge in context needs, a separate context
  class should be created.

---

## End of Contract

This document defines the **intended meaning** of `PredictionFrameEnsembleManager`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
