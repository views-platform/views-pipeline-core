# Class Intent Contract: ForecastingModelManager

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-08
**Related ADRs:** ADR-001 (Ontology), ADR-004 (Evolution), ADR-006 (Intent Contracts), ADR-008 (Observability), ADR-040 (Authority), ADR-041 (Sniffer Pattern)

---

## 1. Purpose

Central orchestrator for the forecasting model lifecycle. Manages the complete
pipeline of data fetching, training, evaluation, forecasting, and reporting for
a single model. Extends `ModelManager` and delegates persistence to
`PredictionIOManager`, validation to the three core sniffers
(`CoreConfigSniffer`, `CoreDataSniffer`, `CorePredictionSniffer`), and
configuration to `ConfigurationManager`.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** implement model-specific training, evaluation, or forecasting
  logic. Those are abstract methods (`_train_model_artifact`,
  `_evaluate_model_artifact`, `_forecast_model_artifact`, `_evaluate_sweep`)
  that must be implemented by concrete subclasses in `views-models`.
- Does **not** own prediction persistence. All save/load operations delegate to
  `PredictionIOManager` via `self._io`.
- Does **not** own configuration merging or validation. That is
  `ConfigurationManager`'s responsibility.
- Does **not** own data loading or queryset execution. That is
  `ViewsDataLoader`'s responsibility.
- Does **not** perform model-specific config validation. Only the universal
  pipeline contract is enforced via `CoreConfigSniffer`.
- Does **not** own WandB session lifecycle details. That is `WandBModule`'s
  responsibility.

---

## 3. Responsibilities and Guarantees

- Guarantees that `CoreConfigSniffer.sniff_all()` is called before any
  side effects (WandB login, data fetching, inference) in
  `execute_single_run()` and `execute_sweep_run()`.
- Guarantees that `_assert_partition_config_accessible()` is called as a
  Layer 1 structural pre-condition check before `CoreConfigSniffer` runs.
- Guarantees that `CorePredictionSniffer.sniff_predictions()` is called on
  every prediction DataFrame before it is saved (DF path). On the PF path,
  `PredictionFrame` is self-validating at construction.
- Guarantees that type enforcement guards (ADR-042) reject mismatched return
  types: a `dict` return when `prediction_format="dataframe"` is declared, or
  a non-dict return when `prediction_format="prediction_frame"` is declared.
- Guarantees that `_assert_predictions_in_step_window()` validates temporal
  coverage of all prediction sequences against the declared step mapping
  before per-target evaluation begins (DF path only).
- Guarantees that `_get_evaluation_step_mappings()` is the sole authority on
  lead-time-to-month mappings (ADR-031), anchoring each rolling-origin
  sequence at `base_origin + i`.
- Guarantees that pipeline stages (train, evaluate, forecast, report) are
  executed in order and only when requested via `ForecastingModelArgs` flags.
- Guarantees that all pipeline exceptions are wrapped in domain-specific
  exception types (`ModelTrainingException`, `ModelEvaluationException`,
  `ModelForecastingException`, `DataFetchException`) and propagated with
  WandB alerts.

---

## 4. Inputs and Assumptions

- `model_path: ModelPathManager` -- must be a valid, initialized
  `ModelPathManager` pointing to an existing model directory.
- `args: ForecastingModelArgs` -- validated CLI arguments passed to
  `execute_single_run()` or `execute_sweep_run()`. Must be an instance of
  `ForecastingModelArgs`; a `ValueError` is raised otherwise.
- Assumes that the model directory contains valid config scripts
  (`config_deployment.py`, `config_hyperparameters.py`, `config_meta.py`,
  `config_partitions.py`) importable via `importlib`.
- Assumes that `self._partition_dict` contains the declared `run_type` key
  with `"train"` and `"test"` sub-tuples for non-forecasting runs.
- Assumes that subclass implementations of `_evaluate_model_artifact` return
  exactly `MAX_SHIFT_COUNT + 1` (13) sequences for `"standard"` evaluation.
- When `use_prediction_store=True`, `PredictionStoreConfig.from_environment()`
  is called at construction to validate all 9 required Appwrite env vars.
  This fails loud with `ConfigurationException` if any are missing — before
  any compute is invested. The config is then converted to `AppwriteConfig`
  via `to_appwrite_config()` and used to construct `DatastoreModule`.
- The `prediction_format` config key determines which code path is taken:
  `"dataframe"` (legacy `List[pd.DataFrame]`) or `"prediction_frame"`
  (`Dict[str, List[PredictionFrame]]`).

---

## 5. Outputs and Side Effects

- Creates model artifacts in `model_path.artifacts/` (via subclass).
- Creates prediction files in `model_path.data_generated/` (via
  `PredictionIOManager`).
- Creates evaluation metric files (step-wise, time-series-wise, month-wise)
  in `model_path.data_generated/`.
- Creates HTML reports in `model_path.reports/`.
- Creates WandB runs for each pipeline stage (`fetch_data`, `train`,
  `evaluate`, `forecast`, `report`, `sweep`).
- Sends WandB alerts on stage completion and on errors.
- Logs execution timing at pipeline completion.
- On the PF path, creates and cleans up a `_pf_staging/` directory under
  `data_generated/` for intermediate numpy files during streaming evaluation.

---

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| `args` is not `ForecastingModelArgs` | `ValueError` immediately |
| Partition config missing for `run_type` | `KeyError` from `_assert_partition_config_accessible()` before any side effects |
| `CoreConfigSniffer` detects violation | `KeyError` / `ValueError` / `NotImplementedError` immediately |
| Data fetch fails | `DataFetchException` with WandB alert |
| Training fails | `ModelTrainingException` with WandB alert |
| Evaluation fails | `ModelEvaluationException` with WandB alert |
| Forecasting fails | `ModelForecastingException` with WandB alert |
| Return type mismatches `prediction_format` | `ValueError` (ADR-042 fail-loud guard) |
| Prediction temporal coverage outside step window | `ValueError` from `_assert_predictions_in_step_window()` with diagnostic hints |
| Wrong number of evaluation sequences | `ValueError` from `_assert_predictions_in_step_window()` |

All failures are loud. No silent fallbacks, no boolean returns from validators.

---

## 7. Boundaries and Interactions

```
CLI args (ForecastingModelArgs)
    |
    v
ForecastingModelManager
    |-- CoreConfigSniffer          (pre-flight validation)
    |-- ViewsDataLoader            (data fetching)
    |-- ConfigurationManager       (config merging)
    |-- WandBModule                (session lifecycle, alerts)
    |-- PredictionIOManager        (save/load predictions, evaluations)
    |-- EvaluationStage            (metrics orchestration, ADR-045 E2)
    |-- ReportingStage             (report generation, ADR-045 E3)
    |-- ForecastingStage           (forecast post-processing, ADR-045 E4)
    |-- TrainingStage              (training post-processing, ADR-045 E5)
    |-- CorePredictionSniffer      (DF path: prediction validation)
    |-- PredictionFrame            (PF path: self-validating)
    |
    v
Subclass in views-models (implements abstract methods)
```

- `ForecastingModelManager` is the **parent** of `EnsembleManager`.
- It does **not** call `CoreDataSniffer` directly; that is
  `ViewsDataLoader`'s responsibility.
- It delegates all persistence to `PredictionIOManager` (extracted in E1
  refactoring, commit `017c85a`).
- It delegates report generation to `ReportingStage` (extracted in E3
  refactoring). `ForecastReportTemplate` and `EvaluationReportTemplate`
  are now consumed by the stage, not by the facade directly.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.cli import ForecastingModelArgs

# In views-models, a concrete subclass:
class MyModelManager(ForecastingModelManager):
    def _train_model_artifact(self):
        ...  # train and save artifact

    def _evaluate_model_artifact(self, eval_type, artifact_name):
        ...  # return List[pd.DataFrame] or Dict[str, List[PredictionFrame]]

    def _forecast_model_artifact(self, artifact_name):
        ...  # return pd.DataFrame or Dict[str, PredictionFrame]

    def _evaluate_sweep(self, eval_type, model):
        ...  # return predictions for sweep iteration

# Execution
model_path = ModelPathManager("purple_alien")
manager = MyModelManager(model_path=model_path, wandb_notifications=True)
args = ForecastingModelArgs.parse_args()
manager.execute_single_run(args)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: passing raw dict instead of ForecastingModelArgs
manager.execute_single_run({"run_type": "calibration"})
# -> ValueError

# WRONG: calling _execute_model_evaluation() directly without execute_single_run()
manager._execute_model_evaluation()
# -> AttributeError (self._args not set, self._project not set)

# WRONG: subclass returns dict when prediction_format="dataframe"
def _evaluate_model_artifact(self, eval_type, artifact_name):
    return {"target": [pf1, pf2]}  # dict, but format is "dataframe"
# -> ValueError from type enforcement guard

# WRONG: subclass returns wrong number of evaluation sequences
def _evaluate_model_artifact(self, eval_type, artifact_name):
    return [df1, df2, df3]  # 3 instead of 13
# -> ValueError from _assert_predictions_in_step_window()
```

---

## 10. Test Alignment

- `tests/test_managers/test_model.py` -- unit tests for `ModelManager` and
  `ForecastingModelManager` initialization, config loading, arg validation.
- `tests/test_managers/test_model_manager_prediction_format.py` -- tests for
  ADR-042 type enforcement guards on both DF and PF code paths.
- `tests/test_managers/test_streaming_evaluation.py` -- tests for streaming
  PF evaluation via `_evaluate_model_artifact_streaming` and the origin sink
  pattern.
- `tests/test_managers/test_evaluation_stage.py` -- 17 tests for
  `EvaluationStage` (ADR-045 E2): frozen context, DF/PF paths, ensemble
  actuals, step mappings, multiple targets, context contract compliance.
- `tests/test_managers/test_execute_model_evaluation.py` -- 10 characterization
  tests for `_execute_model_evaluation()`: DF validation+save, type enforcement,
  PF streaming, skip-metrics, no-metrics, WandB lifecycle, sequence count.
- `tests/test_managers/test_reporting_stage.py` -- 11 tests for
  `ReportingStage` (ADR-045 E3): frozen context, forecast report (model +
  ensemble paths), evaluation report, missing data errors, WandB alerts,
  context contract compliance.
- `tests/test_managers/test_forecasting_stage.py` -- 10 tests for
  `ForecastingStage` (ADR-045 E4): DF/PF paths, type enforcement guards,
  CorePredictionSniffer validation, per-target save, log creation, alerts.
- `tests/test_managers/test_training_stage.py` -- 7 tests for
  `TrainingStage` (ADR-045 E5): frozen context, log creation, WandB alerts,
  sweep flag in alert text.

---

## 11. Evolution Notes

- The `_evaluate_model_artifact_streaming` method was added for memory-bounded
  evaluation of PredictionFrame models. Subclasses that override it emit one
  origin at a time without accumulating all origins in memory.
- `PredictionIOManager` was extracted from this class (commit `017c85a`) as
  part of SOLID E1 refactoring. All `_save_predictions`, `_save_evaluations`,
  and `_generate_evaluation_table` now delegate to `self._io`.
- The `prepare_actuals_df` hook was added for subclasses that manufacture
  derived targets (e.g., binary signals from raw counts).
- `ReportingStage` was extracted (ADR-045 E3) from
  `_execute_forecast_reporting()` and `_execute_evaluation_reporting()`.
  Both facade methods now construct a frozen `ReportingContext` and delegate.
  Dead code `_save_eval_report()` was removed (zero callers).
- `ForecastingStage` was extracted (ADR-045 E4) from
  `_execute_model_forecasting()`. The facade calls the abstract
  `_forecast_model_artifact()` (subclass-specific) then delegates
  type enforcement, validation, and persistence to the stage.
- `TrainingStage` was extracted (ADR-045 E5) from
  `_execute_model_training()`. The facade calls the abstract
  `_train_model_artifact()` then delegates log creation and alerts.
- `ModelPathManager` was relocated (ADR-045 E6) from
  `managers/model/model.py` to `data/model_path.py`. Re-export shim
  in `managers/model/model.py` maintains all existing import paths.
  This resolves Root Cause #1 (inverted dependencies).

---

## 12. Known Deviations

- **Facade file (~1960 LOC):** `ForecastingModelManager` and its parent
  `ModelManager` coexist in `model.py` (~1960 LOC after E1-E6 extractions,
  down from 3049). `ModelPathManager` has been relocated to `data/model_path.py`.
- **Target name regex assumption (R2):** The DF evaluation path assumes
  prediction columns are named `pred_{target}`. If a model uses a different
  naming convention, the column slice fails silently.
- **Mixed orchestration and evaluation logic:** `_evaluate_prediction_dataframe`
  contains target iteration, adapter construction, metric calculation, and
  result saving -- responsibilities that belong in separate components.
- **Ensemble actuals path (R3 in EnsembleManager):** When called from
  `EnsembleManager._evaluate_prediction_dataframe(ensemble=True)`, actuals
  are loaded from `models[0]`'s raw data, assuming all sub-models share the
  same actuals.

---

## End of Contract

This document defines the **intended meaning** of `ForecastingModelManager`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
