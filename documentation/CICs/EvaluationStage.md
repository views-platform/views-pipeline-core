
# Class Intent Contract: EvaluationStage

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-04-08
**Related ADRs:** ADR-040 (Authority over Inference), ADR-045 (Stage Pattern, E2)

---

## 1. Purpose

Orchestrates model evaluation: loads actuals from raw VIEWSER data, builds
`EvaluationFrame` objects via `EvaluationAdapter` (supporting both the DataFrame and
PredictionFrame paths), computes metrics via `NativeEvaluator`, and publishes results
to WandB and disk. It is the first implementation of the ADR-045 Stage pattern,
receiving an explicit, frozen `EvaluationContext` rather than reaching into a parent
class.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** generate predictions. It receives them as input.
- This class does **not** manage WandB run lifecycle (init/finish). The facade owns
  that.
- This class does **not** inherit from or access `ForecastingModelManager`. It is
  fully decoupled via the `EvaluationContext` dataclass.
- This class does **not** define partition boundaries or fetch data from VIEWSER.
  Actuals are loaded from cached raw data on disk.
- This class does **not** validate predictions structurally. That is
  `CorePredictionSniffer`'s responsibility in the forecasting path.

---

## 3. Responsibilities and Guarantees

- Guarantees that actuals are loaded from the correct raw data file path, determined
  by `context.run_type` and `context.model_path`.
- Guarantees that actuals are prepared via `context.prepare_actuals_df()` before use.
- Guarantees that evaluation frames are built via `EvaluationAdapter.from_prediction_frames()`
  (PF path) or `EvaluationAdapter.from_dataframes()` (DF path) depending on
  `context.prediction_format`.
- Guarantees that `NativeEvaluator.evaluate()` is called with `legacy_compatibility=True`
  to preserve step-wise truncation behavior matching the deleted `EvaluationManager`
  wrapper (C-29).
- Guarantees that step mappings are built by the stage itself via
  `_get_evaluation_step_mappings()`, fulfilling ADR-040 (orchestrator is sole authority
  on lead-times).
- Guarantees that evaluation DataFrames are saved to disk (via `PredictionIOManager`)
  unless `configs["sweep"]` is `True`.
- Guarantees that a WandB summary alert is sent after all targets are processed.
- Guarantees aggressive memory management: `EvaluationFrame` and raw prediction
  objects are deleted and `gc.collect()` is called after each target.

---

## 4. Inputs and Assumptions

- `wandb_module` -- `WandBModule` instance for metrics logging and alerts.
- `io_manager` -- `PredictionIOManager` instance for saving evaluation DataFrames.
- `wandb_notifications: bool` -- gate for WandB alerts.
- `evaluate()` arguments:
  - `df_predictions` -- one of:
    - `List[pd.DataFrame]` (DF path): each DataFrame has `pred_{target}` columns.
    - `Dict[str, List[PredictionFrame]]` (PF path): keyed by target name.
  - `context: EvaluationContext` (frozen dataclass):
    - `configs: Dict` -- must contain `regression_targets`, `classification_targets`,
      `steps`, and optionally `sweep`, `run_type`, `timestamp`.
    - `model_path: ModelPathManager` -- for raw data file paths and `model_name`.
    - `run_type: str` -- `"calibration"`, `"validation"`, or `"forecasting"`.
    - `prediction_format: str` -- `"dataframe"` or `"prediction_frame"`.
    - `partition_dict: Dict` -- partition time ranges.
    - `data_loader: Any` -- `ViewsDataLoader` or `None`; required for forecasting
      to determine `month_last`.
    - `prepare_actuals_df: Callable` -- transforms raw VIEWSER DataFrame to actuals.
  - `ensemble: bool` -- if `True`, loads actuals from `configs["models"][0]`.

---

## 5. Outputs and Side Effects

- **Primary output:** `None`. Results are published as side effects.
- **Side effects:**
  - Logs evaluation metrics to WandB via `log_evaluation_results()`.
  - Saves step-wise, time-series-wise, and month-wise evaluation DataFrames to
    `data_generated/` via `PredictionIOManager.save_evaluations()`.
  - Sends a WandB summary alert with the evaluation table.
  - Reads raw VIEWSER data from disk.
  - Triggers garbage collection after each target evaluation.

---

## 6. Failure Modes and Loudness

- `ValueError` from `_get_evaluation_step_mappings()` if `run_type="forecasting"`
  and `context.data_loader` is `None`.
- `KeyError` from `_get_evaluation_step_mappings()` if `run_type` is not found in
  `context.partition_dict`.
- Missing targets in the PF path (`df_predictions.pop(target)` returns `None`) are
  logged as warnings and skipped; they do **not** abort the evaluation loop.
- Missing `pred_{target}` columns in the DF path are logged as warnings and skipped.
- If no targets are defined (`regression_targets` and `classification_targets` both
  empty), `_load_actuals()` returns `None` and `evaluate()` returns early.

---

## 7. Boundaries and Interactions

- **Depends on:**
  - `views_evaluation.NativeEvaluator` -- metric computation.
  - `views_pipeline_core.modules.validation.adapter.EvaluationAdapter` -- EF construction.
  - `views_pipeline_core.files.utils.read_dataframe` -- loading raw actuals.
  - `PredictionIOManager` -- saving evaluation results.
  - `WandBModule` -- metrics logging and alerts.
  - `ModelPathManager` -- raw data file paths.
- **Does not depend on:**
  - `ForecastingModelManager` or any model manager.
  - `CoreDataSniffer`, `CorePredictionSniffer`, or `CoreConfigSniffer`.
  - Appwrite or any external storage.
- **Injected collaborators:** `wandb_module` and `io_manager` are injected at
  construction, not imported at class level.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.evaluation.stage import EvaluationStage, EvaluationContext

stage = EvaluationStage(
    wandb_module=wandb_mod,
    io_manager=io_mgr,
    wandb_notifications=True,
)

context = EvaluationContext(
    configs=configs,
    model_path=model_path,
    run_type="calibration",
    prediction_format="dataframe",
    partition_dict=partition_dict,
    data_loader=None,
    prepare_actuals_df=my_prepare_fn,
)

stage.evaluate(
    df_predictions=[df_pred_seq1, df_pred_seq2],
    context=context,
)
```

---

## 9. Examples of Incorrect Usage

- **Passing a `ViewsDataLoader` as `data_loader` for calibration/validation:**
  The `data_loader` field is only used for forecasting runs (to determine
  `month_last`). Passing it for other run types is harmless but misleading.
- **Mutating `context` after construction:** `EvaluationContext` is a frozen
  dataclass. Attempting to set fields will raise `FrozenInstanceError`.
- **Passing `Dict[str, PredictionFrame]` when `prediction_format="dataframe"`:**
  The DF path expects `List[pd.DataFrame]`. A dict will fail at column lookup.

---

## 10. Test Alignment

- **Green tests:** Unit tests with mocked `NativeEvaluator`, `EvaluationAdapter`,
  and `PredictionIOManager` can verify the orchestration flow, step mapping
  computation, and correct delegation for both DF and PF paths.
- **Red tests:** Tests should verify that `ValueError` is raised when `data_loader`
  is `None` for forecasting, that `KeyError` is raised for unknown run types in the
  partition dict, and that missing targets produce warnings (not crashes).

---

## 11. Evolution Notes (Optional)

- `legacy_compatibility=True` is passed to `NativeEvaluator.evaluate()` for backward
  compatibility with the deleted `EvaluationManager` (C-29). This flag may be removed
  once all models have been retrained and re-evaluated.
- The PF path uses `df_predictions.pop(target)`, which mutates the input dict. A
  future revision may copy the dict to avoid side effects on the caller.

---

## End of Contract

This document defines the **intended meaning** of `EvaluationStage`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
