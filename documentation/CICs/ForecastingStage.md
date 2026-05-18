
# Class Intent Contract: ForecastingStage

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-04-08
**Related ADRs:** ADR-042 (Type Enforcement), ADR-045 (Stage Pattern, E4)

---

## 1. Purpose

Handles forecast post-processing: receives raw predictions from the model artifact
method, enforces type contracts (ADR-042), validates DataFrame predictions via
`CorePredictionSniffer`, converts `PredictionFrame` outputs via
`PredictionFrameConverter`, persists results via `PredictionIOManager`, creates
execution logs, and sends WandB completion alerts. It is the ADR-045 Stage pattern
implementation for the forecasting path.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** run model inference. It receives predictions, not models.
- This class does **not** manage WandB run lifecycle (init/finish). The facade owns
  that.
- This class does **not** inherit from or access `ForecastingModelManager`.
- This class does **not** load data or define partitions.
- This class does **not** compute evaluation metrics. That is `EvaluationStage`'s
  responsibility.

---

## 3. Responsibilities and Guarantees

- Guarantees **ADR-042 type enforcement**: if `prediction_format="prediction_frame"`
  and the predictions are not a `dict`, raises `ValueError`. Conversely, if
  `prediction_format="dataframe"` and the predictions are a `dict`, raises
  `ValueError`. Mismatched return types are never silently coerced.
- Guarantees that on the DF path, `CorePredictionSniffer(level=configs["level"]).sniff_predictions()`
  is called before persistence, validating MultiIndex structure and prediction columns.
- Guarantees that on the PF path with savers (Phase 6 Task 5), each target's
  `PredictionFrame` is passed directly to every saver in the composed chain via
  `_save_via_savers()`. Each target's `PredictionMetadata.filename` includes the
  target name via `target_identifier`, producing a unique filename per target to
  prevent file collisions in multi-target models (ADR-013 amendment 2026-05-06).
  No DataFrame conversion occurs for local persistence.
- Guarantees that on the PF path without savers (legacy fallback),
  `PredictionFrameConverter.to_prediction_df()` and `audit_prediction_structure()`
  are called for each target before persistence via `_save_via_io_manager()`.
  Each call to `save_predictions()` passes `target_identifier` to produce a
  unique filename per target.
- Guarantees that on the DF path, predictions are saved via
  `PredictionIOManager.save_predictions()` with `run_type`, `timestamp`, `level`,
  and `targets` metadata.
- Guarantees that an execution log entry is created via `handle_single_log_creation()`
  after successful processing.
- Guarantees that a WandB completion alert is sent after processing.

---

## 4. Inputs and Assumptions

- `wandb_module` -- `WandBModule` instance for alerts (no lifecycle management).
- `io_manager` -- `PredictionIOManager` instance for prediction persistence (DF path + legacy PF fallback).
- `wandb_notifications: bool` -- gate for WandB alerts.
- `savers: Optional[List[PredictionSaver]]` -- composed saver chain for PF path (Phase 6 Task 5). When provided, the PF path bypasses `io_manager` and delegates to savers directly. Constructed by `ForecastingModelManager.__init__()` with `[LocalParquetSaver]` + conditionally `ViewsForecastsSaver` + `AppwriteSaver`.
- `process_and_save_forecast()` arguments:
  - `predictions` -- one of:
    - `pd.DataFrame` (DF path): with `pred_{target}` columns.
    - `Dict[str, PredictionFrame]` (PF path): keyed by target name.
  - `context: ForecastingContext` (frozen dataclass):
    - `configs: Dict` -- must contain `name`, `level`, `targets`, and optionally
      `timestamp`.
    - `model_path: ModelPathManager` -- for `target`, `model_name`, `data_generated`.
    - `run_type: str` -- the run type string.
    - `prediction_format: str` -- `"dataframe"` or `"prediction_frame"`.

---

## 5. Outputs and Side Effects

- **Primary output:** `None`. Results are persisted as side effects.
- **Side effects:**
  - Saves prediction DataFrames to `data_generated/` via `PredictionIOManager`.
  - Creates an execution log file via `handle_single_log_creation()`.
  - Sends a WandB alert on completion.
  - Logs progress at `INFO` level.

---

## 6. Failure Modes and Loudness

- `ValueError` (fail-loud) if prediction type does not match declared
  `prediction_format`. This is the ADR-042 type enforcement guard.
  - PF declared but predictions are not `dict` -> `ValueError`.
  - DF declared but predictions are `dict` -> `ValueError`.
- `CorePredictionSniffer` raises on MultiIndex or prediction column violations
  (DF path only).
- `PredictionFrameConverter.audit_prediction_structure()` raises on structural
  mismatches (PF path only).
- `PredictionIOManager.save_predictions()` propagates I/O errors.
- WandB alert failures are handled by `WandBModule.send_alert()` (logged, not
  raised).

---

## 7. Boundaries and Interactions

- **Depends on:**
  - `CorePredictionSniffer` -- structural validation (DF path).
  - `PredictionFrameConverter` -- PF-to-DF conversion and audit (PF path).
  - `PredictionIOManager` -- prediction persistence.
  - `WandBModule` -- completion alerts.
  - `handle_single_log_creation()` -- execution log creation.
  - `ModelPathManager` -- path resolution for `data_generated` and model metadata.
- **Does not depend on:**
  - `ForecastingModelManager` or any model manager.
  - `CoreDataSniffer`, `CoreConfigSniffer`.
  - `EvaluationStage`.
  - Data loaders, Appwrite, or VIEWSER.
- **Injected collaborators:** `wandb_module` and `io_manager` are injected at
  construction.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.forecasting.stage import ForecastingStage, ForecastingContext

stage = ForecastingStage(
    wandb_module=wandb_mod,
    io_manager=io_mgr,
    wandb_notifications=True,
)

context = ForecastingContext(
    configs=configs,
    model_path=model_path,
    run_type="forecasting",
    prediction_format="dataframe",
)

# DF path: predictions is a pd.DataFrame
stage.process_and_save_forecast(predictions=df_predictions, context=context)
```

```python
# PF path: predictions is Dict[str, PredictionFrame]
context_pf = ForecastingContext(
    configs=configs,
    model_path=model_path,
    run_type="forecasting",
    prediction_format="prediction_frame",
)
stage.process_and_save_forecast(predictions=pf_dict, context=context_pf)
```

---

## 9. Examples of Incorrect Usage

- **Declaring `prediction_format="dataframe"` but returning a dict from the model:**
  This triggers the ADR-042 type enforcement `ValueError`. The model contract must
  match the declared format.
- **Omitting `level` from `configs`:** The DF path accesses `configs["level"]`
  to construct `CorePredictionSniffer`. A missing key will raise `KeyError`.
- **Omitting `targets` from `configs`:** The DF path passes `configs["targets"]`
  to `sniff_predictions()`. A missing key will raise `KeyError`.
- **Mutating `context` after construction:** `ForecastingContext` is a frozen
  dataclass. Attempting to set fields will raise `FrozenInstanceError`.

---

## 10. Test Alignment

- **Green tests:** Unit tests with mocked `CorePredictionSniffer`,
  `PredictionFrameConverter`, and `PredictionIOManager` can verify type enforcement
  guards, correct delegation to DF vs PF paths, and log/alert creation.
- **Red tests:** Tests must verify that `ValueError` is raised for type mismatches
  in both directions (dict when DF expected, non-dict when PF expected). Tests
  should also verify that `CorePredictionSniffer` is called before
  `save_predictions()` on the DF path.

---

## 11. Evolution Notes (Optional)

- The PF path iterates over `predictions.items()` and saves each target separately.
  A batched save may improve performance for models with many targets.
- `handle_single_log_creation()` is called with `train=False` unconditionally.
  This is correct for forecasting but the parameter name is misleading.

---

## End of Contract

This document defines the **intended meaning** of `ForecastingStage`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
