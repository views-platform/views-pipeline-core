# Class Intent Contract: PredictionIOManager

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-08
**Related ADRs:** ADR-001 (Ontology), ADR-004 (Evolution), ADR-008 (Observability), ADR-048 (PredictionSaver Protocol)

---

## 1. Purpose

`PredictionIOManager` is the single-responsibility persistence layer for predictions and evaluations. It was extracted from `ForecastingModelManager` to separate I/O concerns from orchestration logic.

It handles three persistence targets:
1. **Local disk** (parquet files via `save_dataframe` or `pyarrow.parquet.write_table`).
2. **Prediction store** (views-forecasts store + Appwrite datastore).
3. **WandB** (evaluation metrics logging, artifact saving, and alert notifications).

The orchestration layer (`ForecastingModelManager`) owns WHAT to persist and WHEN. This class owns HOW.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** run inference, train models, or compute predictions.
- Does **not** decide when to save -- that is the orchestrator's responsibility.
- Does **not** validate prediction content (that is `CorePredictionSniffer`'s job).
- Does **not** convert between prediction formats (that is `PredictionFrameConverter`'s job).
- Does **not** manage WandB run lifecycle (init, finish). It only calls `log`, `save`, and `send_alert` on an injected `wandb_module`.

---

## 3. Responsibilities and Guarantees

- **`save_predictions(df_predictions, path_generated, run_type, timestamp, ...)`**:
  - Creates the output directory if it does not exist (`mkdir(parents=True, exist_ok=True)`).
  - Generates a filename via `PredictionFileNamer` using the run type, timestamp, optional sequence number, optional `target_identifier`, and `PipelineConfig.dataframe_format`. When `target_identifier` is provided, it is included in the filename to prevent collisions across targets in multi-target models (ADR-013 amendment 2026-05-06).
  - Dispatches to `pyarrow.parquet.write_table` for `pa.Table` inputs or `save_dataframe` for `pd.DataFrame` inputs.
  - Optionally uploads to the prediction store (if `use_prediction_store=True`).
  - Sends a WandB alert on success (unless `send_alert=False`).
  - Wraps all errors in `PipelineException`.

- **`save_evaluations(df_step, df_ts, df_month, path_generated, target_id, run_type, timestamp)`**:
  - Saves three evaluation DataFrames (step-wise, time-series-wise, month-wise) to disk.
  - Saves all three files as WandB artifacts via `wandb_module.save()`.
  - Logs all three as `wandb.Table` objects via `wandb_module.log()`.
  - Sends a WandB alert on success.
  - Wraps all errors in `PipelineException`.

- **`generate_evaluation_table(metric_dict)`** (static):
  - Formats a WandB summary metrics dict as a markdown table string using `tabulate`.
  - Filters out keys starting with `_` and non-numeric values.
  - Returns the table wrapped in markdown code fences.

---

## 4. Inputs and Assumptions

- **Constructor parameters**:
  - `model_path`: Object with `.model_name`, `.target`, `.root` attributes (typically `ModelPathManager`).
  - `wandb_module`: Object supporting `.send_alert()`, `.save()`, `.log()` methods.
  - `wandb_notifications: bool`: Controls whether WandB alerts are actually sent.
  - `use_prediction_store: bool`: Enables prediction store upload (default `False`).
  - `datastore: Optional`: Appwrite datastore client. Required if `use_prediction_store=True`.
  - `pred_store_name: Optional[str]`: Run name for the prediction store.

- **`save_predictions` parameters**:
  - `df_predictions`: Either `pd.DataFrame` or `pa.Table`.
  - `path_generated`: Directory path (created if absent).
  - `run_type`: String like `"forecasting"`, `"calibration"`.
  - `timestamp`: String for filename generation.
  - `level`, `targets`: Optional metadata for prediction store uploads.
  - `sequence_number`: Optional int for evaluation sequence numbering. `None` for forecasting runs.
  - `target_identifier`: Optional target name for multi-target models (e.g., `"ged_sb_best"`). When provided, included in filename to prevent collisions. Required for any path that saves per-target.
  - `send_alert`: Whether to fire a WandB alert (default `True`).

- Assumes `PipelineConfig.dataframe_format` is set (provides file extension).
- Assumes `wandb_module` is already initialized for the current run.

---

## 5. Outputs and Side Effects

- **Disk**: Creates parquet files in `path_generated` directory.
- **Prediction store**: Uploads via `df_predictions.forecasts.to_store()` (DataFrame path only; Arrow Tables raise `NotImplementedError`).
- **Appwrite**: Uploads via `datastore.upload_data()` if configured.
- **WandB**: Logs evaluation tables, saves artifact files, sends alert notifications.
- **Logging**: Uses `logger.info` for successful Appwrite uploads, `logger.error` for failures.

---

## 6. Failure Modes and Loudness

| Condition | Exception | Message pattern |
|---|---|---|
| Any error in `save_predictions` | `PipelineException` | "Error saving predictions: {e}" |
| Any error in `save_evaluations` | `PipelineException` | "Error saving model outputs: {e}" |
| Arrow Table upload to prediction store | `NotImplementedError` | "Prediction store upload is not yet supported for Arrow Tables" |
| Appwrite upload failure | Logged, not raised | `logger.error("Error uploading predictions to datastore: {e}")` |

`PipelineException` is used to wrap errors and optionally trigger WandB error alerts. Appwrite upload failure is the one exception to the "fail loud" rule -- it logs the error but does not abort the pipeline, as Appwrite is a secondary persistence target.

---

## 7. Boundaries and Interactions

- **Upstream**: Called by `ForecastingModelManager` (the orchestrator) and `ForecastingStage` after predictions or evaluations are computed.
- **`PredictionFileNamer`**: Generates canonical filenames for predictions and evaluations. Extracted from this class in Phase 6 Task 1.
- **`PredictionFrameConverter`**: Produces the `pd.DataFrame` or `pa.Table` that this manager persists.
- **`PipelineConfig`**: Provides `dataframe_format` for filename generation.
- **`save_dataframe`**: Utility function from `files/utils.py` that handles format-specific DataFrame serialisation.
- **WandB**: External dependency injected as `wandb_module`. The `wandb` package is also imported directly inside `save_evaluations` for `wandb.Table`.

### PredictionSaver Protocol (Phase 6, ADR-048)

Phase 6 extracted format-specific persistence into composable saver classes implementing the `PredictionSaver` Protocol. These savers live alongside `PredictionIOManager` in `managers/prediction/savers.py` and accept `PredictionFrame` directly (not DataFrame/Arrow).

| Saver | Track | Format | Failure mode |
|-------|-------|--------|-------------|
| `NpzSaver` | A (internal) | `.npy` + `.npz` (numpy binary) | Raises on I/O error |
| `LocalParquetSaver` | B (delivery) | Parquet via Arrow zero-copy | Raises on I/O error |
| `AppwriteSaver` | Cloud | Delegates to `DatastoreModule.upload_data()` | Graceful: logs error, does not raise |
| `ViewsForecastsSaver` | Store | Converts PF→DF, calls `df.forecasts.to_store()` | Raises (primary external store) |

Task 5 will compose these savers into `save_predictions()`, replacing the current inline persistence logic.

---

## 8. Examples of Correct Usage

```python
io = PredictionIOManager(
    model_path=model_path,
    wandb_module=wandb_module,
    wandb_notifications=True,
    use_prediction_store=False,
)

# Save predictions (DataFrame)
io.save_predictions(
    df_predictions=df,
    path_generated=output_dir,
    run_type="forecasting",
    timestamp="20260401_120000",
)

# Save predictions (Arrow Table, no store upload)
io.save_predictions(
    df_predictions=arrow_table,
    path_generated=output_dir,
    run_type="calibration",
    timestamp="20260401_120000",
    sequence_number=3,
)

# Save evaluations
io.save_evaluations(
    df_step_wise_evaluation=step_df,
    df_time_series_wise_evaluation=ts_df,
    df_month_wise_evaluation=month_df,
    path_generated=eval_dir,
    target_identifier="ln_sb_best",
    run_type="calibration",
    timestamp="20260401_120000",
)

# Format metrics for display
table_str = PredictionIOManager.generate_evaluation_table(wandb.run.summary)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: enabling prediction store with Arrow Table input
io = PredictionIOManager(..., use_prediction_store=True)
io.save_predictions(arrow_table, ...)  # raises NotImplementedError

# WRONG: calling save_predictions without initializing wandb_module
io = PredictionIOManager(model_path=mp, wandb_module=None, ...)
io.save_predictions(df, ...)  # will fail on wandb_module.send_alert()

# WRONG: expecting save_evaluations to compute metrics (it only persists them)
io.save_evaluations(raw_predictions, ...)  # expects pre-computed metric DataFrames
```

---

## 10. Test Alignment

Tests live in `tests/test_managers/test_prediction_io.py`.

| Test area | Covers |
|---|---|
| `io_manager` fixture | Basic construction without prediction store |
| `io_manager_with_store` fixture | Construction with prediction store enabled |
| `save_predictions` tests | DataFrame saving, directory creation, filename generation, WandB alert |
| `save_evaluations` tests | Three-file save, WandB table logging, WandB artifact save |
| `generate_evaluation_table` | Markdown table formatting, key filtering |

Key gap: No integration test for the Arrow Table (`pa.Table`) path through `save_predictions`. The `NotImplementedError` for prediction store + Arrow is tested implicitly.

---

## 11. Evolution Notes

- Extracted from `ForecastingModelManager` in commit `017c85a` to achieve single-responsibility I/O.
- The Arrow Table path (`pa.Table` dispatch in `save_predictions`) was added alongside `PredictionFrameConverter.to_arrow_table` for the zero-copy parquet write path.
- Prediction store upload for Arrow Tables is explicitly blocked with `NotImplementedError` pending upstream support in `views-forecasts`. The new `ViewsForecastsSaver` (Phase 6 Task 4) resolves this by converting `PredictionFrame` → `pd.DataFrame` internally.
- **Phase 6 extractions (2026-04-07):**
  - Task 1: `PredictionFileNamer` extracted for filename generation (SRP).
  - Task 2: `NpzSaver` and `LocalParquetSaver` created as `PredictionSaver` Protocol implementations.
  - Task 3: `PredictionStoreConfig` created for fail-loud env var validation (fixes C-11).
  - Task 4: `AppwriteSaver` and `ViewsForecastsSaver` created as cloud/store savers.
  - Task 5 (pending): Compose savers into `save_predictions()`, replacing inline persistence.

---

## 12. Known Deviations

- **Tight coupling to ForecastingModelManager internals**: The constructor receives `model_path` (a `ModelPathManager`), `wandb_module`, and several configuration flags as separate parameters rather than a structured command/config object. This makes the interface wide and fragile to orchestrator changes.
- **Direct `wandb` import in `save_evaluations`**: The `wandb` package is imported inside the method body (`import wandb`) rather than being fully injected, creating a hard dependency that complicates testing.
- **Appwrite failure is silent**: `_upload_to_prediction_store` catches Appwrite errors with `logger.error` but does not re-raise. This deviates from the project's "Fail Loud and Proud" principle but is intentional -- Appwrite is a secondary target and should not block the pipeline.

---

## End of Contract

This document defines the **intended meaning** of `PredictionIOManager`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
