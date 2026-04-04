# Class Intent Contract: ReconciliationModule

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology), ADR-036 (Ensemble Reconciliation)

---

## 1. Purpose

`ReconciliationModule` performs hierarchical forecast reconciliation between country-level (`_CDataset`) and PRIO-grid-level (`_PGDataset`) predictions. It adjusts grid-level predictions so that they sum to country-level totals while preserving spatial patterns.

The reconciliation is parallelised across all combinations of (country, time step, target variable) using `concurrent.futures.ProcessPoolExecutor`.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** train models, run inference, or compute initial predictions.
- Does **not** aggregate predictions from multiple models (that is `AggregationManager`'s responsibility).
- Does **not** decide which models to reconcile or when to reconcile. The orchestrator makes those decisions.
- Does **not** persist results to disk. It returns the reconciled DataFrame; the caller is responsible for saving.
- Does **not** implement the reconciliation algorithm itself. It delegates to `ForecastReconciler` from `views_pipeline_core.modules.statistics`.
- Does **not** handle country-to-grid spatial mappings directly. It relies on `_PGDataset._build_country_to_grids_cache()`.

---

## 3. Responsibilities and Guarantees

- **Constructor validation**:
  - Type-checks that `c_dataset` is `_CDataset` and `pg_dataset` is `_PGDataset`. Raises `TypeError` on mismatch.
  - Validates that both datasets have the same number of time steps. Raises `ValueError` on mismatch.
  - Validates that both datasets use the same time unit (`_time_id`). Raises `ValueError` on mismatch.
  - Validates that both datasets cover the same time periods (symmetric difference must be empty). Raises `ValueError` on mismatch.
  - Identifies valid countries as the intersection of `pg_dataset._country_to_grids_cache.keys()` and `c_dataset._entity_values`.
  - Identifies valid targets as the intersection of `c_dataset.targets` and `pg_dataset.targets`. Raises `ValueError` if the intersection is empty.
  - Sends a WandB alert on successful initialisation, reporting the number of valid countries and time IDs.

- **`reconcile(lr, max_iters, tol, max_workers)`**:
  - Submits one task per `(country, time_id, target)` combination to a `ProcessPoolExecutor`.
  - Each task (`_reconcile_country_worker`) creates a subset of the country and grid datasets, extracts tensors via `to_reconciler()`, and calls `ForecastReconciler.reconcile_forecast()`.
  - Collects results and updates `pg_dataset.reconciled_dataframe` via `pg_dataset.reconcile()`.
  - Returns `pg_dataset.reconciled_dataframe`.
  - Logs progress every 10 completed countries.
  - Sends WandB alerts: on task failures (error level) and on overall completion (info level).

- **Device detection** (`__detect_torch_device`):
  - Selects the best available PyTorch device: CUDA > MPS > CPU.
  - Prints (not logs) the selected device.

---

## 4. Inputs and Assumptions

- **Constructor parameters**:
  - `c_dataset: _CDataset`: Country-level dataset with predictions. Must be in prediction mode.
  - `pg_dataset: _PGDataset`: Grid-level dataset with predictions. Must be in prediction mode and have a built `country_to_grids_cache`.
  - `wandb_notifications: bool`: Controls WandB alert delivery (default `True`).

- **`reconcile` parameters**:
  - `lr: float`: Learning rate (default `0.01`). Currently passed to `ForecastReconciler` but noted as unused.
  - `max_iters: int`: Maximum iterations (default `500`). Currently passed but noted as unused.
  - `tol: float`: Convergence tolerance (default `1e-6`). Currently passed but noted as unused.
  - `max_workers: Optional[int]`: Maximum parallel processes. If `None`, defaults to `min(32, os.cpu_count() + 4)`.

- Assumes both datasets have overlapping time periods, common targets, and compatible spatial hierarchies.
- Assumes `_PGDataset` has a `reconcile(country_id, time_id, reconciled_tensor, feature)` method for applying results.

---

## 5. Outputs and Side Effects

- **`reconcile()` return**: `pd.DataFrame` -- the reconciled grid-level predictions (`pg_dataset.reconciled_dataframe`).
- **Side effects**:
  - Mutates `pg_dataset.reconciled_dataframe` in-place for each successful task.
  - Builds `pg_dataset._country_to_grids_cache` during construction.
  - Prints device selection to stdout.
  - Sends WandB alerts at init, on task failures, and on completion.
  - Spawns multiple processes via `ProcessPoolExecutor`.

---

## 6. Failure Modes and Loudness

| Condition | Exception | Message pattern |
|---|---|---|
| `c_dataset` wrong type | `TypeError` | "Expected _CDataset, got {type}" |
| `pg_dataset` wrong type | `TypeError` | "Expected _PGDataset, got {type}" |
| Different number of time steps | `ValueError` | "The number of time steps ... must match" |
| Different time units | `ValueError` | "trying to reconcile datasets with different time units" |
| Non-overlapping time periods | `ValueError` | "The datasets have different time steps: {set}" |
| No common targets | `ValueError` | "No valid targets to reconcile found" |
| Individual task failure | Logged + WandB alert | "Task failed for country {id}, time {id}, feature {name}: {e}" |
| Bulk task failures | `logger.warning` | "{n} tasks failed during reconciliation" |

Individual task failures are logged and alerted but do **not** abort the overall reconciliation. The commented-out `RuntimeError` raise indicates this is a deliberate design choice, though it means partial results are returned without the caller being forced to handle failures.

---

## 7. Boundaries and Interactions

- **`_CDataset` / `_PGDataset`**: Provides the input data and spatial metadata. `to_reconciler()` extracts tensors; `reconcile()` applies results.
- **`ForecastReconciler`** (`views_pipeline_core.modules.statistics`): Performs the actual mathematical reconciliation via `reconcile_forecast()`. A new instance is created per worker task.
- **`WandBModule`**: Used via static method calls (`WandBModule.send_alert()`) for progress and error reporting.
- **`torch`**: Used for device detection and tensor computation within `ForecastReconciler`.
- **`concurrent.futures.ProcessPoolExecutor`**: Used for parallel task execution across CPU cores.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.data.handlers import CMDataset, PGMDataset
from views_pipeline_core.modules.reconciliation.reconciliation import ReconciliationModule

c_ds = CMDataset(country_predictions_df)
pg_ds = PGMDataset(grid_predictions_df)

reconciler = ReconciliationModule(c_ds, pg_ds, wandb_notifications=True)
reconciled_df = reconciler.reconcile(max_workers=16)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: passing datasets in wrong order
reconciler = ReconciliationModule(pg_ds, c_ds)  # raises TypeError

# WRONG: datasets with different time periods
reconciler = ReconciliationModule(c_ds_2020, pg_ds_2021)  # raises ValueError

# WRONG: datasets with no common targets
c_ds = CMDataset(df_with_pred_x)
pg_ds = PGMDataset(df_with_pred_y)
reconciler = ReconciliationModule(c_ds, pg_ds)  # raises ValueError

# WRONG: expecting reconcile() to detect partial failures automatically
result = reconciler.reconcile()
# If 50 tasks failed, result still returned without error -- caller must check logs
```

---

## 10. Test Alignment

There is **no dedicated test file** for `ReconciliationModule`. It is tested indirectly through statistics module tests and integration tests.

Key testing gaps:
- No unit tests for constructor validation (type checks, time step matching, target intersection).
- No unit tests for `_reconcile_country_worker` in isolation.
- No tests for partial failure handling (some tasks fail, others succeed).
- No tests for device detection across CPU/CUDA/MPS environments.

---

## 11. Evolution Notes

- The `lr`, `max_iters`, and `tol` parameters are passed through to `ForecastReconciler` but are documented as "currently unused" in the worker docstring. These may become active when the reconciliation algorithm is refined.
- The hardcoded `max_workers` formula (`min(32, os.cpu_count() + 4)`) follows the Python 3.8+ `ThreadPoolExecutor` default. `ProcessPoolExecutor` receives the computed `num_of_workers` value (fixed 2026-04-03, C-25).
- WandB alerts are sent via `WandBModule.send_alert()` (static method) rather than through an injected instance, making it harder to test without WandB being configured.

---

## 12. Known Deviations

- **No dedicated test file**: The class is only tested indirectly via statistics tests. This is a significant coverage gap for a class that orchestrates parallel computation across large datasets.
- **Device detection uses `print()` not `logger`**: The `__init__` method calls `print(f"Using device: {self._device}")` instead of `logger.info()`. This is inconsistent with the project's logging conventions.
- **~~`max_workers` parameter is ignored~~**: Fixed 2026-04-03 (C-25). `ProcessPoolExecutor` now receives `max_workers=num_of_workers`.
- **Partial failure is silent to the caller**: Failed tasks are logged and alerted via WandB, but `reconcile()` returns normally with partial results. The commented-out `RuntimeError` raise suggests this was considered but not implemented. Callers have no programmatic way to detect that some tasks failed without parsing logs.
- **Torch device may behave differently across environments**: CUDA availability, MPS availability, and CPU core count all vary. The class handles this gracefully via fallback, but the `print()` output will differ.
- **Log transform heuristic in `to_reconciler`**: The reconciliation worker relies on `_ViewsDataset.to_reconciler()` which detects `ln_` or `lx_` in feature names to decide whether to un-log. This is a naming-convention-based heuristic rather than an explicit declaration.

---

## End of Contract

This document defines the **intended meaning** of `ReconciliationModule`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
