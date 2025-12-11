# VIEWS Pipeline Core: Hierarchical Forecast Reconciliation Module

Module path: `views_pipeline_core/modules/reconciliation/reconciliation.py`  
Primary class: `ReconciliationModule`  
Depends on:  
- Country dataset interface (`_CDataset`)  
- Priogrid dataset interface (`_PGDataset`)  
- `ForecastReconciler` (statistical proportional reconciliation)  
- `WandBModule` (alerting)  
- PyTorch (device management, tensor ops)  
- `tqdm` (progress reporting)  
- `concurrent.futures` (parallel processing)

## Purpose

Performs hierarchical reconciliation so that priogrid-level (PGM) predictions aggregate exactly (or within tight tolerance) to country-level (CM) totals for the same target(s), time steps, and countries. Preserves:
- Spatial pattern (relative proportions among non-zero grids).
- Zero inflation (grid cells that are zero remain zero).
- Target variable integrity (only common targets reconciled).

This is a post-processing integrity step aligned with ADR-027 (ensemble & reconciliation consistency) and supports downstream evaluation, reporting, and publication.

## Class: ReconciliationModule

### Initialization

```python
reconciler = ReconciliationModule(
    c_dataset=country_dataset,
    pg_dataset=grid_dataset,
    wandb_notifications=True
)
```

Args:
- `c_dataset` (`_CDataset`): Country-level dataset with prediction columns.
- `pg_dataset` (`_PGDataset`): Priogrid-level dataset with matching prediction columns.
- `wandb_notifications` (bool): Whether to emit WandB alerts.

Validations performed:
1. Type checks (`_CDataset`, `_PGDataset`).
2. Same number of time steps.
3. Same time index variable (e.g. `month_id`).
4. Exact overlap in time values (no symmetric difference).
5. Intersection of target names non-empty.
6. Country IDs present in both: build list of valid reconciliation countries.
7. Build country → grids cache (accelerates subset extraction).

Side effects:
- GPU / MPS / CPU device auto-detection.
- WandB alert summarizing reconciliation scope.

Raises:
- `TypeError` (wrong dataset type).
- `ValueError` (structural mismatch: time dimension, time units, time overlap, no common targets).

### Device Detection

`__detect_torch_device()` selects:
- `cuda` (NVIDIA GPU)
- `mps` (Apple Silicon GPU)
- `cpu` fallback

Used for potential future GPU-enabled reconciliation kernels.

### Parallel Worker

`_reconcile_country_worker(args)` (staticmethod):

Input tuple:
```
(country_id, time_id, feature, lr, max_iters, tol,
 c_subset_dataframe, pg_subset_dataframe, device_str)
```

Process:
1. Recreate lightweight `_CDataset` / `_PGDataset` wrappers from passed subsets.
2. Convert to tensors via each dataset’s `.to_reconciler(...)` helper.
3. Call `ForecastReconciler.reconcile_forecast(...)`.
4. Return `(country_id, time_id, feature, reconciled_tensor.cpu())`.

Notes:
- A fresh `ForecastReconciler` instance per task avoids shared state issues.
- `lr`, `max_iters`, `tol` placeholders are currently passed through (proportional reconciliation does not iterate).

### Main Method: reconcile()

```python
reconciled_df = reconciler.reconcile(
    lr=0.01,
    max_iters=500,
    tol=1e-6,
    max_workers=None
)
```

Args (currently informational):
- `lr`, `max_iters`, `tol`: Reserved for potential iterative reconciliation (e.g., optimization-based methods).
- `max_workers`: If `None`, uses `min(32, os.cpu_count() + 4)` heuristic; (current code mistakenly sets `max_workers=None` in executor—adjust if customizing).

Workflow:
1. Compute total task count = countries × time_ids × targets.
2. Submit every (country, time, feature) combination to a `ProcessPoolExecutor`.
3. Track per-country task completion; send WandB alert every 10 countries finished.
4. Collect results (tensor outputs) and log failures.
5. Update grid dataset via `_pg_dataset.reconcile(...)` for each successful result.
6. Emit final WandB success alert.

Returns:
- Updated reconciled priogrid-level dataframe (`_pg_dataset.reconciled_dataframe`).

Error handling:
- Failed tasks logged + WandB alert with `ERROR` level.
- Aggregate failures warned (module does not raise unless customized—extend if strict failure required).

Performance considerations:
- Process-level parallelism (copying subsets) can incur overhead for extremely large datasets; consider batching or shared memory if scaling further.
- Device usage is currently per-worker CPU (tensors moved to CPU before return). GPU reconciliation logic could be centralized in future.

### Data Flow Summary

```
Country Dataset (CM) ─┐
                      │ intersection on (targets, time_ids, country_ids)
Priogrid Dataset (PGM) ┤
                      │
                      ├─> Task tuples (country, time, target, subsets) → ProcessPool
                      │
                      ├─> Worker: subset → tensors → ForecastReconciler → reconciled tensor
                      │
                      └─> Aggregation: write reconciled values into pg_dataset.reconciled_dataframe
```

### Integration Points

| Component              | Interaction                                      |
|------------------------|--------------------------------------------------|
| `ForecastReconciler`   | Performs proportional scaling                    |
| `WandBModule`          | Alerts for start, per-10-country progress, errors, completion |
| Managers / Pipelines   | Called post-model prediction stage               |
| Reporting Module       | Consumes reconciled outputs for HTML summaries   |
| Evaluation Module      | Uses reconciled priogrid predictions for metrics |
| ADR-027                | Alignment with documented reconciliation strategy |

### Typical Usage Example

```python
from views_pipeline_core.data.handlers import CMDataset, PGMDataset
from views_pipeline_core.modules.reconciliation import ReconciliationModule

# Assume cm_df, pg_df loaded with prediction columns (e.g. pred_ln_ged_sb)
c_ds = CMDataset(source=cm_df)
pg_ds = PGMDataset(source=pg_df)

recon = ReconciliationModule(c_dataset=c_ds, pg_dataset=pg_ds)
reconciled_pg = recon.reconcile(max_workers=16)

# Save reconciled priogrid predictions
reconciled_pg.to_parquet("reconciled_priogrid_predictions.parquet")
```

### Best Practices

1. Run reconciliation only after final prediction generation (do not re-train after).
2. Undo log transforms (if required for aggregation logic) inside dataset handlers before evaluation.
3. Monitor WandB alerts for failed task counts—investigate systemic issues early.
4. Persist both original and reconciled priogrid predictions for audit trail.
5. Avoid mixing time units (ensure both datasets use identical `month_id` or `year_id`).
6. Pre-filter countries if known to have incomplete priogrid coverage to reduce failures.
7. If scaling > millions of tasks, consider switching to thread-based execution with shared memory or PyTorch distributed.

### Troubleshooting

| Symptom | Cause | Action |
|---------|-------|--------|
| ValueError: time steps mismatch | Divergent temporal coverage | Align data extraction queries |
| No valid targets | Different target naming conventions | Standardize prediction column names across models |
| Many failed tasks | Data subset slicing or tensor conversion errors | Inspect logs; verify `.to_reconciler()` output |
| Slow completion | Too many workers contending | Set `max_workers` lower; ensure I/O not bottlenecked |
| High memory usage | Large DataFrames copied per task | Introduce chunking or shared memory mapping |

### Extending

To support iterative or optimization-based reconciliation (e.g., constrained least squares):
- Implement iterative logic inside `ForecastReconciler.reconcile_forecast`.
- Use `lr`, `max_iters`, `tol` arguments meaningfully.
- Return convergence diagnostics alongside reconciled tensor (extend tuple).

### Logging & Alerts

Severity levels:
- INFO: Start, submission, progress.
- WARNING: Task failures count.
- ERROR: Individual task exception details (forwarded via WandB alert).
- SUCCESS (implicit): Completion alert.

Configure logging via global pipeline logging configuration (`logging.yaml` per ADR-025).

### Security / Robustness Notes

- No external network calls beyond WandB alert emission.
- Process isolation limits side effects.
- Dataset mutation confined to `reconciled_dataframe` property (original predictions remain accessible).

### Related ADRs

- ADR-027 Ensemble & Reconciliation
- ADR-012/013 Artifact & Output naming (reconciled outputs should follow established patterns)
- ADR-020 Logging & Real-time alerts (WandB integration strategy)

### FAQ

| Question | Answer |
|----------|--------|
| Does reconciliation change country totals? | No, country totals are treated as authoritative. |
| Are zero priogrid cells altered? | They remain zero (proportional scaling applies only to non-zero cells). |
| Can it run on GPU? | Device detection supports GPU, but current worker design returns CPU tensors. |
| What if targets differ in transformation state? | Only exact name intersections are reconciled; ensure consistent naming upstream. |
| How to enforce strict failure? | After reconcile, raise if `failed_tasks` not empty. |

---
