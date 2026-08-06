# Ensemble Management

File: `views_pipeline_core/managers/ensemble/ensemble.py`  
Classes:
- `EnsemblePathManager`
- `EnsembleManager`

Provides orchestration for multi–model ensembles: training each constituent model, evaluating and aggregating their predictions, producing ensemble forecasts, and optionally performing hierarchical reconciliation (PGM ↔ CM).

---

## EnsemblePathManager

### Purpose
Specialized path manager for ensembles. Extends `ModelPathManager` but switches target namespace to `ensembles/`. Ensures consistent directory/script resolution for ensemble assets (artifacts, generated data, reports, logs).

### Key Differences vs ModelPathManager
- `_target = "ensemble"`
- Base directory: `<root>/ensembles/<ensemble_name>/`
- Inherits all path + artifact helpers (latest artifact discovery, queryset import if present).

### Constructor
```python
pm = EnsemblePathManager("blended_dragon")
```

### Class Method
```python
@classmethod
def _initialize_class_paths(cls, current_path: Path = None)
```
Initializes project root and sets `cls._models = <root>/ensembles`.

### Usage
```python
epm = EnsemblePathManager("hybrid_lynx")
print(epm.data_generated)   # Path to generated ensemble outputs
```

---

## EnsembleManager

### Purpose
Extends `ForecastingModelManager` to:
1. Loop through constituent models listed in ensemble config (`configs["models"]`).
2. Delegate train/evaluate/forecast to each model via shell execution (re‑invoking their pipeline).
3. Aggregate predictions across models (mean or median).
4. Optionally reconcile aggregated priogrid forecasts to country totals.
5. Emit WandB alerts on stage completion or failure.

### Lifecycle Methods
| Method | Action |
|--------|--------|
| `execute_single_run(args)` | Entry point: updates config, validates (if not training), runs selected stages. |
| `_execute_model_training()` | Iterates models → `_train_model_artifact`. |
| `_execute_model_evaluation()` | Collects per‑model evaluation predictions, aggregates per sequence, computes metrics. |
| `_execute_model_forecasting()` | Collects per‑model forecast predictions, aggregates, applies reconciliation if configured. |
| `_execute_model_tasks()` | Dispatches stage combination based on CLI flags. |

### Ensemble Orchestration
| Method | Description |
|--------|-------------|
| `_train_ensemble()` | Progress bar over `configs["models"]`. |
| `_evaluate_ensemble()` | Returns list of aggregated evaluation DataFrames (one per evaluation sequence). |
| `_forecast_ensemble()` | Returns a single aggregated forecast DataFrame. |

### Model Artifact Delegation
Each constituent model is driven by a shell command derived from its own `ForecastingModelArgs`:
- `_train_model_artifact(model_name)`
- `_evaluate_model_artifact(model_name)` → returns list of DataFrames (sequences).
- `_forecast_model_artifact(model_name)` → returns forecast DataFrame.

Internally uses:
```python
self._execute_shell_script(model_path, model_name, model_args)
```

### Prediction Loading Logic
If a prediction file matching pattern exists locally (`predictions_{run_type}_{timestamp}_{seq}.parquet`) it is reused; otherwise the model is re-run to generate it. Optional prediction store integration (if enabled) supersedes local file lookup.

### Aggregation
```python
df_agg = EnsembleManager._get_aggregated_df(list_of_dfs, aggregation="mean"|"median")
```
Rules:
- Converts single‑element lists in cells to scalars.
- Rejects multi‑value lists (no distribution aggregation).
- Groups by MultiIndex (`month_id`, `entity_id`).

### Reconciliation (Optional)
Triggered when:
```python
configs["reconciliation"] == "pgm_cm_point"
```
Steps:
1. Load latest country predictions (from prediction store or local path of `reconcile_with` model).
2. Wrap priogrid + country DataFrames in `_PGDataset` / `_CDataset`.
3. Call `ReconciliationModule.reconcile(...)`.
4. Replace priogrid predictions with reconciled output.

Alerts sent via WandB on success/failure.

### Key Config Fields
| Key | Meaning |
|-----|---------|
| `models` | List of model names (must have valid directories). |
| `aggregation` | `"mean"` or `"median"`. |
| `reconciliation` | `"pgm_cm_point"` to activate. |
| `reconcile_with` | Country-level model name supplying totals. |

### Error Handling
Errors wrapped into `PipelineException` with traceback and WandB alert:
- Shell execution failure
- Missing artifact paths
- Invalid aggregation type
- Reconciliation load failures

### Helper Methods
| Method | Purpose |
|--------|---------|
| `_create_model_args(train, evaluate, forecast)` | Build constituent model CLI args with inherited flags (e.g. override timestep). |
| `_load_or_generate_prediction(...)` | Reuse existing prediction or trigger model run. |
| `_apply_reconciliation(df)` | Guard + dispatch reconciliation based on config. |
| `__reconcile_pg_with_c(pg_dataframe, c_dataframe)` | Internal reconciliation workflow. |
| `_load_c_dataset(cm_model, c_dataframe)` | Attempts prediction store → local file → provided DataFrame. |

### Minimal Example
```python
from views_pipeline_core.managers.ensemble.ensemble import (
    EnsembleManager, EnsemblePathManager
)
from views_pipeline_core.cli.args import ForecastingModelArgs

epm = EnsemblePathManager("fusion_orca")
manager = EnsembleManager(epm, wandb_notifications=True)

args = ForecastingModelArgs(
    run_type="forecasting",
    train=True,
    evaluate=True,
    forecast=True,
    report=True,
    eval_type="standard"
)

manager.execute_single_run(args)
```

### Best Practices
| Goal | Recommendation |
|------|----------------|
| Deterministic aggregation | Fix aggregation method early (do not switch mid‑evaluation). |
| Reconciliation clarity | Store both original and reconciled outputs. |
| Failure isolation | Let a failed constituent model raise immediately—avoid suppressing errors. |
| Efficient reuse | Enable `--saved` for base models once artifacts exist. |
| Consistent targets | Ensure all base models share identical target naming schemes. |

### Common Pitfalls
| Issue | Resolution |
|-------|-----------|
| Timestamp mismatch across models | Regenerate all predictions in one run for alignment. |
| Multi-value list cells | Preprocess base model outputs to scalar predictions only. |
| Missing country model for reconciliation | Add `reconcile_with` to config or disable reconciliation. |
| Silent aggregation error | Check logs for ValueError (invalid aggregation string). |

### FAQ
| Question | Answer |
|----------|--------|
| Can I aggregate probabilistic distributions? | No—lists >1 length are rejected. |
| Must all models be trained in the same run? | Not required; existing artifacts are reused if present. |
| Is WandB mandatory? | No; disable with `wandb_notifications=False`. |
| Can I customize reconciliation logic? | Extend `ReconciliationModule` or override `_apply_reconciliation`. |
---