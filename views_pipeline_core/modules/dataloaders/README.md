# VIEWS Pipeline Core: Data Loading Module

Core Classes — **one per file since #431** (they were both in `dataloaders.py` until then):

| class | file |
|---|---|
| `ViewsDataLoader` (partition-aware end‑to‑end data fetch, caching, drift monitoring) | `views_pipeline_core/modules/dataloaders/dataloaders.py` |

Imported from the package: `from views_pipeline_core.modules.dataloaders import ViewsDataLoader`.


---

## Purpose

This module standardizes data acquisition for forecasting pipelines in the VIEWS ecosystem. It covers:

| Functionality | Description |
|---------------|-------------|
| Queryset parsing | Extract raw variables, output names, ordered transformation chains |
| Partitioning | Calibration / validation / forecasting month span definitions (train/test split logic) |
| Drift detection | Input monitoring via configurable statistical tests (ADR‑014 alignment) |
| Caching | Local raw parquet persistence with provenance log file creation |
| Type safety | Numeric stabilization (`ensure_float64`) and index consistency |

---

## High-Level Workflow

```
        ┌────────────────────────┐
        │   Model Configuration  │
        │  (config_queryset.py)  │
        └──────────┬────────────┘
                   │
             Queryset object
                   │
          ViewsDataLoader.get_data()
                   │
          ┌────────┴────────┐
          │ Fetch (viewser) │ ← drift detection (optional)
          └────────┬────────┘
                   │
        Cache parquet + log file
                   │
             Validation (partition)
                   │
              DataFrame + alerts
```

---

## Class: ViewsDataLoader

### Overview

Primary orchestration class for partition-aware model data ingestion. Integrates queryset resolution, partition slicing, drift detection, , caching, and validation.

### Initialization

```python
loader = ViewsDataLoader(
    model_path=ModelPathManager("purple_alien"),
    steps=36
)
```

Args:
- `model_path (ModelPathManager)`: Provides directory scaffold.
- `partition_dict (Dict | None)`: Override default partition ranges.
- `steps (int)`: Forecast horizon (used in forecasting partition).
- `**kwargs`: Optional overrides (e.g. `partition`, `override_month`).

### Partitions (Default)

| Partition | Train (month_id) | Test (month_id) |
|-----------|------------------|-----------------|
| calibration | 121–396 | 397–444 |
| validation | 121–444 | 445–492 |
| forecasting | 121–(current−1) | (current)–(current+steps) |

Month ID 121 = 1990-01 (months since 1980-01).

### get_data()

```python
df, alerts = loader.get_data(
    self_test=False,
    partition="calibration",
    use_saved=True,
    validate=True
)
```

Args:
- `self_test (bool)`: Enable drift self-test.
- `partition (str)`: One of `calibration`, `validation`, `forecasting`.
- `use_saved (bool)`: Load cached parquet if present.
- `validate (bool)`: Enforce partition alignment.
- `override_month (int | None)`: Adjust end month (forecasting only).

Returns:
- DataFrame (MultiIndex: `month_id`, entity id)
- Drift alerts list (empty or structured objects)

Process:
1. Determine partition dict (default or provided).
2. Compute month range (`_get_month_range()`).
3. If `use_saved` and file exists → load.
4. Else → `_fetch_data_from_viewser()`:
   - Queryset resolution
   - Drift detection call (`fetch_with_drift_detection`)
   - Fallback on `KeyError` → `fetch()` without drift
   - Convert numeric types (`ensure_float64`)
5. Save file + create fetch log (`create_data_fetch_log_file`).
6. Partition validation (`_validate_df_partition()`).
7. Return (df, alerts).

Raises:
- `RuntimeError` (missing queryset / incompatible partition)
- `ValueError` (invalid partition string)

### Internal Methods

| Method | Purpose |
|--------|---------|
| `_get_partition_dict(steps)` | Build default partition windows |
| `_fetch_data_from_viewser(self_test)` | Queryset fetch + drift + update |
| `_get_month_range()` | Final month_first / month_last resolution |
| `_validate_df_partition(df)` | Temporal alignment check |

### Drift Detection Integration

- Config via `drift_detection.drift_detection_partition_dict[partition]`.
- Alerts logged (contains offender metadata).
- Used primarily in forecasting pipeline runs (production gating).

### Caching & Provenance

| Artifact | Location |
|----------|----------|
| Raw parquet | `model_path/data/raw/{partition}_viewser_df.parquet` |
| Fetch log | `model_path/data/raw/data_fetch_log_{partition}_{timestamp}.txt` |

### Example (Forecasting)

```python
df_forecast, drift_alerts = loader.get_data(
    self_test=False,
    partition="forecasting",
    use_saved=False,
    override_month=530
)
if drift_alerts:
    for a in drift_alerts:
        print(a)
```

---

## Validation Logic

| Partition | Expected Range Check |
|-----------|----------------------|
| calibration / validation | DataFrame min = train.start, max = test.end |
| forecasting | DataFrame min = train.start, max = train.end (override respected) |

Failure → log error + raise `RuntimeError`.

---

## Error Handling Summary

| Context | Failure | Action |
|---------|---------|--------|
| Queryset missing | `None` | Raise `RuntimeError` |
| Drift fetch key error | Missing feature for drift | Retry without drift; log error |
| Partition mismatch | Month range misaligned | Raise `RuntimeError` |
| `.env` missing | File absent | Raise `FileNotFoundError` |

---

## Performance Notes

| Aspect | Consideration |
|--------|---------------|
| Drift detection | Adds overhead; disable for development |
| Large parquet writes | Use snappy compression (configured externally) |

---

## Best Practices

| Goal | Recommendation |
|------|----------------|
| Reproducibility | Commit queryset configuration; store fetch logs |
| Partition integrity | Always keep `validate=True` in production |
| Debugging | Temporarily disable drift (`drift_config_dict=None`) |
| Memory | Drop unused intermediate columns after transformation |

---

## Common Pitfalls

| Pitfall | Resolution |
|---------|------------|
| Forgetting raw variables in queryset | Ensure `raw_*` rename present |
| Using wrong LOA in `.env` paths | Match `priogrid_month` vs `country_month` |
| Partition dict mis-specified | Pass full dict keyed by partition name |
| Silent drift alerts | Inspect WARNING logs for “offender” entries |

---

## Minimal End-to-End Example

```python
from viewser import Queryset
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.modules.dataloaders import ViewsDataLoader

model_path = ModelPathManager("purple_alien")
loader = ViewsDataLoader(model_path, steps=24)

# Calibration data (fresh fetch)
calib_df, calib_alerts = loader.get_data(
    self_test=False,
    partition="calibration",
    use_saved=False,
    validate=True
)

# Forecasting (reuse cached)
forecast_df, forecast_alerts = loader.get_data(
    self_test=False,
    partition="forecasting",
    use_saved=True,
    validate=True
)
```

---

## FAQ

| Question | Answer |
|----------|--------|
| Can I skip drift detection? | Yes—set `drift_config_dict=None` before fetch. |
| Why MultiIndex required? | Enables (month_id, entity_id) alignment for temporal + spatial transformations. |
| What if transformation changes index length? | Module reindexes to original with warning. |

---