# VIEWS Pipeline Core: Data Loading & Viewser Update Module

File: `views_pipeline_core/modules/dataloaders/dataloaders.py`  
Core Classes:  
- `UpdateViewser` (incremental GED / ACLED refresh + transformation replay)  
- `ViewsDataLoader` (partition-aware end‑to‑end data fetch, caching, drift monitoring, optional updating)

Ancillary Assets:
- `transformation_mapping` (name → callable registry)
- `TRANSFORMATIONS_EXPECTING_DF` (transformations needing DataFrame input)
- Lazy Ingester / transformation imports (avoid CI breakages without certificates)

---

## Purpose

This module standardizes data acquisition for forecasting pipelines in the VIEWS ecosystem. It covers:

| Functionality | Description |
|---------------|-------------|
| Queryset parsing | Extract raw variables, output names, ordered transformation chains |
| Incremental updates | Replace raw GED / ACLED slices with newest values; recompute derived transformations |
| Partitioning | Calibration / validation / forecasting month span definitions (train/test split logic) |
| Drift detection | Input monitoring via configurable statistical tests (ADR‑014 alignment) |
| Caching | Local raw parquet persistence with provenance log file creation |
| Type safety | Numeric stabilization (`ensure_float64`) and index consistency |
| Environment-driven updates | `.env` + LOA-driven update file resolution (country vs priogrid) |

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
          _overwrite_viewser? (UpdateViewser)
                   │
          Transform replay (raw_* → derived)
                   │
        Cache parquet + log file
                   │
             Validation (partition)
                   │
              DataFrame + alerts
```

---

## Transformation Registry

`transformation_mapping` provides name → function resolution for queryset replay:

| Key | Callable | Category |
|-----|----------|----------|
| ops.ln | `views2.ln` | Log transform |
| temporal.tlag | `views2.tlag` | Temporal lag |
| temporal.decay | `views2.decay` | Decay weighting |
| temporal.time_since | `views2.time_since` | Time since event |
| temporal.moving_sum | `views2.moving_sum` | Rolling aggregation |
| missing.fill | `missing.fill` | Imputation |
| missing.replace_na | `missing.replace_na` | NA substitution |
| spatial.countrylag | `splag_country.get_splag_country` | Country lag |
| spatial.lag | `splag4d.get_splag4d` | 4D spatial lag |
| spatial.treelag | `spatial_tree.get_tree_lag` | Tree lag |
| spatial.sptime_dist | `spacetime_distance.get_spacetime_distances` | Space–time distances |
| bool.gte | `views2.greater_or_equal` | Boolean threshold |
| temporal.moving_average | `views2.moving_sum` (same primitive) | Rolling average |

Special handling: `TRANSFORMATIONS_EXPECTING_DF = {"spatial.lag", "spatial.sptime_dist"}` forces DataFrame input.

---

## Class: UpdateViewser

### Overview

Incrementally refresh VIEWSER-derived DataFrame with most recent GED / ACLED raw values while preserving transformation fidelity. Designed for production monthly updates without full refetch cost.

### Initialization

```python
updater = UpdateViewser(
    queryset=queryset,
    viewser_df=original_viewser_df,
    data_path="updates/ged_acled_latest.parquet",
    months_to_update=[528, 529, 530]
)
```

Args:
- `queryset (Queryset)`: Must include at least one renamed raw variable (`raw_*`).
- `viewser_df (pd.DataFrame)`: Existing data (MultiIndex: `month_id`, entity id).
- `data_path (str | Path)`: Parquet file containing updated raw columns.
- `months_to_update (List[int])`: Month IDs to replace.

Raises:
- `ValueError` (no `raw_` variables or external data older than current viewser slice)
- `FileNotFoundError` (missing update file)

### Attributes

| Attribute | Description |
|-----------|-------------|
| `base_variables` | Fully-qualified original source names (e.g. `country_month.ged_sb_best_sum_nokgi`) |
| `var_names` | Final queryset output names (including `raw_` and derived) |
| `transformation_list` | Ordered transformation dictionaries per variable |
| `df_external` | Loaded update parquet (MultiIndex compatible) |
| `result` | Cached final DataFrame (set after `run()`) |

### Public Method: run()

Executes complete refresh.

Steps:
1. Early return if cached.
2. Preprocess update file → month slice + raw renaming.
3. In-place `.update()` of matching raw columns.
4. Sequential transformation replay (log, lags, spatial).
5. Drops `raw_*` columns (only transformed outputs retained).
6. Returns updated DataFrame.

Returns:
- Updated `pd.DataFrame` (cached on first execution)

Idempotent: Subsequent calls reuse `self.result`.

### Internal Methods

| Method | Role |
|--------|------|
| `_extract_from_queryset()` | Parse operations into parallel lists |
| `_preprocess_update_df()` | Column + month filtering + renaming |
| `_apply_all_transformations(df_old)` | Replay transformations respecting order |
| `_smart_cast(arg)` | Convert string arguments → Python literals (`ast.literal_eval`) |

### Transformation Replay Notes

- Skips non-GED/ACLED derived outputs (name filter).
- Respects original order by reversing captured operations.
- Handles index alignment warnings (reindex fallback).
- Special-case: `spatial.countrylag` → forward fill per group.

### Example

```python
updated_df = updater.run()
assert "ln_ged_sb_tlag_1" in updated_df.columns
```

---

## Class: ViewsDataLoader

### Overview

Primary orchestration class for partition-aware model data ingestion. Integrates queryset resolution, partition slicing, drift detection, optional VIEWSER update, caching, and validation.

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
   - Optional overwrite via UpdateViewser
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
| `_overwrite_viewser(df, queryset_base, args)` | Conditional GED/ACLED refresh |
| `_get_viewser_update_config(queryset_base)` | Resolve `.env` config (months, path) |
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

### Environment-Based Update (.env)

Required keys for overwrite:
- `month_to_update=[528, 529, 530]`
- `pgm_path=/path/to/priogrid_updates.parquet`
- `cm_path=/path/to/country_updates.parquet`

Errors:
- Missing `.env` → `FileNotFoundError`
- Missing `month_to_update` → `ValueError`

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
| Update preprocessing | No column overlap | Raise `ValueError` |
| External update staleness | viewser month > external month | Raise `ValueError` |
| `.env` missing | File absent | Raise `FileNotFoundError` |

---

## Performance Notes

| Aspect | Consideration |
|--------|---------------|
| Transformation replay | Sequential per variable; optimize by grouping if needed |
| Drift detection | Adds overhead; disable for development |
| Large parquet writes | Use snappy compression (configured externally) |
| Update mechanism | Only raw slice modifications → cheap |

---

## Best Practices

| Goal | Recommendation |
|------|----------------|
| Reproducibility | Commit queryset configuration; store fetch logs |
| Freshness | Use incremental update (UpdateViewser) monthly |
| Partition integrity | Always keep `validate=True` in production |
| Debugging | Temporarily disable drift (`drift_config_dict=None`) |
| Memory | Drop unused intermediate columns after transformation |

---

## Common Pitfalls

| Pitfall | Resolution |
|---------|------------|
| Forgetting raw variables in queryset | Ensure `raw_*` rename present |
| External file older than viewser | Refresh update parquet before run |
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
| Do updates mutate original DataFrame? | Yes—raw columns updated in place before transformation replay. |
| How are transformation arguments parsed? | Safely with `ast.literal_eval` (`_smart_cast`). |
| Can I partially update months? | Yes—pass specific `months_to_update` list in `.env`. |
| What if transformation changes index length? | Module reindexes to original with warning. |

---