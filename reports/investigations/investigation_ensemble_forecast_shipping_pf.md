# Investigation: Ensemble ↔ PredictionFrame & Forecast Shipping

**Date:** 2026-03-17
**Branch:** `feature/samples_for_fao`
**Investigator:** Claude (repo assimilation + targeted code tracing)

## Context

Two open architectural questions prompted this investigation:
1. What is the relationship between the ensemble system and PredictionFrame?
2. How are true future forecasts (run_type="forecasting") shipped — where, how, and in what format?

---

## Question 1: Ensemble ↔ PredictionFrame Relationship

### Answer: PredictionFrame is model-scoped, never ensemble-scoped

**EnsembleManager never sees a PredictionFrame.** The conversion happens at the member model level before the ensemble touches anything.

```
Member Model (prediction_format="prediction_frame")
  → _forecast_model_artifact() returns Dict[str, PredictionFrame]
  → ModelManager converts via PredictionFrameConverter.to_prediction_df()
  → Saves as parquet: predictions_{run_type}_{ts}.parquet
        ↓
EnsembleManager._load_or_generate_prediction()
  → Loads parquet → pd.DataFrame (always)
  → Passes to AggregationManager (Polars-based)
  → Aggregates → saves ensemble result as parquet
```

### Key facts

- `EnsembleManager._evaluate_model_artifact()` returns `List[pd.DataFrame]` (ensemble.py line 392)
- `EnsembleManager._forecast_model_artifact()` returns `pd.DataFrame` (ensemble.py line 431)
- `AggregationManager.add_model()` accepts paths or DataFrames — never PredictionFrame
- **Mixed ensembles work seamlessly**: a PF-format model and a DF-format model both produce identical parquet on disk. The ensemble can't tell them apart.
- The parquet format is the **universal cross-model contract**: MultiIndex (time, unit) with `pred_{target}` column containing list-in-cell values

### How ensemble loads member predictions (3-tier fallback)

1. **Prediction Store** (if `use_prediction_store=True`): `pd.DataFrame.forecasts.read_store(run=..., name=...)`
2. **Local parquet file**: `read_dataframe(path_generated / "predictions_*.parquet")`
3. **Subprocess generation** (last resort): runs member model via shell command, then reloads from disk

### AggregationManager's implicit PF awareness

`AggregationManager._load_parquet_direct()` (aggregator.py lines 711-763) is explicitly designed to read parquet files produced by `PredictionFrameConverter.to_arrow_table()`. It expects flat columns:

```
month_id        int64
country_id      int64  (or priogrid_id for pgm)
pred_{target}   List<float32>
```

It does zero-copy Arrow reads. But it never touches a PredictionFrame object — only the parquet files they produce.

### Implication

**No work needed to make ensembles PF-aware.** The Strangler Fig pattern (ADR-033) correctly isolates PredictionFrame within individual model boundaries. The parquet layer is the permanent interface. This is by design, not a gap.

---

## Question 2: How True Future Forecasts Are Shipped

### Execution path: `--run_type forecasting --forecast`

```
CLI: python run.sh --run_type forecasting --forecast [--prediction_store]
  → ForecastingModelArgs.parse_args()
  → manager.execute_single_run(args)
    → _execute_data_fetching()     [fetches latest VIEWSER data]
    → _execute_model_forecasting() [WandB job_type="forecast"]
```

The `--monthly` shorthand auto-sets: `run_type=forecasting, train=True, forecast=True, prediction_store=True, wandb_notifications=True, report=True`.

### Two format paths in `_execute_model_forecasting()`

| Step | PredictionFrame path | DataFrame path |
|------|---------------------|----------------|
| **Inference** | `_forecast_model_artifact()` → `Dict[str, PredictionFrame]` | `_forecast_model_artifact()` → `pd.DataFrame` |
| **Type guard** | Fail-loud if not dict (ADR-033) | Fail-loud if dict returned |
| **Validation** | `audit_prediction_structure()` (PF is self-validating) | `CorePredictionSniffer.sniff_predictions()` |
| **Conversion** | `PredictionFrameConverter.to_prediction_df()` → DataFrame | (none needed) |
| **Save local** | `save_dataframe()` → parquet | `save_dataframe()` → parquet |
| **Upload** | Same as DF path (after conversion) | See below |

Both paths converge to the same on-disk format before any upload occurs.

### Where forecasts are saved

**Local disk**: `models/{model_name}/data/generated/predictions_forecasting_{timestamp}.parquet`

**On-disk format**: MultiIndex (month_id, priogrid_gid/country_id) with `pred_{target}` column. Each cell contains a list of floats (samples for distributional models, single float for point models).

**Filename convention** (files/utils.py `generate_output_file_name()`):
```
predictions_{run_type}_{timestamp}_{sequence_number}.{extension}
# For forecasting: sequence_number is None, so omitted
# Example: predictions_forecasting_20250317_103045.parquet
```

### Prediction store upload (when `--prediction_store` flag is set)

Two upload destinations, both triggered in `_save_predictions()` (model.py):

**1. views-forecasts store** (internal VIEWS package extension):
```python
df_predictions.forecasts.set_run(self._pred_store_name)  # e.g., "v010203_2025_03"
df_predictions.forecasts.to_store(name=name, overwrite=True)
```

**2. Appwrite cloud datastore** (if env vars configured):
```python
self._datastore.upload_data(
    file=path_generated / predictions_name,
    filename=predictions_name,
    loa=level,              # "pgm" or "cm"
    name=model_name,
    targets=targets,
    category="forecast",
    type="model"            # or "ensemble"
)
```

### Prediction store naming

`__get_pred_store_name()` (model.py) returns: `v{major}{minor}{patch}_{year}_{month}`
- Example: GitHub version "1.2.3" → `v010203_2025_03`
- Fallback: `v000100_2025_03` if version fetch fails

### Required environment variables for upload

```
APPWRITE_ENDPOINT                         — Appwrite server URL
APPWRITE_DATASTORE_PROJECT_ID             — Appwrite project ID
APPWRITE_DATASTORE_API_KEY                — API key for authentication
APPWRITE_PROD_FORECASTS_BUCKET_ID         — Storage bucket ID for forecast files
APPWRITE_PROD_FORECASTS_BUCKET_NAME       — Human-readable bucket name
APPWRITE_PROD_FORECASTS_COLLECTION_ID     — Database collection ID for metadata
APPWRITE_PROD_FORECASTS_COLLECTION_NAME   — Human-readable collection name
APPWRITE_METADATA_DATABASE_ID             — Database ID for metadata storage
APPWRITE_METADATA_DATABASE_NAME           — Human-readable database name
```

**Risk**: All via `os.getenv()` with no null validation. If unset, `AppwriteConfig` receives `None` values and will fail at upload time (not at init time).

### Arrow path limitation

`_save_predictions()` explicitly guards:
```python
if isinstance(df_predictions, pa.Table):
    raise NotImplementedError("Arrow Tables not yet supported in prediction store")
```

PredictionFrame forecasts are always converted to pandas DataFrame before upload. The zero-copy Arrow path (`to_arrow_table()`) is only used for evaluation track B (on-disk persistence for ensemble consumption), not for prediction store uploads.

### Single model vs. ensemble forecast shipping

| Aspect | Single Model | Ensemble |
|--------|-------------|----------|
| Save location | `models/{name}/data/generated/` | `ensembles/{name}/data/generated/` |
| Upload type metadata | `type="model"` | `type="ensemble"` |
| File format | Identical parquet | Identical parquet |
| Upload process | Same `_save_predictions()` | Same `_save_predictions()` |
| Additional step | — | Aggregation + optional reconciliation |

---

## Downstream Consumers

### Who reads forecasts after they're produced?

| Consumer | Mechanism | What it reads |
|----------|-----------|---------------|
| **EnsembleManager** | Local parquet files or prediction store | Member model forecasts → aggregates them |
| **views-faoapi** | Appwrite DataStore download | Latest forecast for REST API endpoints |
| **Forecast reports** | Local parquet reload | `_execute_forecast_reporting()` reads from `data_generated/` |
| **WandB** | Artifact logging during pipeline run | Metrics and metadata (not raw prediction data) |

### views-faoapi endpoints (primary external consumer)

| Endpoint | Purpose |
|----------|---------|
| `GET /data/forecast/latest` | Latest forecast as DataFrame |
| `GET /{level}/data/forecast/subset` | Subset by time/entity/features |
| `GET /{level}/analysis/forecast/hdi-map` | HDI & MAP statistical analysis |
| `GET /files/{bucket_id}/{file_id}/download` | Raw file download |

Supported levels: `pg` (default), `country`, `gaul0`, `gaul1`, `gaul2`.

The FAO API implements 3-tier caching:
1. **In-memory** (per-API-key, per-worker) — fastest
2. **Disk** (shared across workers, 3.5 week TTL) — persistent
3. **Appwrite** (on-demand download) — slowest, source of truth

Authentication: API key required via `X-API-Key` header.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│ Member Model (e.g., purple_alien)                     │
│ prediction_format = "prediction_frame" or "dataframe" │
│                                                       │
│ _forecast_model_artifact()                            │
│   → PF: Dict[str, PredictionFrame] → convert to DF   │
│   → DF: pd.DataFrame (direct)                        │
│   → save parquet to data/generated/                   │
│   → upload to Appwrite (if --prediction_store)        │
└──────────────────┬───────────────────────────────────┘
                   │ parquet file (list-in-cell format)
     ┌─────────────┼─────────────────┐
     ▼             ▼                 ▼
┌──────────┐ ┌──────────┐   ┌───────────────┐
│ Ensemble │ │ Appwrite │   │ Forecast      │
│ Manager  │ │ Store    │   │ Report        │
│ (local)  │ │(central) │   │ (HTML)        │
└────┬─────┘ └────┬─────┘   └───────────────┘
     │             │
     ▼             ▼
┌──────────┐ ┌──────────┐
│ Ensemble │ │ views-   │
│ Forecast │ │ faoapi   │
│ (parquet)│ │(REST API)│
└──────────┘ └──────────┘
     │
     ▼
  Appwrite (ensemble forecasts also uploaded)
```

---

## Key Insights

1. **Parquet is the universal contract.** Both PF and DF paths converge to identical parquet files. This is the permanent interface between models, ensembles, and external consumers.

2. **PredictionFrame is invisible to ensembles.** The Strangler Fig migration (ADR-033) correctly scopes PF within individual model boundaries. No ensemble changes needed — ever.

3. **Two upload destinations.** views-forecasts store (internal) and Appwrite cloud (external). Both triggered by `--prediction_store` flag.

4. **Arrow zero-copy path is evaluation-only.** The `NotImplementedError` guard in `_save_predictions()` prevents Arrow Tables from reaching the prediction store. Forecasts always go through pandas conversion first.

5. **Environment variable coupling is unvalidated.** 10 `os.getenv()` calls for Appwrite config with no null checks. Failures only surface at upload time, not at initialization.

6. **views-faoapi is the primary external consumer.** It downloads from Appwrite, caches aggressively (3-tier), and serves via REST endpoints. It never touches PredictionFrame — only parquet/DataFrame.

7. **Mixed-format ensembles are safe.** An ensemble containing both PF-format and DF-format member models works correctly because both produce identical parquet output. The ensemble reads parquet, not PredictionFrame.

---

## Source Files Referenced

| File | Relevant Lines | What it contains |
|------|---------------|------------------|
| `views_pipeline_core/managers/model/model.py` | 2280-2362, 1089-1110 | `_execute_model_forecasting()`, `_save_predictions()`, AppwriteConfig setup |
| `views_pipeline_core/managers/ensemble/ensemble.py` | 331-362, 383-445, 514-592 | Ensemble forecast/eval orchestration, member prediction loading |
| `views_pipeline_core/managers/prediction/prediction_frame_converter.py` | 45-74, 187-241 | `to_prediction_df()`, `to_arrow_table()` |
| `views_pipeline_core/modules/ensemble_aggregator/aggregator.py` | 82-136, 711-763 | `add_model()`, `_load_parquet_direct()` |
| `views_pipeline_core/modules/datastore/datastore.py` | 275-372 | `upload_data()` implementation |
| `views_pipeline_core/files/utils.py` | 196-230, 232-248, 303-326 | `save_dataframe()`, `save_arrow_parquet()`, filename generation |
| `views_pipeline_core/cli/args.py` | 126-508 | `ForecastingModelArgs` with `--prediction_store` validation |
| `views-faoapi/src/views_faoapi/managers/api.py` | 834-1080 | FAO API forecast endpoints |
| `views-faoapi/src/views_faoapi/managers/prediction.py` | — | `PredictionStoreManager` Appwrite download |
