# Product Development Plan: Forecast Shipping Refactoring

**Date:** 2026-03-17
**Branch:** `feature/samples_for_fao`
**Prerequisite:** Read `reports/rd_roadmap_forecast_shipping.md` for architectural context.

---

## Task List (ordered by dependency and safety)

### Task 1: Extract `PredictionFileNamer`

**Priority:** P0 (unblocks Task 2; highest safety)
**SOLID principle:** SRP
**Effort:** Small

**Current state (model.py _save_predictions, lines ~2812-2818):**
Filename generation logic is inline:
```python
self._predictions_name = generate_output_file_name(
    generated_file_type="predictions",
    run_type=self.args.run_type,
    timestamp=self.configs["timestamp"],
    sequence_number=origin_idx,
    file_extension=PipelineConfig.dataframe_format,
)
```

**Target:**
```python
# New: views_pipeline_core/managers/prediction/file_namer.py
class PredictionFileNamer:
    def __init__(self, run_type: str, timestamp: str,
                 file_extension: str = ".parquet"):
        ...
    def prediction_name(self, sequence_number: Optional[int] = None) -> str: ...
    def evaluation_name(self, eval_type: str, target: str) -> str: ...
```

**Files to modify:**
- Create: `views_pipeline_core/managers/prediction/file_namer.py`
- Modify: `views_pipeline_core/managers/model/model.py` — `_save_predictions()` uses injected namer

**Test strategy:**
- Unit tests for `PredictionFileNamer` with known inputs/outputs
- Existing tests pass unchanged

---

### Task 2: Create `PredictionSaver` protocol + `LocalParquetSaver`

**Priority:** P0 (foundation for Tasks 3-4)
**SOLID principle:** SRP, OCP, DIP
**Effort:** Medium

**Current state (model.py _save_predictions, lines ~2820-2824):**
```python
if isinstance(df_predictions, pa.Table):
    pq.write_table(df_predictions, path_generated / self._predictions_name)
else:
    save_dataframe(df_predictions, path_generated / self._predictions_name)
```

**Target:**
```python
# New: views_pipeline_core/managers/prediction/savers.py

@dataclass
class PredictionMetadata:
    model_name: str
    level: str          # "pgm" or "cm"
    targets: List[str]
    category: str       # "forecast" or "evaluation"
    run_type: str
    filename: str

class PredictionSaver(Protocol):
    def save(self, data: Union[pd.DataFrame, pa.Table],
             path: Path, metadata: PredictionMetadata) -> None: ...

class LocalParquetSaver:
    """Writes prediction data to local filesystem as parquet."""
    def save(self, data, path, metadata) -> None:
        dest = path / metadata.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, pa.Table):
            pq.write_table(data, dest)
        else:
            save_dataframe(data, dest)
```

**Files to modify:**
- Create: `views_pipeline_core/managers/prediction/savers.py`
- Modify: `views_pipeline_core/managers/model/model.py` — inject `List[PredictionSaver]` into constructor

**Test strategy:**
- Unit tests for `LocalParquetSaver` with tmp_path
- Test Arrow Table save + DataFrame save paths
- Existing tests pass (default saver list = [LocalParquetSaver()])

**Reuse:** `save_dataframe()` from `views_pipeline_core/files/utils.py:196` and `save_arrow_parquet()` from `views_pipeline_core/files/utils.py:232`

---

### Task 3: Create `PredictionStoreConfig` with env var validation

**Priority:** P1 (unblocks Task 4)
**SOLID principle:** DIP, SRP (fail-loud)
**Effort:** Small

**Current state (model.py __init__, lines ~1089-1103):**
10 `os.getenv()` calls with no validation:
```python
self._appwrite_config = AppwriteConfig(
    endpoint=os.getenv("APPWRITE_ENDPOINT"),       # may be None
    project_id=os.getenv("APPWRITE_DATASTORE_PROJECT_ID"), # may be None
    ...
)
```

**Target:**
```python
# New: views_pipeline_core/configs/prediction_store.py

@dataclass(frozen=True)
class PredictionStoreConfig:
    endpoint: str
    project_id: str
    api_key: str
    bucket_id: str
    bucket_name: str
    collection_id: str
    collection_name: str
    database_id: str
    database_name: str
    pred_store_name: str

    @classmethod
    def from_environment(cls) -> "PredictionStoreConfig":
        """Read all env vars and fail-loud if any are missing."""
        required = {
            "endpoint": "APPWRITE_ENDPOINT",
            "project_id": "APPWRITE_DATASTORE_PROJECT_ID",
            "api_key": "APPWRITE_DATASTORE_API_KEY",
            ...
        }
        values = {}
        missing = []
        for field, env_var in required.items():
            val = os.getenv(env_var)
            if val is None:
                missing.append(env_var)
            values[field] = val
        if missing:
            raise ConfigurationException(
                f"Missing required environment variables for prediction store: {missing}"
            )
        return cls(**values)
```

**Files to modify:**
- Create: `views_pipeline_core/configs/prediction_store.py`
- Modify: `views_pipeline_core/managers/model/model.py` — `__init__()` receives config, doesn't create it

**Test strategy:**
- Unit test: all env vars present → config created
- Unit test: any env var missing → `ConfigurationException` raised immediately
- Existing tests: inject config directly, no env var manipulation needed

---

### Task 4: Create `AppwriteSaver` and `ViewsForecastsSaver`

**Priority:** P1 (depends on Tasks 2 + 3)
**SOLID principle:** SRP, OCP
**Effort:** Medium

**Current state (model.py _save_predictions, lines ~2826-2850):**
Two separate upload systems hardcoded in one method:
```python
if self._use_prediction_store:
    if isinstance(df_predictions, pa.Table):
        raise NotImplementedError(...)
    df_predictions.forecasts.set_run(self._pred_store_name)
    df_predictions.forecasts.to_store(name=name, overwrite=True)
    if self._datastore is not None:
        self._datastore.upload_data(...)
```

**Target:**
```python
# In views_pipeline_core/managers/prediction/savers.py

class ViewsForecastsSaver:
    """Uploads to views-forecasts prediction store."""
    def __init__(self, pred_store_name: str): ...
    def save(self, data, path, metadata) -> None:
        if isinstance(data, pa.Table):
            data = data.to_pandas()  # Convert instead of raising
        data.forecasts.set_run(self._pred_store_name)
        data.forecasts.to_store(name=..., overwrite=True)

class AppwriteSaver:
    """Uploads to Appwrite cloud datastore."""
    def __init__(self, datastore: DatastoreModule): ...
    def save(self, data, path, metadata) -> None:
        self._datastore.upload_data(
            file=path / metadata.filename,
            filename=metadata.filename,
            loa=metadata.level,
            name=metadata.model_name,
            targets=metadata.targets,
            category=metadata.category,
            type="model",
        )
```

**Key improvement:** Arrow `NotImplementedError` is resolved — `ViewsForecastsSaver` converts Arrow → DataFrame internally instead of raising. The caller never sees format-specific exceptions.

**Files to modify:**
- Modify: `views_pipeline_core/managers/prediction/savers.py` (add two classes)
- Modify: `views_pipeline_core/managers/model/model.py`:
  - `__init__()`: compose saver list based on `use_prediction_store` flag
  - `_save_predictions()`: iterate `self._savers`

**Test strategy:**
- Unit tests for `ViewsForecastsSaver` with mocked `df.forecasts`
- Unit tests for `AppwriteSaver` with mocked `DatastoreModule`
- Integration test: Arrow Table → `ViewsForecastsSaver` converts without raising

---

### Task 5: Refactor `_save_predictions()` to compose savers

**Priority:** P1 (depends on Tasks 1-4)
**SOLID principle:** SRP (final consolidation)
**Effort:** Small

**Target `_save_predictions()`:**
```python
def _save_predictions(self, data, path_generated, origin_idx=None,
                      send_alert=True):
    metadata = PredictionMetadata(
        model_name=self._model_path.model_name,
        level=self.configs.get("level"),
        targets=self.configs.get("targets"),
        category="forecast" if self.args.run_type == "forecasting" else "evaluation",
        run_type=self.args.run_type,
        filename=self._file_namer.prediction_name(origin_idx),
    )
    for saver in self._savers:
        saver.save(data, path_generated, metadata)
    if send_alert:
        self._wandb_module.send_alert(...)
```

**Result:** ~96 lines → ~15 lines. Each saver has exactly one responsibility.

**Files to modify:**
- Modify: `views_pipeline_core/managers/model/model.py` — rewrite `_save_predictions()`

**Test strategy:**
- Existing tests pass (mock savers injected in test fixtures)
- Add test: verify each saver called exactly once per save
- Add test: verify alert sent when `send_alert=True`

---

### Task 6: Create `PredictionFormatHandler` protocol (Phase 4, deferred)

**Priority:** P2 (optional, high risk)
**SOLID principle:** OCP, DRY
**Effort:** Large

**Current state:** `_execute_model_forecasting()` (lines 2301-2340) uses string dispatch:
```python
if self._prediction_format == "prediction_frame":
    # PF path: 20 lines
else:
    # DF path: 18 lines
```

Same pattern in `_execute_model_evaluation()` and `_execute_model_sweeping()`.

**Target:**
```python
class PredictionFormatHandler(Protocol):
    def handle_forecast(self, forecast_fn, artifact_name, configs,
                        savers: List[PredictionSaver]) -> None: ...

class DataFrameFormatHandler:
    def handle_forecast(self, ...): ...  # CorePredictionSniffer + save

class PredictionFrameFormatHandler:
    def handle_forecast(self, ...): ...  # PFConverter + audit + save
```

**Deferred because:**
- Touches the two most complex methods in the codebase (~230 lines each)
- Requires comprehensive test coverage first (currently tested but with heavy mocking)
- Risk of breaking evaluation/sweeping paths

**Prerequisite:** Complete Tasks 1-5 first; add integration tests for forecast + evaluation paths.

---

## Dependency Graph

```
Task 1 (FileNamer)           — independent
Task 2 (PredictionSaver)     — independent
Task 3 (PredictionStoreConfig) — independent
Task 4 (Appwrite/ViewsForecastsSavers) — depends on Tasks 2 + 3
Task 5 (Compose savers)      — depends on Tasks 1 + 4
Task 6 (FormatHandler)       — depends on Task 5; DEFERRED
```

**Recommended execution order:** 1 → 2 → 3 → 4 → 5 (→ 6 when ready)

---

## Verification (after Tasks 1-5)

```bash
# Full test suite
conda run -n views_pipeline pytest tests/ --tb=short -q

# Lint modified files
conda run -n views_pipeline ruff check \
  views_pipeline_core/managers/model/model.py \
  views_pipeline_core/managers/prediction/ \
  views_pipeline_core/configs/prediction_store.py

# Line count target for _save_predictions
grep -A 20 "def _save_predictions" views_pipeline_core/managers/model/model.py | wc -l
# Target: < 20 lines (from ~96)

# Verify no os.getenv in model.py
grep -c "os.getenv" views_pipeline_core/managers/model/model.py
# Target: 0 (moved to PredictionStoreConfig)

# Verify no NotImplementedError in _save_predictions
grep "NotImplementedError" views_pipeline_core/managers/model/model.py
# Target: 0 (resolved in ViewsForecastsSaver)
```

---

## Scope Boundaries

**DO:**
- Extract savers behind protocols
- Validate config at startup
- Resolve Arrow NotImplementedError by converting in saver (not raising)
- Inject dependencies via constructors

**DO NOT:**
- Change the parquet on-disk format
- Modify `views-forecasts` package API (`df.forecasts.*`)
- Modify `DatastoreModule` / `AppWriteFileModule` internals
- Change `PredictionFrameConverter` logic
- Change how downstream consumers (ensemble, faoapi) read forecasts
- Attempt `PredictionFormatHandler` (Task 6) until Tasks 1-5 are stable and tested
