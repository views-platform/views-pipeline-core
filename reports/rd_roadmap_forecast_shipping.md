# R&D Roadmap: Forecast Shipping Architecture

**Date:** 2026-03-17
**Branch:** `feature/samples_for_fao`
**Scope:** `views_pipeline_core/managers/model/model.py` (_save_predictions, _execute_model_forecasting, __init__), `views_pipeline_core/modules/datastore/`, `views_pipeline_core/managers/prediction/`, `views_pipeline_core/files/utils.py`

---

## 1. Current State

### What exists

When a model runs `--run_type forecasting --forecast`, `_execute_model_forecasting()` in ForecastingModelManager (model.py) dispatches by `prediction_format` config:

- **PredictionFrame path:** model returns `Dict[str, PredictionFrame]` → `PredictionFrameConverter.to_prediction_df()` converts to DataFrame → `_save_predictions()` writes parquet
- **DataFrame path:** model returns `pd.DataFrame` → `CorePredictionSniffer` validates → `_save_predictions()` writes parquet

Both paths converge at `_save_predictions()` (model.py, ~96 lines), which handles:
1. Local parquet write (file I/O)
2. Filename generation
3. views-forecasts store upload (`df.forecasts.set_run()` / `.to_store()`)
4. Appwrite DatastoreModule upload
5. WandB alerting

Downstream consumers:
- **EnsembleManager** reads member forecasts from local parquet or prediction store
- **views-faoapi** downloads from Appwrite for REST API endpoints (3-tier caching)

### SOLID violations inventory

| Principle | Location | Issue | Severity |
|-----------|----------|-------|----------|
| **SRP** | `_save_predictions()` model.py:2766-2862 | 5 responsibilities in one method | Critical |
| **SRP** | `__init__()` model.py:1089-1103 | Infrastructure config assembly mixed with model lifecycle | High |
| **OCP** | `_save_predictions()`:2820-2824 | No abstraction for storage backends; hardcoded to parquet | High |
| **OCP** | `_save_predictions()`:2828 | `NotImplementedError` for Arrow Tables blocks valid format | High |
| **OCP** | `_execute_model_forecasting()`:2301-2340 | String dispatch + isinstance for format routing | Medium |
| **DIP** | `__init__()`:1089-1103 | Concrete `AppwriteConfig` + `DatastoreModule`; 10 unvalidated `os.getenv()` | Critical |
| **DIP** | `_save_predictions()`:2839 | Direct call to concrete `DatastoreModule` | High |
| **DRY** | isinstance() dispatch | Repeated across 8 call sites in model.py | Medium |
| **ISP** | `_save_predictions()` + `DatastoreModule` | Depends on 9-method module, uses only `upload_data()` | Medium |

---

## 2. Target State

### Architectural principles

1. **`_save_predictions()` becomes a thin loop over injected savers.** Each saver handles exactly one concern.
2. **Storage backends are protocols.** Adding S3, GCS, or a new database requires implementing one interface — no existing code modified.
3. **Configuration is validated at startup, not at save time.** Fail-loud on missing env vars before any model runs.
4. **Format dispatch uses strategy pattern, not isinstance().** New prediction formats are addable without modifying _execute_model_forecasting().

### Target class diagram

```
ForecastingModelManager
  ├── PredictionSaver           [protocol, injected]
  │     ├── LocalParquetSaver   (writes to data/generated/)
  │     ├── ViewsForecastsSaver (df.forecasts.set_run/to_store)
  │     └── AppwriteSaver       (DatastoreModule.upload_data)
  ├── PredictionFormatHandler   [protocol, injected]
  │     ├── DataFrameHandler    (CorePredictionSniffer + direct save)
  │     └── PredictionFrameHandler (PFConverter + audit + save)
  ├── PredictionFileNamer       [extracted utility]
  └── PredictionStoreConfig     [validated at startup]
```

---

## 3. Phased Milestones

### Phase 1: Extract `PredictionFileNamer` (SRP, low risk)

**Goal:** Remove filename generation logic from `_save_predictions()`.

**What:**
- Create `PredictionFileNamer` class that encapsulates `generate_output_file_name()` and `generate_evaluation_file_name()` calls
- `_save_predictions()` receives the filename, doesn't compute it

**Acceptance criteria:**
- Naming logic has one home (PredictionFileNamer)
- `_save_predictions()` reduced by ~10 lines
- All tests pass

**Risk:** Low. Pure extraction.

---

### Phase 2: Create `PredictionSaver` protocol + implementations (SRP + OCP + DIP)

**Goal:** Decompose `_save_predictions()` into composable, single-responsibility savers.

**What:**
```python
class PredictionSaver(Protocol):
    def save(self, data: Union[pd.DataFrame, pa.Table], path: Path,
             metadata: PredictionMetadata) -> None: ...

class LocalParquetSaver:
    """Writes to local filesystem as parquet."""

class ViewsForecastsSaver:
    """Uploads to views-forecasts prediction store."""

class AppwriteSaver:
    """Uploads to Appwrite cloud datastore."""
```

- `_save_predictions()` becomes:
```python
def _save_predictions(self, data, path, ...):
    for saver in self._savers:
        saver.save(data, path, metadata)
```

**Acceptance criteria:**
- `_save_predictions()` is < 15 lines (loop over savers)
- New storage backend (e.g., S3) addable by implementing `PredictionSaver` — zero changes to model.py
- Arrow `NotImplementedError` resolved: `AppwriteSaver` handles conversion internally, not in the caller

**Risk:** Medium. Must preserve exact upload behavior including `set_run()` / `to_store()` API.

---

### Phase 3: Extract configuration validation (DIP, fail-loud)

**Goal:** Validate environment variables at startup, not at save time.

**What:**
- Create `PredictionStoreConfig` dataclass that validates all required env vars in `__init__`
- Create `PredictionStoreConfigFactory.from_environment()` that reads env vars once and fails loud
- ForecastingModelManager receives `PredictionStoreConfig` (or None) via constructor
- Remove 10 `os.getenv()` calls from ForecastingModelManager.__init__()

**Acceptance criteria:**
- Missing env var → clear error at program start, not at save time
- ForecastingModelManager has zero `os.getenv()` calls
- Tests can inject config without env var manipulation

**Risk:** Low-medium. Changes constructor signature; existing tests that mock env vars need updating.

---

### Phase 4: Create `PredictionFormatHandler` protocol (OCP + DRY)

**Goal:** Eliminate isinstance() dispatch and string matching in forecasting/evaluation paths.

**What:**
```python
class PredictionFormatHandler(Protocol):
    def handle_forecast(self, model_artifact_fn, artifact_name, configs) -> None: ...
    def handle_evaluation(self, model_artifact_fn, eval_type, configs) -> None: ...

class DataFrameFormatHandler:
    """Handles prediction_format='dataframe' — validates with CorePredictionSniffer, saves directly."""

class PredictionFrameFormatHandler:
    """Handles prediction_format='prediction_frame' — converts via PredictionFrameConverter, audits, saves."""
```

- `_execute_model_forecasting()` becomes:
```python
self._format_handler.handle_forecast(
    self._forecast_model_artifact, self.args.artifact_name, self.configs
)
```

**Acceptance criteria:**
- No isinstance() checks in `_execute_model_forecasting()` or `_execute_model_evaluation()`
- Adding a third format requires implementing `PredictionFormatHandler` + registering — zero changes to model.py
- 8 isinstance() call sites consolidated into 2 handler classes

**Risk:** High. Touches the core dispatch logic in the two most complex methods. Requires extensive test coverage first.

---

## 4. Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Low | Pure extraction; backward-compatible |
| 2 | Medium | Must preserve exact upload behavior; integration test with mock Appwrite |
| 3 | Low-Medium | Changes constructor; tests need updating |
| 4 | High | Core dispatch logic; add comprehensive tests before refactoring |

---

## 5. What NOT to change

- The parquet on-disk format (universal contract with ensemble + faoapi)
- The views-forecasts `df.forecasts.set_run()` / `to_store()` API (external package)
- The DatastoreModule / AppWriteFileModule internals (stable, tested)
- The PredictionFrameConverter logic (proven, correct)
- The `prediction_format` config key as dispatch authority (ADR-031/033)
- How downstream consumers (EnsembleManager, views-faoapi) read forecasts
