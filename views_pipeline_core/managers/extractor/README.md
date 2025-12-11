# Extractor Management

File: `views_pipeline_core/managers/extractor/extractor.py`  
Classes:
- `ExtractorPathManager`
- `ExtractorManager` (abstract base for concrete extractors)

Purpose: Provide a standardized, reproducible framework to implement data extractors that download, preprocess, and persist external datasets into the VIEWS Pipeline Core environment with optional WandB run logging.

---

## 1. ExtractorPathManager

### Overview
Specialized path manager for extractors. Inherits from `ModelPathManager` but switches its namespace root to `<project_root>/extractors/<extractor_name>/`. Supplies directory scaffolding (data, artifacts, logs, configs, reports) and script path registration identical in pattern to models and ensembles.

### Key Differences vs ModelPathManager
| Aspect | Value |
|--------|-------|
| `_target` | `"extractor"` |
| Base directory | `<root>/extractors/<name>/` |
| Purpose | External/raw data ingestion rather than model artifacts |
| Validation | Same regex (`^[a-z]+_[a-z]+$`) |

### Class Initialization
```python
from views_pipeline_core.managers.extractor.extractor import ExtractorPathManager
epm = ExtractorPathManager("ged_updates")
print(epm.data_raw)            # Path to raw data dir
print(epm.logging)             # Path to logging dir
epm.view_directories()
```

### Class Method: `_initialize_class_paths`
Called once on first instantiation. Resolves project root (ascends until `.gitignore`), then sets:
```text
_root = <project_root>
_models = <project_root>/extractors
```

### Responsibilities
- Validate extractor name or derive from path.
- Ensure directory creation (raw, processed, generated, artifacts, reports, logs).
- Provide script lookup for configuration modules (if present).
- Allow later extension (custom extractor-specific directories).

### Failure Modes
| Condition | Result |
|-----------|--------|
| Invalid name format | Raises `ValueError` upstream in base class |
| Root discovery failure | Raises `RuntimeError` (no marker) |
| Filesystem permission denied | Propagates `OSError` |

---

## 2. ExtractorManager (Abstract)

### Overview
Abstract pipeline controller for data extraction tasks. Defines the contract:
1. `_download()` — acquire remote / external data.
2. `_preprocess()` — clean, normalize, enrich, convert types.
3. `_save()` — write processed data into canonical storage (e.g., database, parquet).

Implements a `run()` method that wraps the core lifecycle in a WandB run context (if notifications enabled), catching exceptions and emitting structured failure signals.

### Constructor
```python
from views_pipeline_core.managers.extractor.extractor import (
    ExtractorManager, ExtractorPathManager
)

epm = ExtractorPathManager("ged_updates")
manager = MyGedExtractor(epm, wandb_notifications=True)
```

Args:
- `model_path`: `ExtractorPathManager` instance.
- `wandb_notifications` (bool): Enable WandB alerts/logging.

Internal effects:
- Calls `ModelManager.__init__` with `use_prediction_store=False`.
- Sets `self.data = None` placeholder for intermediate storage.

### Abstract Methods
Implement in subclass:
```python
def _download(self): ...
def _preprocess(self): ...
def _save(self): ...
```

Guidelines:
- `_download`: Populate `self.data_raw` or an in-memory structure.
- `_preprocess`: Transform raw into model-ready or standardized DataFrame, assign to `self.data`.
- `_save`: Persist `self.data` (to parquet, database, or artifact).

### Lifecycle: `run()`
Execution sequence:
1. Opens WandB run: project name pattern `"{configs['name']}_save"`.
2. Inside try block:
   - Calls `_download()`.
   - Calls `_preprocess()`.
   - Calls `_save()`; expected to write outputs and optionally log artifacts.
3. On exception:
   - Raises `PipelineException` with message and passes `wandb_module` for alerting.
4. Finally:
   - Ensures `wandb_module.finish_run()` is invoked.

Note: `_download()` and `_preprocess()` are only called inside the WandB context. If early failure surfacing is desired, consider manually invoking these methods before opening the WandB run.

### Example Subclass
```python
class GedMonthlyExtractor(ExtractorManager):
    def _download(self):
        # Fetch from remote API / S3 / internal store
        self.raw = pl.read_parquet("remote/ged_latest.parquet")

    def _preprocess(self):
        self.data = (
            self.raw
            .with_columns([
                pl.col("fatalities").cast(pl.Float64),
                (pl.col("event_date").str.strptime(pl.Date, "%Y-%m-%d"))
            ])
            .groupby(["country_id", "month_id"])
            .agg(pl.sum("fatalities").alias("ged_sb_month_sum"))
            .sort(["country_id", "month_id"])
        )

    def _save(self):
        out_path = self._model_path.data_processed / "ged_sb_month_sum.parquet"
        self.data.write_parquet(out_path)
```
Usage:
```python
epm = ExtractorPathManager("ged_monthly")
manager = GedMonthlyExtractor(epm, wandb_notifications=True)
manager.run()
```

### Error Handling
| Stage | Exception | Action |
|-------|-----------|--------|
| Download | Network / I/O | Propagates → wrapped as `PipelineException` |
| Preprocess | Type mismatch | Propagates → wrapped |
| Save | Filesystem / serialization | Propagates → wrapped |

### WandB Integration
If `wandb_notifications=True`:
- Run tagged with job_type `"save"`.
- Failures produce alert message.
- Extend subclass to log artifacts:
  ```python
  self._wandb_module.log_artifact(file_path=out_path, name="ged_monthly_parquet", type="dataset")
  ```

### Best Practices
| Goal | Recommendation |
|------|----------------|
| Idempotency | Ensure repeated `.run()` produces identical output for same raw source |
| Schema stability | Keep column naming consistent (document in extractor README) |
| Provenance | Write a fetch log or embed source metadata (timestamp, source URL) |
| Validation | Add post-save assertion (e.g. non-null counts) |
| Performance | Use Polars lazy operations for heavy transformations |
| Testing | Unit test each abstract step separately (mock remote I/O in `_download`) |

### Common Pitfalls
| Pitfall | Mitigation |
|---------|------------|
| Double download overhead | Remove pre-flight `_download()` call if not needed |
| Silent partial save | Add row count logging before and after `_save()` |
| Schema drift breaking consumers | Version output (e.g. `ged_v2.parquet`) |
| Missing failure alerts | Ensure `wandb_notifications=True` in production |

### Extension Ideas
| Feature | Approach |
|---------|---------|
| Parallel ingestion | Use `concurrent.futures` inside `_download()` |
| Incremental updates | Compare latest month in stored parquet vs remote |
| Quality checks | Add `_validate()` method called before `_save()` |
| Multi-output save | Write both raw and processed artifacts (log both) |

### Minimal Polars Pipeline Pattern
```python
def _preprocess(self):
    self.data = (
        self.raw.lazy()
        .filter(pl.col("fatalities") >= 0)
        .groupby(["country_id", "month_id"])
        .agg(pl.col("fatalities").sum().alias("ged_sb_sum"))
        .collect()
    )
```

### FAQ
| Question | Answer |
|----------|--------|
| Why use `ExtractorManager` instead of a script? | Standard lifecycle, WandB tracking, uniform logging. |
| Can I skip WandB? | Yes—pass `wandb_notifications=False`. |
| Does `run()` return anything? | No—persist outputs; assign `self.data` for in-memory access. |
| Can extractors feed directly into modeling? | Yes—downstream modules can load processed parquet via `ModelPathManager`. |

### Suggested Logging
```python
logger.info("Starting GED extractor.")
logger.info(f"Raw rows: {self.raw.shape[0]}")
logger.info(f"Processed rows: {self.data.shape[0]}")
```
---