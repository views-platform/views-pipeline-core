# Postprocessor Management

File: `views_pipeline_core/managers/postprocessor/postprocessor.py`  
Classes:
- `PostprocessorPathManager`
- `PostprocessorManager` (abstract base for concrete postprocessors)

---

## PostprocessorPathManager

### Purpose
Path orchestration for a postprocessor component (downstream transformation of already produced model / ensemble outputs). Extends `ModelPathManager` with `_target="postprocessor"`.

### Initialization
```python
ppm = PostprocessorPathManager("spatial_smoothing")
```
Creates (if absent):
- `<root>/postprocessors/spatial_smoothing/data/raw` (input staging)
- Standard inherited directories: `data/generated`, `data/processed`, `artifacts`, `reports`, `logs`, `configs`

Adds:
- `queryset_path = configs/config_queryset.py` (optional; used if the postprocessor replays transformations or needs variable metadata)

### Key Attributes (inherited + extended)
| Attribute | Description |
|-----------|-------------|
| `model_name` | Postprocessor name (`^[a-z]+_[a-z]+$`) |
| `data_raw` | Raw intermediate input storage |
| `queryset_path` | Optional queryset spec for reference |

### Typical Use
```python
ppm.view_directories()
print(ppm.queryset_path.exists())
```

---

## PostprocessorManager

### Purpose
Defines a four-stage lifecycle for postprocessing previously generated predictions or features:
1. `_read()`      → load inputs (e.g., parquet predictions, evaluation outputs)
2. `_transform()` → apply deterministic transformations (e.g., smoothing, scaling, enrichment)
3. `_validate()`  → structural / statistical checks (schema, nulls, bounds, shape)
4. `_save()`      → persist outputs (e.g., new parquet, artifact logging)

Wrapped in a WandB run if `wandb_notifications=True`.

### Constructor
```python
manager = MyPostprocessor(PostprocessorPathManager("spatial_smoothing"), wandb_notifications=True)
```
Internally calls `ModelManager` base with `use_prediction_store=False`.

### Abstract Methods (must implement)
| Method | Expected Action |
|--------|------------------|
| `_read()` | Populate in-memory raw data objects |
| `_transform()` | Produce processed result |
| `_validate()` | Raise on failure; silent pass if OK |
| `_save()` | Write processed artifacts + optionally log to WandB |

### run()
```python
def run(self, args: Namespace):
    self._read()
    self._transform()
    self._validate()
    self._save()
```
Encapsulated in:
```python
with wandb.init(project=f"{configs['name']}_postprocessor", job_type="postprocessor_run"):
    ...
```
On success: sends an INFO WandB alert.  
On exception: logs error and raises `PipelineException`; run finalized in `finally`.

### WandB Integration
Project naming: `<postprocessor_name>_postprocessor`  
Alert example: “Postprocessing run for spatial_smoothing complete.”

### Error Handling
Any unhandled exception inside `run()` becomes a `PipelineException` with message forwarded to WandB (if enabled).

### Minimal Subclass Example
```python
class SmoothingPostprocessor(PostprocessorManager):
    def _read(self):
        self.df = pl.read_parquet(self._model_path.data_generated / "forecast_predictions.parquet")

    def _transform(self):
        self.output = (
            self.df
            .groupby(["month_id","entity_id"])
            .agg(pl.mean("prediction").alias("smoothed_prediction"))
        )

    def _validate(self):
        if self.output["smoothed_prediction"].is_null().any():
            raise ValueError("Null values after smoothing.")

    def _save(self):
        out = self._model_path.data_processed / "smoothed_predictions.parquet"
        self.output.write_parquet(out)
```

### Best Practices
- Keep transformations pure and reproducible (log parameters in `_save()`).
- Validate aggressively (schema, min/max, dtypes).
- Preserve original inputs (do not overwrite raw source).
- Add checksum or row counts to WandB summary.

### Common Pitfalls
| Pitfall | Mitigation |
|---------|------------|
| Silent schema drift | Explicit column assertion in `_validate()` |
| Overwriting raw data | Always write to `data/processed` or `data/generated` |
| Missing alert | Ensure `wandb_notifications=True` and no swallowed exceptions |

### FAQ
| Question | Answer |
|----------|--------|
| Can I chain multiple postprocessors? | Yes—feed output parquet from one into `_read()` of the next. |
| Is queryset required? | No—only if you need variable metadata. |
| Can I log extra artifacts? | Yes—use `self._wandb_module.log_artifact()` in `_save()`. |
---