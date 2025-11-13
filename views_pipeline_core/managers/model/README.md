# VIEWS Pipeline Core: Model Management

File: `views_pipeline_core/managers/model/model.py`  
Classes documented:  
- `ModelPathManager` (path & filesystem orchestration)  
- `ModelManager` (abstract base pipeline controller)  
- `ForecastingModelManager` (forecast-specific implementation scaffold)

---

## 1. ModelPathManager

### Purpose
Centralizes all filesystem path resolution for a model (or ensemble/preprocessor) in the ViEWS Pipeline Core. Enforces naming convention (`adjective_noun`), discovers project root via a marker file (`.gitignore`), initializes directory and script paths, and provides helper utilities for artifact discovery and queryset import.

### Key Responsibilities
- Validate and process model name or infer it from a deeper path (e.g. `.../models/purple_alien/main.py`).
- Locate project root by ascending until marker file found.
- Build standardized directory tree:
  - `artifacts/`
  - `configs/`
  - `data/raw`, `data/generated`, `data/processed`
  - `reports/`
  - `notebooks/` (model-only)
  - `logs/`
- Track required script paths (`config_deployment.py`, `config_hyperparameters.py`, etc.).
- Provide latest artifact retrieval based on timestamp naming pattern.
- Load queryset config dynamically (if present) via `generate()` method.

### Naming Constraints
Model name must match: `^[a-z]+_[a-z]+$` (two lowercase tokens separated by underscore). Examples: `purple_alien`, `silver_stag`. Fails early on invalid names if `validate=True`.

### Selected Methods

| Method | Summary |
|--------|---------|
| `get_root()` | Lazy global project root resolution. |
| `get_models()` | Path to the models base directory. |
| `check_if_model_dir_exists(name)` | Returns boolean existence. |
| `get_model_name_from_path(path)` | Extracts name from any path containing one of: models/ensembles/preprocessors/postprocessors/extractors/apis. |
| `validate_model_name(name)` | Regex enforcement. |
| `find_project_root(current_path, marker)` | Ascends until marker file found. |
| `_get_model_dir()` | Resolves and validates model-specific directory. |
| `_initialize_directories()` | Populates directory attributes; warns if missing (when validating). |
| `_initialize_scripts()` | Populates script paths and queryset references. |
| `get_latest_model_artifact_path(run_type)` | Finds latest artifact matching `{run_type}_model_{YYYYMMDD_HHMMSS}.*`. |
| `get_queryset()` | Dynamically imports and executes queryset config if present. |
| `_get_raw_data_file_paths(run_type)` | Returns list of raw dataset parquet files. |
| `_get_generated_predictions_data_file_paths(run_type)` | Returns generated prediction file paths. |
| `_get_eval_file_paths(run_type, conflict_type)` | Returns evaluation metric files. |
| `view_directories()` / `view_scripts()` | Human-readable console views. |
| `get_directories()` / `get_scripts()` | Structured dict export (for JSON/logging). |

### Artifact Pattern
```
{run_type}_model_{YYYYMMDD_HHMMSS}.{ext}
```
Supported extensions include common ML formats (`.pt`, `.pkl`, `.json`, `.bst`, `.onnx`, etc.).

### Usage Example
```python
mpm = ModelPathManager("purple_alien")
artifact = mpm.get_latest_model_artifact_path("calibration")
queryset_spec = mpm.get_queryset()  # May return None if missing
mpm.view_directories()
```

---

## 2. ModelManager (Abstract Base)

### Purpose
Defines the shared interface and orchestration mechanics for running a full model pipeline: configuration assembly, data fetching, artifact management, evaluation, forecasting, reporting, and WandB integration.

This class is not directly instantiated; extend it to implement concrete training/evaluation routines (e.g. random forest, neural net, gradient boosting).

### Core Attributes

| Attribute | Description |
|-----------|-------------|
| `_model_path` | Instance of `ModelPathManager`. |
| `_wandb_module` | Wrapper for WandB runs, alerts, artifacts. |
| `_config_manager` | Merges hyperparameters, deployment, meta, partitions, runtime overrides. |
| `_data_loader` | `ViewsDataLoader` instance if queryset is available. |
| `_args` | Parsed `ForecastingModelArgs` once execution begins. |
| `_project` | Derived WandB project name (`{model_name}_{run_type}` or sweep variant). |
| `_eval_type` | Evaluation scope string (`standard`, `long`, `complete`, `live`). |
| `_pred_store_name` | Prediction store run identifier (optional). |

### Execution Flow (Single Run)
1. `execute_single_run(args)`
2. Login to WandB.
3. Merge/update configuration (adds `eval_type`, timestamp).
4. Data fetch via `_execute_data_fetching()`.
5. Conditional stages:
   - `_execute_model_training()`
   - `_execute_model_evaluation()`
   - `_execute_model_forecasting()`
   - `_execute_forecast_reporting()`
   - `_execute_evaluation_reporting()`

### Execution Flow (Sweep)
1. `execute_sweep_run(args)` sets `_sweep=True`.
2. Data fetched once.
3. WandB sweep created from `config_sweep.py`.
4. Agent triggers `_execute_model_tasks()` for each hyperparameter set.
5. `_evaluate_sweep()` invoked using in-memory model object.

### Abstract Methods (Must Implement in Subclass)
| Method | Contract |
|--------|----------|
| `_train_model_artifact()` | Train and persist model artifact. |
| `_evaluate_model_artifact(eval_type, artifact_name)` | Produce evaluation prediction sequences. |
| `_forecast_model_artifact(artifact_name)` | Produce future horizon predictions. |
| `_evaluate_sweep(eval_type, model)` | Evaluate in-memory model in sweep context. |

### Internal Stage Methods

| Method | Role |
|--------|------|
| `_execute_data_fetching()` | Fetch or load partitioned viewser data; create WandB run. |
| `_execute_model_training()` | Train model + save artifact + log hyperparameters. |
| `_execute_model_evaluation()` | Generate multi-sequence predictions + validate + metrics. |
| `_execute_model_forecasting()` | Generate future predictions + undo transformations hack + save. |
| `_execute_forecast_reporting()` | Build forecast HTML report using templates. |
| `_execute_evaluation_reporting()` | Build evaluation HTML report using metrics from latest WandB run. |
| `_evaluate_prediction_dataframe(df_predictions, eval_type, ensemble)` | Compute metrics (step / month / time-series). |
| `_save_evaluations(...)` | Save metric parquet files and log them. |
| `_save_predictions(...)` | Save predictions locally + optionally to prediction store. |
| `_save_model_artifact(run_type)` | Publish latest artifact to WandB. |

### Metrics Handling
Uses `EvaluationManager` (external module) to compute:
- Step-wise metrics (per horizon)
- Time-series metrics (per sequence)
- Month-wise metrics (temporal slices)

Conflict type (`sb`, `os`, `ns`) auto-detected from target variable tokens for grouping and filenames.

### Temporary Transformation Undo
Forecasting stage applies:
```python
forecast_transformation_module.undo_all_transformations()
```
To revert transformed targets (e.g. `ln_ged_sb`) before final saving (pending architectural ADR refinement).

### Error Handling
Raises specialized exceptions with optional WandB alerting:
- `DataFetchException`
- `ModelTrainingException`
- `ModelEvaluationException`
- `ModelForecastingException`
- `PipelineException`

### Representation
- `__repr__`: Verbose multi-line summary (including state flags).
- `__str__`: Concise one-line summary.

### Usage Skeleton
```python
class MyForecastingManager(ForecastingModelManager):
    def _train_model_artifact(self):
        # implement training
        ...
    def _evaluate_model_artifact(self, eval_type, artifact_name):
        # return list of prediction DataFrames
        ...
    def _forecast_model_artifact(self, artifact_name):
        # return future prediction DataFrame
        ...
    def _evaluate_sweep(self, eval_type, model):
        # return list of prediction DataFrames
        ...

mp = ModelPathManager("purple_alien")
mgr = MyForecastingManager(mp, wandb_notifications=True)
args = ForecastingModelArgs.parse_args()
mgr.execute_single_run(args)
```

---

## 3. ForecastingModelManager

### Purpose
Specialized subclass configuring the full forecasting lifecycle: data ingestion, training, multi-sequence evaluation, future horizon inference, reporting. Leaves algorithm specifics to further subclassing or mixins.

### Added Functionality
| Method | Purpose |
|--------|---------|
| `_get_conflict_type(target)` | Extracts conflict class token from target names (`sb`, `os`, `ns`). |
| `_resolve_evaluation_sequence_number(eval_type)` | Determines number of evaluation sequences (e.g. `standard` → 12). |
| `dataset_class(loa)` | Returns dataset constructor (`CMDataset` or `PGMDataset`) via partial. |

### Evaluation Types
| Type | Meaning | Sequences |
|------|---------|-----------|
| `standard` | 1-year horizon evaluation | 12 |
| `long` | 3-year horizon evaluation | 36 |
| `complete` | Full span (dynamic) | None |
| `live` | Current-year dynamic evaluation | 12 |

### Forecast Reporting
- Gathers historical raw dataset (model or ensemble).
- Loads latest forecast predictions (post-transformation undo).
- Uses `ForecastReportTemplate` for HTML generation.

### Evaluation Reporting
- Resolves latest run (`get_latest_run`) for model and run type.
- Builds per-target evaluation report using `EvaluationReportTemplate`.
- Saves report locally and triggers WandB alert.

### Typical Forecast Run
```python
mp = ModelPathManager("purple_alien")
manager = ForecastingModelManager(mp, wandb_notifications=True)
args = ForecastingModelArgs(run_type="forecasting", train=True, evaluate=True, forecast=True, report=True)
manager.execute_single_run(args)
```

### Common Extension Points
- Override training to plug in custom ML framework (e.g. PyTorch, XGBoost).
- Customize prediction saving (e.g. add probabilistic sample columns).
- Extend reporting templates to include custom uncertainty visualizations.

---

## Best Practices

| Concern | Recommendation |
|---------|----------------|
| Reproducibility | Persist config files alongside artifacts. |
| Artifact Naming | Do not alter filename pattern; downstream tooling depends on it. |
| Transformation Undo | Track transformation logic externally until ADR standardization. |
| Metrics Scope | Choose eval_type matching use-case (short-term vs medium-term evaluation). |
| Forecast Integrity | Never incorporate future ground truth when building features for forecasting stage. |
| Error Surfacing | Allow exceptions to propagate; avoid silent failure paths. |
| Parallel Validation | Current threadpool usage acceptable; optimize only if sequences grow large. |

---

## Known Temporary Hacks
- Transformation module applied inside forecasting stage for undoing log transforms (will be replaced when standardized raw vs transformed storage ADR is enforced).

---

## FAQ

| Question | Answer |
|----------|--------|
| Can I use ModelManager directly? | No, it’s abstract; implement subclass methods. |
| Do I need a queryset file? | Yes for models; ensembles may not have one. |
| How are artifacts chosen for evaluation? | By filename timestamp sorting (latest first). |
| How to add a new eval type? | Extend `_resolve_evaluation_sequence_number` logic. |
| What if WandB is unavailable? | Wrap initialization; module logs errors but pipeline can proceed with reduced functionality. |

---

## References
- Configuration merging: `ConfigurationManager`
- Data ingestion: `ViewsDataLoader`
- Validation: `validate_prediction_dataframe`
- Transformation handling: `DatasetTransformationModule`
- Metrics: `EvaluationManager` (external dependency from `views-evaluation`)
- Alerting & artifacts: `WandBModule`
- ADR coverage: evaluation scope, artifact naming, reporting templates (internal repo docs)
---