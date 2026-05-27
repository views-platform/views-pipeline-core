# Class Intent Contract: ModelPathManager

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-07
**File:** `views_pipeline_core/data/model_path.py` (canonical); re-exported from `managers/model/model.py` for backward compatibility
**Related ADRs:** ADR-001 (Ontology), ADR-002 (Topology), ADR-009 (Boundary Contracts), ADR-045 (Pipeline Stage Architecture, E6)

---

## 1. Purpose

Centralized path resolution for all model artifacts, configurations, data,
scripts, and reports within the ViEWS Pipeline. Provides a single source of
truth for where every file lives, validates model names against the
`adjective_noun` convention, and discovers the project root by walking up the
directory tree to find a `.gitignore` marker. Serves as the base class for
`EnsemblePathManager`, `ExtractorPathManager`, and `PostprocessorPathManager`.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** create directories. It resolves and validates paths but does
  not call `mkdir()` (except downstream consumers may do so).
- Does **not** load, parse, or execute configuration files. Config loading is
  `ModelManager`'s responsibility via `importlib`.
- Does **not** manage model artifacts (training, saving, loading). That is
  `ForecastingModelManager`'s responsibility.
- Does **not** enforce that all resolved paths actually exist when
  `validate=False`. Non-validated mode is for model creation workflows.
- Does **not** resolve paths for the pipeline-core package itself. It resolves
  paths within the `views-platform` monorepo (models/, ensembles/, etc.).

---

## 3. Responsibilities and Guarantees

- Guarantees that `model_name` follows the lowercase `adjective_noun` regex
  pattern (`^[a-z]+_[a-z]+$`). Raises `ValueError` if the name is invalid.
- Guarantees that the project root is discovered by searching upward for a
  `.gitignore` marker file. Raises `FileNotFoundError` if the marker is not
  found.
- Guarantees that class-level state (`_root`, `_models` via `get_models()`)
  is initialized lazily on first access and shared across all instances of the
  same class.
- Guarantees that when `validate=True`, the model directory must exist or
  `FileNotFoundError` is raised.
- Guarantees that `get_latest_model_artifact_path(run_type)` returns the most
  recent artifact file by timestamp-sorted filename. Raises
  `FileNotFoundError` if no artifacts exist.
- Guarantees that `resolve_artifact_path(run_type, artifact_name)` returns the
  named artifact when `artifact_name` is provided (raising `FileNotFoundError`
  if missing), or delegates to `get_latest_model_artifact_path(run_type)` when
  `artifact_name` is `None`.
- Guarantees that `get_model_name_from_path(path)` extracts the model name
  from a filesystem path by locating exactly one of the valid parent
  directories (`models`, `ensembles`, `preprocessors`, `postprocessors`,
  `extractors`, `apis`).
- Guarantees that `.env` is loaded from the project root at instance
  construction via `dotenv.load_dotenv()`.
- Guarantees that each instance has a unique hash based on
  `(model_name, validate, target)`.

---

## 4. Inputs and Assumptions

- `model_path: Union[str, Path]` -- either a model name string (e.g.,
  `"purple_alien"`) or a filesystem path containing a valid parent directory
  and model name.
- `validate: bool` -- when `True` (default), all resolved directories must
  exist. When `False`, paths are constructed without existence checks.
- Assumes the project root contains a `.gitignore` file. If running from
  outside the monorepo, `pyprojroot.here()` is used as a fallback starting
  point.
- Assumes the model directory structure follows the convention:
  `{root}/models/{model_name}/artifacts/`, `configs/`, `data/`, etc.

---

## 5. Outputs and Side Effects

- Sets instance attributes for all standard directories: `root`, `models`,
  `model_dir`, `artifacts`, `configs`, `data`, `data_generated`,
  `data_processed`, `data_raw`, `reports`, `notebooks`, `logging`.
- Sets `scripts` list with paths to expected config and entry-point files.
- Sets `queryset_path` for model instances (not ensembles).
- Loads `.env` from project root (side effect: populates `os.environ`).
- Increments class-level `__instances__` counter.
- `get_queryset()` dynamically imports and executes the queryset config
  module via `importlib`.

---

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| Invalid model name (not `adjective_noun`) | `ValueError` immediately |
| Model directory does not exist (`validate=True`) | `FileNotFoundError` |
| `.gitignore` marker not found in directory hierarchy | `FileNotFoundError` |
| No artifacts found for `run_type` | `FileNotFoundError` from `get_latest_model_artifact_path()` |
| Path contains zero or multiple valid parent dirs | `get_model_name_from_path()` returns `None` (logged at DEBUG) |
| Subdirectory missing (`validate=True`) | `logger.warning`, attribute set to `None` |
| Queryset import fails | `logger.error`, returns `None` |

---

## 7. Boundaries and Interactions

```
ModelPathManager
    |-- pyprojroot.here()          (fallback root discovery)
    |-- dotenv.load_dotenv()       (environment loading)
    |-- importlib                  (queryset loading)
    |
    Subclasses:
    |-- EnsemblePathManager        (_target = "ensemble")
    |-- ExtractorPathManager       (_target = "extractor")
    |-- PostprocessorPathManager   (_target = "postprocessor")
    |
    Consumers:
    |-- ModelManager / ForecastingModelManager
    |-- EnsembleManager
    |-- ViewsDataLoader
    |-- PredictionIOManager
```

- `ModelPathManager` is consumed by `ModelManager.__init__()` as the
  `model_path` argument.
- Subclasses override `_target` and `_initialize_class_paths()` to point
  `_models` at the correct directory (e.g., `ensembles/` instead of
  `models/`).

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.model import ModelPathManager

# Existing model with validation
path_mgr = ModelPathManager("purple_alien")
print(path_mgr.artifacts)  # /path/to/models/purple_alien/artifacts

# New model without validation
path_mgr = ModelPathManager("new_model", validate=False)

# From a script path
path_mgr = ModelPathManager(Path(__file__))

# Get latest artifact
artifact = path_mgr.get_latest_model_artifact_path("calibration")

# Get queryset
queryset = path_mgr.get_queryset()
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: CamelCase name
ModelPathManager("PurpleAlien")
# -> ValueError

# WRONG: single word name
ModelPathManager("purple")
# -> ValueError

# WRONG: accessing data_raw on an ensemble
ens = EnsemblePathManager("mighty_coalition")
ens.data_raw  # -> AttributeError (ensembles don't have data_raw)

# WRONG: assuming _root is set without any instance created
ModelPathManager._root  # -> None (not yet initialized)
```

---

## 10. Test Alignment

- `tests/test_managers/test_model_path.py` -- tests for name validation,
  path extraction from filesystem paths, root discovery, directory
  initialization, artifact file resolution, and `resolve_artifact_path()`
  (named artifact resolution, missing artifact error, timestamp extraction).

---

## 11. Evolution Notes

- `ModelPathManager` was originally the only path manager. `EnsemblePathManager`,
  `ExtractorPathManager`, and `PostprocessorPathManager` were added as
  subclasses that override `_target` to resolve to `ensembles/`, `extractors/`,
  and `postprocessors/` respectively.
- The `valid_parents` set was expanded over time to include `postprocessors`,
  `extractors`, and `apis`.
- Instance hashing via `generate_hash()` was added for deduplication in
  multi-manager scenarios.

---

## 12. Known Deviations

- **Class-level state set by first instance (R5):** `_root` is set once by
  the first instance created (via `_initialize_class_paths`). If the first
  instance is created from an unusual working directory, all subsequent
  instances inherit the wrong root. There is no reset mechanism.
- **`.gitignore` as root marker is fragile:** Any `.gitignore` file in a
  parent directory will be mistaken for the project root. This fails in
  monorepo setups where multiple `.gitignore` files exist at different levels.
  `pyprojroot.here()` is used as the fallback but has the same fragility.
- **`_build_absolute_directory` returns mixed types:** When `validate=True`
  and a `.py` script path does not exist, the method returns the bare filename
  string instead of `None` or `Path`. This inconsistency can cause downstream
  `TypeError` when callers expect `Path`.
- **`get_queryset()` swallows import errors:** If the queryset module fails
  to import, the error is logged but `None` is returned. This can mask
  configuration errors in model setup.

---

## End of Contract

This document defines the **intended meaning** of `ModelPathManager`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
