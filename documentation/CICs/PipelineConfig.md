# Class Intent Contract: PipelineConfig

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology), ADR-009 (Boundary Contracts)

---

## 1. Purpose

Provides pipeline-wide configuration constants as a module-level singleton.
Exposes the canonical dataframe format, organization name, package name,
current version, and version range constraint. The singleton is instantiated
at module import time as `PipelineConfig` and is the only instance of
`_PipelineConfigImpl` that should ever exist.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** manage model-specific configuration. That is
  `ConfigurationManager`'s responsibility.
- Does **not** manage environment variables. That is `dotenv` /
  `ModelPathManager`'s responsibility.
- Does **not** manage WandB configuration or project settings.
- Does **not** own path resolution. That is `ModelPathManager`'s
  responsibility.
- Does **not** provide a registry of models, ensembles, or other pipeline
  components.
- Is **not** intended to be instantiated directly. The module-level
  `PipelineConfig` instance is the only access point.

---

## 3. Responsibilities and Guarantees

- Guarantees that `dataframe_format` always starts with a period (`.`).
  Setting a value that does not match `^\..*` raises `ValueError`.
- Guarantees that `organization_name` returns `"views"`.
- Guarantees that `package_name` returns `"views-pipeline-core"`.
- Guarantees that `current_version` is read from `pyproject.toml` at the
  package's install location (`tool.poetry.version`). The value is cached
  after first access.
- Guarantees that `views_pipeline_core_version_range` returns a semver range
  string `">={latest_github_release}, <3.0.0"` by querying
  `PackageManager.get_latest_release_version_from_github()`.
- Guarantees that only one instance of `_PipelineConfigImpl` exists (module-
  level singleton pattern).

---

## 4. Inputs and Assumptions

- Assumes that `pyproject.toml` exists at `Path(__file__).parent.parent.parent
  / "pyproject.toml"` relative to the `pipeline.py` source file. If the file
  does not exist, `current_version` returns an empty string `""`.
- Assumes that `pyproject.toml` follows the Poetry format with
  `tool.poetry.version`.
- Assumes that `PackageManager.get_latest_release_version_from_github()` is
  callable and returns a valid semver string when
  `views_pipeline_core_version_range` is accessed.
- The `dataframe_format` default is `".parquet"`. Callers may change it at
  runtime via the setter, but the value must always start with `"."`.

---

## 5. Outputs and Side Effects

- `dataframe_format` -- returns the current format string (default
  `".parquet"`). Setting it validates format and updates the internal state.
- `organization_name` -- returns `"views"` (read-only).
- `package_name` -- returns `"views-pipeline-core"` (read-only).
- `current_version` -- returns the version string from `pyproject.toml`.
  First access triggers file I/O via `toml.load()`. Subsequent accesses
  return the cached value.
- `views_pipeline_core_version_range` -- triggers a lazy import of
  `PackageManager` and a GitHub API call on every access (not cached).
- `logger.debug` messages emitted when `dataframe_format` is read or set.

---

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| `dataframe_format` set to value not starting with `"."` | `ValueError` immediately |
| `pyproject.toml` not found | `current_version` returns `""` (silent) |
| `pyproject.toml` missing `tool.poetry.version` path | `current_version` returns `""` (silent) |
| `PackageManager` GitHub API call fails | Exception propagates from `views_pipeline_core_version_range` property |
| Attempting to instantiate `_PipelineConfigImpl` directly | No error (no enforcement), but violates intent |

---

## 7. Boundaries and Interactions

```
_PipelineConfigImpl (singleton)
    |
    Module-level: PipelineConfig = _PipelineConfigImpl()
    |
    Properties:
    |-- dataframe_format           (mutable, validated)
    |-- organization_name          (read-only)
    |-- package_name               (read-only)
    |-- current_version            (lazy-loaded from pyproject.toml)
    |-- views_pipeline_core_version_range  (lazy import of PackageManager)
    |
    Consumers:
    |-- ModelPathManager           (dataframe_format for file extensions)
    |-- ModelManager               (ascii splash, version display)
    |-- ForecastingModelManager    (dataframe_format for prediction files)
    |-- EnsembleManager            (dataframe_format for prediction files)
    |-- File utilities             (dataframe_format for read/write)
```

- `PipelineConfig` is a Layer 1 (configs) component. The
  `views_pipeline_core_version_range` property creates a topology violation
  by importing `PackageManager` (Layer 7, managers/package) from Layer 1.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.configs.pipeline import PipelineConfig

# Read defaults
fmt = PipelineConfig.dataframe_format   # ".parquet"
org = PipelineConfig.organization_name  # "views"
ver = PipelineConfig.current_version    # e.g., "2.4.1"

# Change format (e.g., for testing)
PipelineConfig.dataframe_format = ".csv"

# Use in file paths
file_path = data_dir / f"predictions{PipelineConfig.dataframe_format}"
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: instantiating directly
my_config = _PipelineConfigImpl()
# -> Creates a second instance, violating singleton intent

# WRONG: setting format without period
PipelineConfig.dataframe_format = "parquet"
# -> ValueError

# WRONG: assuming current_version is always populated
assert PipelineConfig.current_version != ""
# -> May fail if pyproject.toml is not at the expected path

# WRONG: calling views_pipeline_core_version_range in a tight loop
for _ in range(100):
    r = PipelineConfig.views_pipeline_core_version_range
# -> 100 GitHub API calls (not cached)
```

---

## 10. Test Alignment

- `tests/test_configs/test_pipeline_config.py` -- tests for singleton
  identity, format validation, default values, version reading, and
  property access.

---

## 11. Evolution Notes

- `PipelineConfig` was originally a plain module with global variables. It was
  refactored to a singleton class to support validated property setters (e.g.,
  `dataframe_format` validation).
- `current_version` was originally read eagerly at import time. It was changed
  to lazy loading (first-access caching) to avoid import-time file I/O issues
  when the working directory is wrong.
- `views_pipeline_core_version_range` was added to support version pinning in
  generated `pyproject.toml` files for new models.

---

## 12. Known Deviations

- **Lazy import of `PackageManager` creates topology violation (R8):** The
  `views_pipeline_core_version_range` property imports
  `views_pipeline_core.managers.package.PackageManager` at access time. This
  creates a Layer 1 to Layer 7 dependency, violating the intended layering
  where configs should have no upward dependencies. This was introduced as a
  convenience for `make_new_model.py` workflows and should be extracted to a
  separate utility.
- **`pyproject.toml` path is relative to source file:** The path
  `Path(__file__).parent.parent.parent / "pyproject.toml"` assumes a
  specific directory layout. If the package is installed in site-packages
  (rather than developed in-tree), this path may not resolve correctly, and
  `current_version` will silently return `""`.
- **Singleton is not enforced:** There is no `__new__` override or metaclass
  preventing additional instances of `_PipelineConfigImpl`. The singleton
  contract is purely conventional (documented in the docstring).
- **`views_pipeline_core_version_range` is not cached:** Every access triggers
  a fresh `PackageManager.get_latest_release_version_from_github()` call,
  which involves GitHub API I/O. This is a latency and rate-limiting risk.
- **`dataframe_format` is globally mutable:** Any code anywhere can change the
  format via the setter, affecting all consumers. There is no scoping or
  context management.

---

## End of Contract

This document defines the **intended meaning** of `PipelineConfig`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
