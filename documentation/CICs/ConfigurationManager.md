# Class Intent Contract: ConfigurationManager

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology), ADR-003 (Authority), ADR-009 (Boundary Contracts)

---

## 1. Purpose

Manages configuration loading, merging, validation, and runtime updates for
model pipelines. Centralizes five configuration sources into a single merged
dict with a defined priority ordering. Provides a dict-like interface
(`__getitem__`, `__setitem__`, `__contains__`, `__delitem__`, `get`, `keys`,
`values`, `items`) for convenient access to the merged configuration.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** load configuration files from disk. File loading belongs to
  `managers/configuration/script_config.load_config_from_script()`, which executes the
  named method out of a `config_*.py` script via `importlib` (#433). `ModelManager` and
  both ensemble managers expose it as a thin protected `_load_config()`; the
  implementation deliberately sits outside the manager hierarchy, so loading a config
  requires inheriting nothing.
- Does **not** own the configuration file format or schema. Each config file
  (`config_deployment.py`, `config_hyperparameters.py`, etc.) defines its own
  structure.
- Does **not** enforce immutability after validation. The runtime config can
  be modified at any time via `add_config()` or `__setitem__` without
  re-validation.
- Does **not** perform model-specific validation. It validates the universal
  pipeline contract keys only.
- Does **not** persist configuration to disk. It is an in-memory merge layer.
- Does **not** own WandB configuration. It receives sweep configs but does
  not interact with WandB directly (except passing `wandb_module` for error
  alerting).

---

## 3. Responsibilities and Guarantees

- Guarantees a deterministic 5-source priority merge:
  `partition_dict < hyperparameters < deployment < meta < runtime` (later
  sources override earlier ones for duplicate keys).
- Guarantees that `get_combined_config()` normalizes task keys
  (`regression_targets`, `classification_targets`, `*_metrics`) to lists.
- Guarantees that `get_combined_config()` synthesises a `"targets"` key from
  `regression_targets + classification_targets` when no explicit `"targets"`
  key is present (backward compatibility with legacy models).
- Guarantees that `_runtime_config["timestamp"]` is set at construction time
  in `YYYYMMDD_HHMMSS` format.
- Guarantees that `update_for_single_run()` applies CLI args (`run_type`,
  `eval_type`, `sweep`) to the runtime config.
- Guarantees that `update_for_sweep_run()` merges WandB sweep parameters into
  the runtime config with highest priority.
- Guarantees that `_apply_timestep_override()` recalculates the forecasting
  partition when `args.override_timestep` is specified.
- Guarantees that `__delitem__` only removes keys from `_runtime_config`, not
  from other config sources. Raises `KeyError` if the key is not in runtime
  config.
- Guarantees that `get_combined_config()` returns a fresh dict on each call
  (no caching, no shared references).

---

## 4. Inputs and Assumptions

- `config_hyperparameters: Dict` -- model hyperparameters. May be `None`
  (treated as empty dict).
- `config_deployment: Dict` -- deployment settings. May be `None`.
- `config_meta: Dict` -- project metadata. May be `None`.
- `partition_dict: Optional[Dict]` -- time partition definitions. May be
  `None` (treated as empty dict). Expected structure:
  `{"calibration": {"train": (start, end), "test": (start, end)}, ...}`.
- `config_sweep: Optional[Dict]` -- WandB sweep configuration. May be `None`
  (disables sweep functionality).
- `args: ForecastingModelArgs` -- required by `update_for_single_run()` and
  `update_for_sweep_run()`. Must have `run_type`, `eval_type`, `sweep`, and
  optionally `override_timestep` attributes.
- Assumes that the caller (typically `ModelManager`) passes complete config
  dicts. `ConfigurationManager` does not validate completeness at construction
  time.

---

## 5. Outputs and Side Effects

- `get_combined_config()` returns a merged dict containing all keys from all
  five sources, with normalization applied to task keys.
- `get_combined_sweep_config()` returns the same merge with the same
  normalization (used during sweep runs).
- `update_for_single_run()` mutates `_runtime_config` with args values.
- `update_for_sweep_run()` mutates `_runtime_config` with WandB sweep values.
- `add_config()` mutates `_runtime_config` with arbitrary key-value pairs.
- `_apply_timestep_override()` mutates `_runtime_config["forecasting"]` with
  recalculated partition ranges and logs the override.
- No disk I/O. No WandB calls (except error alerting via passed module).

---

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| Key not found in merged config via `__getitem__` | `KeyError` |
| Key not found in runtime config via `__delitem__` | `KeyError` |
| `override_timestep` specified but `"steps"` missing from config | `logger.warning`, override skipped |
| `None` passed for required config source | Treated as empty dict (no error) |

`ConfigurationManager` is relatively permissive by design -- it is a merge
layer, not a validator. Structural validation of the merged config is the
responsibility of `CoreConfigSniffer`.

---

## 7. Boundaries and Interactions

```
Config files (loaded by ModelManager via importlib)
    |
    v
ConfigurationManager
    |-- partition_dict             (lowest priority)
    |-- config_hyperparameters
    |-- config_deployment
    |-- config_meta
    |-- _runtime_config            (highest priority)
    |
    Consumers:
    |-- ModelManager.configs property
    |-- ForecastingModelManager (all pipeline stages)
    |-- EnsembleManager
    |-- CoreConfigSniffer (reads merged config)
    |-- WandBModule (receives merged config for run init)
```

- `ConfigurationManager` is instantiated inside `ModelManager.__init__()`.
- It is accessed via `ModelManager.configs` property, which calls
  `get_combined_config()` or `get_combined_sweep_config()` depending on
  `self._sweep`.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers import ConfigurationManager

config_mgr = ConfigurationManager(
    config_hyperparameters={"algorithm": "rf", "steps": [1, 2, 3]},
    config_deployment={"name": "purple_alien", "version": "1.0"},
    config_meta={"author": "VIEWS", "regression_targets": ["ged_sb"]},
    partition_dict={"calibration": {"train": (121, 396), "test": (397, 444)}},
)

# Update for run
args = ForecastingModelArgs.parse_args()
config_mgr.update_for_single_run(args)

# Dict-like access
config = config_mgr.get_combined_config()
print(config["algorithm"])  # "rf"

# Direct access
print(config_mgr["name"])   # "purple_alien"
config_mgr["custom"] = 42
print(config_mgr["custom"]) # 42
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: assuming config is immutable after update_for_single_run
config_mgr.update_for_single_run(args)
config_mgr["run_type"] = "forecasting"  # silently changes run_type
# -> No error raised, but now config contradicts the args

# WRONG: deleting a key from hyperparameters via __delitem__
del config_mgr["algorithm"]
# -> KeyError (not in _runtime_config, even though it's in merged config)

# WRONG: relying on get_combined_config() returning the same object
config1 = config_mgr.get_combined_config()
config1["hacked"] = True
config2 = config_mgr.get_combined_config()
assert "hacked" not in config2  # True -- fresh dict each time
```

---

## 10. Test Alignment

- `tests/test_managers/test_configuration.py` -- tests for priority merge
  ordering, dict-like interface, runtime config updates, timestep override,
  task key normalization, and `targets` synthesis.

---

## 11. Evolution Notes

- The dict-like interface (`__getitem__`, `__setitem__`, etc.) was added to
  allow `ConfigurationManager` to be used interchangeably where raw dicts
  were previously expected.
- Task key normalization (forcing `regression_targets`, etc., to lists) and
  `targets` synthesis were added to support the transition from the legacy
  single-`targets` key to the split `regression_targets` /
  `classification_targets` convention.
- `get_combined_sweep_config()` was added as a parallel merge path for sweep
  runs, though it currently performs identical normalization to
  `get_combined_config()`.

---

## 12. Known Deviations

- **No immutability enforcement after validation:** `_runtime_config` can be
  modified at any time via `add_config()`, `__setitem__`, or
  `update_for_sweep_run()` without triggering re-validation. This means a
  validated config can be silently corrupted after validation.
- **Priority ordering is implicit (not declarative):** The merge order is
  encoded as sequential `dict.update()` calls in `_get_raw_combined_config()`.
  There is no schema or declarative priority specification that can be
  inspected or tested independently.
- **`get_combined_sweep_config()` duplicates `get_combined_config()`:** The
  two methods contain identical normalization logic. Any change to one must
  be mirrored in the other, which is a maintenance risk.
- **`None` configs silently become empty dicts:** Passing `None` for a
  required config source (e.g., `config_hyperparameters=None`) does not raise
  an error. The merged config will simply lack those keys, which may cause
  failures much later in the pipeline (at sniffing or execution time).

---

## End of Contract

This document defines the **intended meaning** of `ConfigurationManager`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
