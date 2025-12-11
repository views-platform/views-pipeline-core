# Configuration Manager

File: `views_pipeline_core/managers/configuration/configuration.py`  
Primary class: `ConfigurationManager`  
Related types:  
- `ForecastingModelArgs` (CLI / runtime arguments)  
- `validate_config` (model-level validation)  
- `ConfigurationException` (error signaling)  

## Overview

`ConfigurationManager` centralizes all configuration handling for a VIEWS model pipeline run. It ingests multiple configuration sources, merges them according to a strict priority order, applies runtime overrides (e.g. forecasting timestep adjustment), and validates the resulting structure before execution. This guarantees a single, authoritative configuration dictionary accessible throughout training, evaluation, forecasting, reporting, artifact logging, and WandB integration.

Priority order (lowest → highest):
1. Partition configuration (`partition_dict`)
2. Hyperparameters (`config_hyperparameters`)
3. Deployment configuration (`config_deployment`)
4. Meta configuration (`config_meta`)
5. Runtime configuration (`_runtime_config` – added dynamically, always highest)

If duplicate keys appear, later sources overwrite earlier values.

## Typical Use Cases

| Scenario | Capability |
|----------|------------|
| Single run preparation | `update_for_single_run(args)` |
| Hyperparameter sweep iteration | `update_for_sweep_run(wandb_config, args)` |
| Fast key lookup | Dict-style access: `config_mgr["algorithm"]` |
| Runtime override injection | `add_config({"run_type": "calibration"})` |
| Forecast horizon override | `_apply_timestep_override()` via CLI arg |
| Validation enforcement | Automatic inside update methods |

## Class: ConfigurationManager

### Initialization

```python
from views_pipeline_core.managers.configuration.configuration import ConfigurationManager

config_mgr = ConfigurationManager(
    config_hyperparameters={
        "algorithm": "random_forest",
        "hyperparameters": {"n_estimators": 250},
        "features": ["ln_pop", "lag_conflict"],
        "targets": ["ln_ged_sb"],
        "steps": list(range(1, 37))
    },
    config_deployment={
        "name": "purple_alien",
        "environment": "production",
        "version": "1.2.0"
    },
    config_meta={
        "description": "State-based conflict fatality forecasting",
        "author": "VIEWS Team",
        "metrics": ["mse", "mae", "r2"]
    },
    partition_dict={
        "calibration": {"train": (121, 396), "test": (397, 444)},
        "validation": {"train": (121, 444), "test": (445, 492)},
        "forecasting": {"train": (121, 528), "test": (529, 564)}
    }
)

print(config_mgr._runtime_config["timestamp"])  # e.g. '20251113_101522'
```

Notes:
- Any None input replaced with `{}` internally.
- No validation occurs at construction time.
- Timestamp always injected under key `timestamp`.

### Merged Configuration

Call:
```python
merged = config_mgr.get_combined_config()
```

Merging order ensures deterministic override behavior. Examples:
- Runtime key `name='override_model'` overwrites deployment key `name='purple_alien'`.
- Meta section can inject extra descriptive keys (e.g. `license`, `contact`) and they remain accessible downstream.

### Dict-like Interface

| Method | Purpose |
|--------|---------|
| `__getitem__(key)` | Strict access (raises `KeyError`) |
| `__setitem__(key, value)` | Add/override runtime value (no validation) |
| `__contains__(key)` | Membership test (`"algorithm" in config_mgr`) |
| `__delitem__(key)` | Remove from runtime layer only |
| `get(key, default=None)` | Safe access with fallback |
| `keys()/values()/items()` | Iterate merged configuration content |

Example:
```python
config_mgr["learning_rate"] = 0.005
print("learning_rate" in config_mgr)        # True
print(config_mgr.get("optimizer", "adam"))  # 'adam'
for k, v in config_mgr.items():
    print(k, v)
```

### Runtime Augmentation

Use `add_config()` for multi-key updates:
```python
config_mgr.add_config({
    "run_type": "calibration",
    "eval_type": "standard",
    "custom_flag": True
})
```

These instantly appear in downstream calls to `get_combined_config()`.

### Single Run Update

```python
from views_pipeline_core.cli.args import ForecastingModelArgs

args = ForecastingModelArgs(
    run_type="forecasting",
    eval_type="standard",
    sweep=False,
    override_timestep=530  # Optional
)

config_mgr.update_for_single_run(args)
final_conf = config_mgr.get_combined_config()
print(final_conf["run_type"])  # 'forecasting'
```

Behavior:
- Populates `run_type`, `eval_type`, `sweep`.
- Applies override to forecasting partition if `override_timestep` provided and `steps` present.
- Calls `validate_config()`; raises `ConfigurationException` on failure.

Override logic (if steps length = 36 and override_timestep=530):
- Forecasting train: `(121, 530)`
- Forecasting test: `(531, 567)`

### Sweep Run Update

```python
wandb_config = {
    "hyperparameters": {
        "n_estimators": 500,
        "max_depth": 8
    }
}

args = ForecastingModelArgs(run_type="calibration", sweep=True)
config_mgr.update_for_sweep_run(wandb_config, args)
print(config_mgr.get_combined_config()["hyperparameters"]["n_estimators"])  # 500
```

Notes:
- Sweep configuration overrides existing hyperparameters.
- Must pass `sweep=True`.
- Validation invoked post-merge; failures raise `ConfigurationException`.

### Timestep Override (Forecast Debugging)

Called internally when `args.override_timestep` present:
```python
# Log output example:
INFO: Applied timestep override: train=(121, 530), test=(531, 567)
```

Only affects forecasting partition. If `steps` missing, warns and skips.

### Validation

Triggered inside:
- `update_for_single_run`
- `update_for_sweep_run`

Checks performed by `validate_config()` include:
- Targets normalization (string → list coercion)
- Deployment status rules
- Basic structure for partitions
- Model naming and deprecation prevention

On failure:
- `ConfigurationException` thrown, optionally sends WandB alert if `wandb_module` passed.

### Error Handling

| Condition | Raised |
|-----------|--------|
| Missing runtime key deletion | `KeyError` (in `__delitem__`) |
| Access of absent key via `__getitem__` | `KeyError` |
| Invalid post-merge configuration | `ConfigurationException` |
| Override without steps | Logged warning (skip override) |

### Performance

- Merge operation: linear in total key count (negligible).
- Suitable for frequent access (e.g. each stage retrieving configs).
- No caching layer; direct recompute keeps result accurate.

### Thread Safety

- Read operations are safe.
- Write operations (`add_config`, `__setitem__`) mutate internal state and are not thread-safe.
- For multi-process sweeps, each agent holds its own instance—no cross-process contention.

## Best Practices

| Goal | Recommendation |
|------|----------------|
| Ensure reproducibility | Persist original source config files with artifacts |
| Reduce accidental overrides | Limit ad-hoc `__setitem__` calls; prefer structured CLI args |
| Facilitate debugging | Log merged config once after update |
| Avoid silent mis-spec | Validate immediately after any runtime bulk additions |
| Keep targets consistent | Provide targets as list in source configs to avoid coercion ambiguity |
| Forecast override hygiene | Use override only for testing; never in production automation |

## Common Pitfalls

| Pitfall | Mitigation |
|---------|------------|
| Overwriting core keys unintentionally | Namespace custom runtime keys (e.g. `custom.phase`) |
| Forgetting to call update before run | Enforce pipeline stage precondition |
| Missing steps during override | Ensure `steps` in hyperparameters config |
| Assuming deletion removes key globally | `__delitem__` only affects runtime layer |

## Integration Points

| Component | Use of Configuration |
|-----------|----------------------|
| DataLoader | Partition ranges | 
| Model Trainer | Hyperparameters / algorithm selection |
| Evaluation Module | Metrics, run_type, eval_type |
| WandBModule | Entity, name, version, sweep metadata |
| Reporting Module | Author, description, timestamp |
| Reconciliation | Steps (horizon size for forecast aggregation) |

## Example End-to-End

```python
# 1. Construct manager with base configs
config_mgr = ConfigurationManager(
    config_hyperparameters={"algorithm": "rf", "steps": [1,2,3], "targets": ["ln_ged_sb"]},
    config_deployment={"name": "purple_alien", "environment": "production", "version": "1.0.1"},
    config_meta={"description": "Fatality forecasting", "author": "VIEWS", "metrics": ["mse"]},
    partition_dict={"forecasting": {"train": (121, 528), "test": (529, 532)}}
)

# 2. Add runtime flags
config_mgr.add_config({"eval_type": "standard"})

# 3. Apply CLI args (override end timestep)
args = ForecastingModelArgs(run_type="forecasting", eval_type="standard", sweep=False, override_timestep=530)
config_mgr.update_for_single_run(args)

# 4. Retrieve final configuration
conf = config_mgr.get_combined_config()
print(conf["forecasting"])  # {'train': (121, 530), 'test': (531, 534)}
print(conf["timestamp"])
```

## Suggested Logging Pattern

Log merged configuration once per run for audit:
```python
logger.info("Merged configuration:")
for k, v in config_mgr.items():
    logger.info(f"{k}={v}")
```

Avoid logging sensitive items (tokens, secrets) if ever added to runtime config.

## FAQ

| Question | Answer |
|----------|--------|
| Can I modify hyperparameters after validation? | Yes, but you must re-validate manually—prefer modifying before update call. |
| Does `get_combined_config()` cache? | No; always recomputed for accuracy. |
| How are target strings handled? | Automatically coerced to list if a single string found in meta. |
| Can I run sweeps outside calibration? | Not recommended—design assumes calibration context for sweeps. |
| What if validation fails? | `ConfigurationException` stops pipeline early (WandB alert optional). |

## References

- Validation Logic: `views_pipeline_core.modules.validation.model.validate_config`
- CLI Argument Structure: `views_pipeline_core.cli.args.ForecastingModelArgs`
- ADRs: Deployment & Configuration separation (internal design documents)
- Logging Standards: ADR-025 (severity conventions)
