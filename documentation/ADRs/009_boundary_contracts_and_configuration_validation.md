# ADR-009: Boundary Contracts and Configuration Validation

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

This library sits at the center of the VIEWS pipeline ecosystem. It defines contracts consumed by:
- **Downstream model repositories** (call `ForecastingModelManager.execute_single_run()`)
- **Ensemble repositories** (call `EnsembleManager.execute_single_run()`)
- **External libraries** (`views-evaluation`, `views-forecasts`, `viewser`)
- **Cloud services** (Appwrite, WandB)

Each boundary has implicit expectations about input format, configuration structure, and output schema. The Sniffer Pattern (ADR-041) was introduced to enforce these contracts structurally.

## Decision

All architectural boundaries must declare explicit contracts. Configuration is a first-class artifact, not a convenience layer. Inputs must be validated at entry, before state mutation.

### Boundary Types in This Project

| Boundary | Contract Enforced By | Validates |
|----------|---------------------|-----------|
| **Config entry** | `CoreConfigSniffer.sniff_all()` | Mandatory keys, supported values, deployment status, evaluation contract |
| **Data entry** | `CoreDataSniffer.sniff_loaded_data()` | MultiIndex structure, partition compatibility, non-empty data |
| **Prediction output** | `CorePredictionSniffer.sniff_predictions()` | Non-empty, pred_* columns, MultiIndex structure |
| **PredictionFrame construction** | `PredictionFrame._validate_input()` | n_rows > 0, sample_count >= 1, required identifiers |
| **CLI entry** | `ForecastingModelArgs.__post_init__()` | Argument constraints, mutual exclusions |
| **Configuration merge** | `ConfigurationManager` | Priority ordering, type consistency |
| **External library** | `EvaluationAdapter` | DataFrame/PredictionFrame → EvaluationFrame conversion |
| **Cloud storage** | `AppWriteFileModule` | File metadata, bucket existence |

### Configuration Validation Rules

1. **Validate before use:** `CoreConfigSniffer.sniff_all()` runs before any model task executes.
2. **No semantic defaults:** Missing configuration keys raise exceptions, not fall back to defaults. The `level` parameter is required (no `Optional` default) in both `CoreDataSniffer` and `CorePredictionSniffer`.
3. **Constants over inline checks:** Supported values are defined as module-level constants (e.g., `SUPPORTED_LEVELS`, `SUPPORTED_DEPLOYMENT_STATUSES`), not inline string comparisons. To add a new supported value, extend the constant.
4. **Partition structure is declared:** `{"train": (first_month, last_month), "test": (first_month, last_month)}` — this structure is assumed throughout the pipeline and validated by `CoreConfigSniffer._check_evaluation_contract()`.

### Cross-Repo Contract

This library defines the contract that downstream model repositories must satisfy:

- Config scripts must export `get_hp_config()`, `get_deployment_config()`, `get_meta_config()`
- Partition dict must have `train` and `test` keys with `(first, last)` tuples
- Model scripts must export train/predict functions with expected signatures
- Predictions must be either `pd.DataFrame` with `pred_*` columns and correct MultiIndex, or `PredictionFrame` with valid identifiers

## Rationale

Boundary contracts prevent the "garbage in, garbage out" failure mode. By validating at entry, errors are caught before expensive computation (training, evaluation) begins. The sniffer pattern makes validation both systematic and extensible.

## Consequences

### Positive
- Invalid configurations fail fast with clear error messages
- Contract enforcement is centralized in sniffers (not scattered)
- Constants make it easy to audit what values are supported

### Negative
- Adding new supported values requires updating constants (intentional friction)
- Cross-repo contract is implicit — downstream repos must discover it from templates and documentation

### Open Questions

- Should the cross-repo contract be formalized as a schema (e.g., JSON Schema for config)?
- Should `CoreConfigSniffer` validate the existence of required scripts, not just config values?

## References

- [ADR-003: Authority of Declarations Over Inference](003_authority_of_declarations_over_inference.md)
- [ADR-008: Observability and Explicit Failure](008_observability_and_explicit_failure.md)
- [ADR-041: Sniffer Pattern](041_sniffer_pattern.md)
- [ADR-042: Prediction Frame Adoption](042_prediction_frame_adoption.md)
- Sniffers: `views_pipeline_core/modules/validation/core_*.py`

---
*End of ADR-009.*
