# ADR-003: Authority of Declarations Over Inference

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

This project has direct experience with the cost of semantic inference. Prior to February 2026, the pipeline inferred intent by inspecting runtime data:

- **Sniffing:** Checking if a DataFrame cell contained a `list` to infer whether the task was probabilistic or a point forecast (documented in ADR-040).
- **Positional step inference:** Assuming a prediction's lead time based on its row position in a list of DataFrames.

These inference patterns led to silent errors (e.g., misclassifying single-sample Monte Carlo runs as point forecasts) and were removed during the February 2026 refactoring. The "no-sniffing" rule (ADR-040) was the project-specific expression of this principle.

## Decision

All meaningful semantics must be explicitly declared. Inference across boundaries is forbidden.

### Core Principle

> When semantics are missing, ambiguous, or contradictory, the system must fail explicitly and immediately.

### What This Means in Practice

1. **Prediction format** is declared in config (`prediction_format: "prediction_frame"` or `"dataframe"`), not inferred from DataFrame cell types.
2. **Evaluation type** is declared (`eval_type: "standard"` or `"rolling"`), not inferred from partition structure.
3. **Level** is declared (`level: "cm"` or `"pgm"`), not inferred from index column names.
4. **Run type** is declared (`run_type: "calibration"` etc.), not inferred from artifact filenames.
5. **Step mapping** is explicitly constructed and passed, not inferred from DataFrame ordering.

### Enforcement Points

- `CoreConfigSniffer.sniff_all()` — validates all declared values before inference runs
- `CoreDataSniffer.sniff_loaded_data()` — validates structural declarations in loaded data
- `CorePredictionSniffer.sniff_predictions()` — validates prediction output matches declared format
- `PredictionFrame._validate_input()` — validates declared identifiers at construction

### Fail-Loud Mandate

Silent failure is a bug. When a declaration is missing or contradictory:
- Raise an exception (not return `False`, not log a warning)
- Include the expected vs actual values in the error message
- Include the location where the declaration should have been made

This is the project's "Fail Loud and Proud" philosophy.

## Rationale

This codebase produces conflict forecasts consumed by policy decision-makers. A silent misclassification (e.g., treating probabilistic predictions as point estimates) can propagate incorrect uncertainty bounds into production outputs. Declaration-over-inference eliminates this class of error.

## Consequences

### Positive
- Eliminates silent semantic errors
- Makes configuration the single source of truth
- Sniffers can validate declarations mechanically

### Negative
- More configuration boilerplate required
- Breaking change if downstream models relied on inference behavior (migration completed Feb 2026)

## References

- [ADR-040: Authority Over Inference](040_authority_over_inference.md) — project-specific expression of this principle
- [ADR-041: Sniffer Pattern](041_sniffer_pattern.md) — enforcement mechanism
- [ADR-008: Observability and Explicit Failure](008_observability_and_explicit_failure.md)
- [ADR-009: Boundary Contracts](009_boundary_contracts_and_configuration_validation.md)

---
*End of ADR-003.*
