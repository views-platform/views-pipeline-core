# The Hardened Protocol: Contributor Governance for views-pipeline-core

This document defines mandatory engineering and numerical standards for the `views-pipeline-core` repository. Adherence to this protocol is required for all contributions to guarantee scientific integrity and reproducibility of conflict forecasts.

---

## 1. Core Principles

### A. The Authority of Declarations (ADR-003)
**"Never infer; only trust declarations."**
All meaningful semantics (prediction format, evaluation type, level, run type, step mapping) must be explicitly declared in configuration.
- **Prohibited:** Cell-type sniffing, positional step inference, directory-structure logic.
- **Requirement:** If a parameter affects forecast identity, it must be declared in config and validated by `CoreConfigSniffer`.

### B. The Fail-Loud Mandate (ADR-008)
**"A crash is a successful defense of scientific integrity."**
Silent failures, implicit fallbacks, and "best-effort" corrections are forbidden.
- **Requirement:** Violations of structural, temporal, or configuration invariants must raise `PipelineException` (or subclass) immediately.
- **Prohibited:** Using `nan_to_num` on predictions, silent clipping, converting errors to warnings, catching and swallowing `PipelineException`.

### C. The Numerical Airlock (Sniffer Pattern, ADR-041)
All data entering the pipeline must pass through structural auditing.
- **Requirement:** `CoreConfigSniffer.sniff_all()` before any model task.
- **Requirement:** `CoreDataSniffer.sniff_loaded_data()` after data fetch.
- **Requirement:** `CorePredictionSniffer.sniff_predictions()` after model inference.
- **Requirement:** `PredictionFrame._validate_input()` enforces `n_rows > 0`, `sample_count >= 1`, required identifiers.

### D. Extend Constants, Not Inline Checks
**"All magic strings live in module-level constants."**
- **Requirement:** New supported values go in `SUPPORTED_LEVELS`, `SUPPORTED_DEPLOYMENT_STATUSES`, etc.
- **Prohibited:** Inline string comparisons for supported value checks.

---

## 2. Contributor Requirements

### Adding a New Model (Downstream Repos)
1. **Config scripts:** Export `get_hp_config()`, `get_deployment_config()`, `get_meta_config()`.
2. **Partition dict:** Must have `train` and `test` keys with `(first_month, last_month)` tuples.
3. **Model scripts:** Export train/predict functions matching expected signatures.
4. **Predictions:** Must be `pd.DataFrame` with `pred_*` columns and correct MultiIndex, or `PredictionFrame` with valid identifiers.

### Adding a New Feature to Pipeline Core
1. **Ontological placement:** New class must belong to exactly one category (ADR-001).
2. **CIC required:** If class is non-trivial, write CIC before merging (ADR-006).
3. **Layer rules:** Respect topology (ADR-002). No upward imports.
4. **Tests:** GREEN + BEIGE minimum. RED team for validators and data representations.

---

## 3. Mandatory Testing Taxonomy (ADR-005)

Every Pull Request must include tests covering the following three perspectives:

### Green Team (Stability & Correctness)
* **Goal:** Ensure the system works as intended and remains stable.
* **Examples:** PredictionFrame collapse arithmetic, sniffer pass conditions, config merge priority.

### Beige Team (Configuration & Human Error)
* **Goal:** Catch failures caused by common configuration mistakes or missing parameters.
* **Examples:** Missing mandatory config keys, single-row predictions, NaN/Inf in predictions, wrong MultiIndex names.

### Red Team (Adversarial)
* **Goal:** Expose failure modes by deliberately trying to make the system fail silently.
* **Examples:** Non-standard target names, mutation of PredictionFrame internals, ensemble with mismatched indices, target names without conflict type codes.

---

## 4. Operational Invariants

- **Partition geometry:** `time_steps = 36`, `rolling_origin_stride = 1`, `MAX_SHIFT_COUNT = 12`, `test_len = 48`. Changes require ADR update.
- **Level constraint:** `level ∈ {"cm", "pgm"}`. Adding a new level requires updating sniffers, constants, and this protocol.
- **MultiIndex structure:** `(entity_id, month_id)` — enforced by `CoreDataSniffer` and `CorePredictionSniffer`.
- **PredictionFrame identifiers:** `time` and `unit` are required (no Optional default).

---

**"In this repository, we value correct forecasts over convenient execution."**
