# ADR-041: The Sniffer Pattern — Structural Auditing at Trust Boundaries

**Status:** Accepted
**Date:** 2026-03-02
**Deciders:** Project maintainers

---

## Context

The pipeline has multiple **trust boundaries** — points where data crosses from one
component to another and structural correctness can no longer be assumed:

1. Model config dict → execution engine (pre-inference)
2. VIEWSER data store → training/evaluation pipeline (post-fetch, pre-training)
3. Model inference output → evaluation/persistence layer (post-inference, pre-evaluation)

Before March 2026, these boundaries were guarded by scattered standalone functions with
inconsistent semantics:
- `validate_config()` — printed coloured text, mutated the config in place, returned `None`.
- `_validate_df_partition()` — returned a `bool`; callers had to check the return value.
- `validate_prediction_dataframe()` — mixed print statements with raises; accepted legacy
  formats silently.

These functions were untestable in isolation, produced silent failures in edge cases, and
had no consistent naming convention or architectural home.

## Decision

All trust-boundary validation in this repository follows the **Sniffer Pattern**:

### Rules

1. **State-bearing audit class.** A sniffer is instantiated with its validation context
   (partition, level, config). Context is pre-computed and immutable after construction.

2. **Single named suite method.** Each sniffer exposes one public entry point named
   `sniff_*()` (e.g. `sniff_all()`, `sniff_loaded_data()`, `sniff_predictions()`).
   The suite calls all relevant checks in order.

3. **Read-only invariant.** A sniffer never modifies data, configs, indices, or any
   stored state. Violation of this invariant is a bug, not a design choice.

4. **Fail Loud and Proud.** On any contract violation, the sniffer raises immediately
   with a self-identifying error message prefixed `"ClassName: ..."`. Silent failure
   paths do not exist.

5. **Logged success.** On a clean pass, the suite method emits one `logger.info()`
   message confirming the audit completed.

6. **Naming convention.** Sniffer classes are named `Core<Domain>Sniffer`. They live
   in `views_pipeline_core/modules/validation/` and are exported from
   `views_pipeline_core/modules/validation/__init__.py`.

7. **Composition over inheritance.** Shared check logic is extracted as module-level
   utility functions (e.g. `_check_multiindex()` in `core_data_sniffer.py`).
   Sniffers call these functions; they do not extend a common base class. The three
   sniffers have incompatible constructors and audit different artefacts — a shared
   base class would violate ISP.

8. **One CIC per sniffer.** Every `Core*Sniffer` class has a corresponding Class
   Intent Contract in `documentation/CICs/`.

9. **Knowledge locality — the most important rule.** A sniffer may only audit what
   the repository it lives in can legitimately know. `views_pipeline_core` knows only
   the **universal pipeline contract**: the keys and values mandated across every
   model and every run. It does not — and must not — know model-specific config keys,
   evaluation-specific semantics, or any knowledge that belongs to a downstream or
   external component.

   Concretely:
   - Model-specific config sniffers (checking keys that only a given model declares)
     live in the **model repository**, not here.
   - Evaluation sniffers (checking things that only `views-evaluation` can know)
     live in the **evaluation repository**, not here.
   - Future integration (e.g. pipeline-core activating remote sniffers via hooks)
     does not move the sniffer code here — the code remains where the knowledge lives;
     only the activation point changes.

   Absorbing foreign knowledge into `views_pipeline_core` to give its sniffers
   broader reach is an architectural violation. When in doubt: if `views_pipeline_core`
   would need to import from an external model or evaluation repo to perform a check,
   that check does not belong here.

### Naming note — relation to ADR-040

ADR-040 ("The No-Sniffing Rule") forbids **semantic inference**: deducing computation
semantics (task type, prediction type, lead-time step) from data content.

The `Core*Sniffer` naming is adopted from the hydranet `DataSniffer` pattern and refers
to **structural auditing**: validating that data meets an explicit, pre-declared contract.
A sniffer makes no computation decisions; it either passes silently or raises immediately.
ADR-040's prohibition is orthogonal and remains fully in force.

### Active sniffers

| Class | Suite method | Trust boundary |
|-------|-------------|----------------|
| `CoreConfigSniffer` | `sniff_all(run_type)` | Model config → execution engine |
| `CoreDataSniffer` | `sniff_loaded_data(df)` | VIEWSER store → training pipeline |
| `CorePredictionSniffer` | `sniff_predictions(df, targets)` | Inference output → evaluation/persistence |

## Consequences

### Positive

- Validation is testable, named, and architectural; each sniffer has an isolated test
  file and a CIC.
- Adding a new trust boundary means adding a new `Core*Sniffer` — a clear, low-risk
  extension point.
- Error messages are self-identifying, making debugging fast.
- The read-only invariant means sniffers can safely be added to any boundary without
  risk of mutating pipeline state.

### Negative

- Per-call instantiation (sniffer constructed fresh for each invocation) is slightly
  more verbose than a standalone function call.
- Adding a new sniffer requires a CIC and a test file, not just a function — a
  deliberate trade-off for architectural clarity.

### On knowledge locality

The Sniffer Pattern is intentionally distributed across repositories.
`views_pipeline_core` implements only the sniffers for the universal pipeline contract.
Model repositories, the evaluation repository, and any other downstream component may
implement their own `*Sniffer` classes for knowledge they own.

A future integration where `views_pipeline_core` activates downstream sniffers at the
earliest possible opportunity (e.g. via hooks) does not require relocating the sniffer
code: the activation point changes, the knowledge locality does not. The code remains
where the knowledge lives.
