# ADR-005: Testing as Mandatory Critical Infrastructure

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

This project already uses a RED/BEIGE/GREEN test taxonomy in `test_audit_security_robustness.py`, `test_explicit_tasks.py`, and `test_prediction_frame.py`. The test suite contains ~1,000+ test functions across 35 files, with strong coverage of managers and validation modules but gaps in visualization, mapping, templates, and package management.

The standalone `audit_suite.py` provides additional non-pytest auditing with the same traffic-light classification.

## Decision

Testing is critical infrastructure, not optional documentation. Tests must cover three distinct threat models, and no category may substitute for another.

### Test Taxonomy

| Team | Symbol | Purpose | Examples in this project |
|------|--------|---------|------------------------|
| **Green** | `# GREEN` | Prove intended functionality works | `test_prediction_frame.py`: arithmetic mean collapse, identifier handling |
| **Beige** | `# BEIGE` | Test realistic boundary conditions and boring-but-dangerous patterns | `test_prediction_frame.py`: single row, large S=10000, NaN/Inf handling |
| **Red** | `# RED` | Adversarial and failure-mode testing | `test_prediction_frame.py`: unknown methods, mutation safety; `audit_suite.py`: naming fragility, ensemble data coupling |

### Test File Conventions

- Test files mirror source structure: `tests/test_managers/`, `tests/test_modules/`, `tests/test_utils/`, `tests/test_data/`, `tests/test_configs/`
- Test functions use `test_` prefix
- Fixtures provide canonical test data (e.g., `_make_pf()`, `sample_cm_dataframe`)
- External dependencies mocked at `sys.modules` level (wandb, views_evaluation, art)

### Coverage Requirements

| Module Category | Required Coverage |
|----------------|------------------|
| Validators (Sniffers) | All three teams (GREEN + BEIGE + RED) |
| Data Representations | All three teams |
| Orchestrators | GREEN + BEIGE minimum |
| Adapters | GREEN + BEIGE minimum |
| Persistence | GREEN minimum |
| All others | GREEN minimum |

### Known Coverage Gaps

These modules currently have no dedicated tests:
- `modules/mapping/` (MappingModule)
- `modules/visualizations/` (PlotDistribution, HistoricalLineGraph)
- `templates/` (code generation)
- `managers/package/` (PackageManager)
- `modules/reconciliation/` (only tested indirectly via statistics)

These gaps are documented, not accepted. They represent technical debt.

## Rationale

This library produces conflict forecasts used by policy decision-makers. A regression in prediction format dispatch or evaluation metric computation can silently corrupt production outputs. The three-team taxonomy ensures coverage of both happy paths and failure modes.

## Consequences

### Positive
- Test taxonomy is explicit and enforceable
- Coverage gaps are documented and trackable
- New CICs must reference their test alignment (CIC Section 10)

### Negative
- Writing three categories of tests requires more effort than basic unit testing
- External dependency mocking creates a test/production boundary that may mask integration issues

## References

- [ADR-006: Intent Contracts](006_intent_contracts_for_non_trivial_classes.md) — CIC Section 10 (Test Alignment)
- [ADR-008: Observability and Explicit Failure](008_observability_and_explicit_failure.md) — fail-loud testing
- Test files: `tests/` directory
- Standalone audit: `audit_suite.py`

---
*End of ADR-005.*
