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

## Amendment — 2026-08-01: the fidelity axis (register C-247, story #350)

**Status of the original text: unchanged and still in force.** This amendment adds an
axis; it removes nothing. The RED/BEIGE/GREEN taxonomy and the coverage table above
remain the requirement.

### What went wrong that this exists to prevent

On 2026-08-01 the Appwrite seam measured **19% RED** — comfortably above every target in
the table above — and shipped a module that reported **436 phantom orphan files against
production**.

The tests were not lazily written. They were well-distributed across the three
categories, they passed review, and CI was green. What they could not do was fail: the
fake they drove returned every metadata document in one call, because its author believed
`list_documents` was unpaged. Appwrite returns 25 per page. Every test agreed with a mock
written from **the same false belief as the code**.

> **Balanced colours over a wrong premise are uniformly wrong.**

The taxonomy classifies a test by its *intent* — is it probing happy path, boundary, or
adversarial behaviour? That is a real and useful question. But it is silent on a second,
independent one: **can this test observe anything the author did not already believe?**

### The axis

Every test also has a **fidelity**, orthogonal to its colour:

| Fidelity | What the test drives | Can it correct a false belief about the substrate? |
|---|---|---|
| **Substrate** | The real dependency, or a subprocess/AST observation of real code | **Yes** |
| **Recorded** | A response captured from the real dependency and committed | **Yes**, for the shape it captured |
| **Derived** | A double built from the dependency's own source or encoding | **Partly** — catches mechanism errors, not wrong constants |
| **Asserted** | A hand-written double expressing the author's belief | **No** |

An *asserted* test is not a bad test. Most tests should be asserted — they are cheap,
fast and precise about our own logic. The failure is not writing them; it is having
**only** them at a boundary where the substrate can surprise you, and then reading a
healthy colour balance as evidence of safety.

### The requirement

**Any module that talks to an external substrate must have at least one test at
Substrate or Recorded fidelity for each behaviour a wrong belief would silently
corrupt.** Report fidelity alongside category balance; a colour distribution quoted
without it is an incomplete claim.

Worked example, this repo: `tests/test_modules/test_appwrite_sdk_contract.py` (Substrate
— drives the real installed SDK), `tests/fixtures/appwrite/list_documents_shape.json`
(Recorded — the capture that established Appwrite returns 25 of 461 with no limit
supplied), `tests/test_import_purity.py` (Substrate — subprocess probes of real imports),
`tests/test_modules/test_appwrite_pagination.py` (Derived — a double built from the SDK's
own query encoding).

### Consequence for C-21's trigger

C-21 asks reviewers to *"verify at least 10% of tests are RED."* The seam hit 19% and was
still certifying a false belief, so that metric — alone — cannot answer the question it is
being asked. It stands as a floor for adversarial intent; it is not evidence of fidelity,
and this amendment is what a reviewer should reach for instead.

## Documents Amended by This Amendment

| Document | Amendment |
|---|---|
| `CICs/AppWriteFileModule.md` §10 | Corrected: the beige tier it described ("integration tests against a live or emulated Appwrite instance") **never existed**. Replaced with the tiers that do, annotated by fidelity (register C-246). |
| ADR-006 §CIC Section 10 | No amendment needed. A CIC's Test Alignment section should now state fidelity as well as colour; the section itself is unchanged. |

---
*End of the 2026-08-01 amendment.*

---
*End of ADR-005.*
