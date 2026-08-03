# ADR-006: Intent Contracts for Non-Trivial Classes

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

This repository contains ~40 non-trivial classes across 15 ontological categories (ADR-001). Five classes already have intent contracts (CICs): `CoreConfigSniffer`, `CoreDataSniffer`, `CorePredictionSniffer`, `EvaluationAdapter`, and `PredictionFrame`. These CICs have proven valuable in preventing scope creep during the February–March 2026 refactoring.

The remaining non-trivial classes — particularly the orchestrators (`ForecastingModelManager`, `EnsembleManager`) and data representations (`_ViewsDataset`, `ConfigurationManager`) — lack explicit intent documentation, making it difficult to determine whether a change is a valid extension or a contract violation.

## Decision

All non-trivial classes must have an explicit Component Intent Contract (CIC) declaring what the class is meant to do, what it is not responsible for, and what guarantees it provides.

### What Makes a Class Non-Trivial

A class requires a CIC if it meets any of these criteria:
- Core domain class (e.g., `PredictionFrame`, `_ViewsDataset`)
- Sits at an architectural boundary (e.g., `EvaluationAdapter`, `PredictionFrameConverter`)
- Orchestrates other components (e.g., `ForecastingModelManager`, `EnsembleManager`)
- Owns state (e.g., `ConfigurationManager`, `AggregationManager`)
- Enforces invariants (e.g., `CoreConfigSniffer`, `CoreDataSniffer`)
- Modifies semantics or transformation (e.g., `UpdateViewser`)

### CIC Structure (11 Sections + Known Deviations)

1. Purpose
2. Non-Goals (Explicit Exclusions)
3. Responsibilities and Guarantees
4. Inputs and Assumptions
5. Outputs and Side Effects
6. Failure Modes and Loudness
7. Boundaries and Interactions
8. Examples of Correct Usage
9. Examples of Incorrect Usage
10. Test Alignment
11. Evolution Notes
12. Known Deviations (mandatory for brownfield classes)

### Current CIC Coverage

| Class | CIC Status |
|-------|-----------|
| CoreConfigSniffer | Active |
| CoreDataSniffer | Active |
| CorePredictionSniffer | Active |
| EvaluationAdapter | Active |
| PredictionFrame | Active |
| ForecastingModelManager | Active |
| EnsembleManager | Active |
| ModelPathManager | Active |
| ConfigurationManager | Active |
| PipelineConfig | Active |
| _ViewsDataset | Active |
| PredictionFrameConverter | Active |
| PredictionIOManager | Active |
| AggregationManager | Active |
| ReconciliationModule | Active |
| DatastoreModule | Active |
| ForecastingModelArgs | Active |
| DatasetTransformationModule | Retired — extracted (ADR-054), shim + CIC removed (#183, 2026-07-24) |
| PosteriorDistributionAnalyzer | Active |
| ReportModule | Active |

### Rules

1. **Tests must align with CIC:** If a CIC declares a guarantee, there must be a test that verifies it.
2. **Changes that violate CIC intent are bugs, not refactors.** A change that moves a class outside its declared purpose requires updating the CIC first.
3. **Known Deviations are mandatory for brownfield classes.** Honest documentation of where behavior doesn't match ideal contracts is required.
4. **CIC-first for new classes:** Classes extracted from Evolving components (ADR-004) must have a CIC before the extraction PR merges.

## Rationale

Intent contracts make implicit expectations explicit. This is especially critical in a codebase where AI assistants contribute code — the CIC provides a machine-readable specification of what a class should and should not do.

## Consequences

### Positive
- Scope creep becomes visible (change vs contract mismatch)
- AI assistants can validate changes against declared intent
- Review becomes faster — check change against CIC, not entire codebase

### Negative
- 20 CICs to maintain
- CICs can become stale if not updated with code changes

## References

- [ADR-001: Ontology of the Repository](001_ontology_of_the_repository.md)
- [ADR-005: Testing as Mandatory Critical Infrastructure](005_testing_as_mandatory_critical_infrastructure.md)
- [CICs directory](../CICs/)
- [CIC template](../CICs/cic_template.md)

---
*End of ADR-006.*
