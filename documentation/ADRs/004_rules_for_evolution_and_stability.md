# ADR-004: Rules for Evolution and Stability

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

Repo-assimilation identified several structural risks that require evolution governance:

- **R1:** `ForecastingModelManager` at 3210 LOC is a god class concentrating orchestration, evaluation, persistence, and format dispatch. Decomposition is planned but must be staged.
- **R5:** `ModelPathManager` uses class-level mutable state (`_root`, `_models`) that is set by the first instance created — a correctness hazard in multi-model scenarios.
- **R9:** `modules/ensemble_aggregator/` appears to duplicate `modules/dataloaders/` — legacy code that needs resolution.
- Multiple product development roadmaps exist in `reports/` for extraction of evaluation logic, prediction I/O, reporting, and forecast shipping.

Without explicit evolution rules, refactoring can introduce regressions or stall indefinitely.

## Decision

We adopt stability tiers and evolution rules to govern how components change over time.

### Stability Tiers

| Tier | Meaning | Change Rules |
|------|---------|-------------|
| **Stable** | API settled; consumed by downstream repos | Additive changes only. Breaking changes require a new ADR. |
| **Evolving** | Active refactoring; API may change | Changes must update CICs. Downstream consumers warned via deprecation. |
| **Experimental** | May be removed or fundamentally redesigned | No stability guarantees. Must not be depended on by stable components. |

### Current Tier Assignments

Per ADR-001 ontology:

| Tier | Categories |
|------|-----------|
| Stable | Data Representations, Validators, Configuration (`PipelineConfig`), CLI, Analysis, Transformations, Integration, Exceptions |
| Evolving | Orchestrators, Adapters, Persistence, Configuration (`ConfigurationManager`) |
| Experimental | (none currently) |

### Evolution Rules

1. **Strangler Fig for god classes:** `ForecastingModelManager` decomposition follows the Strangler Fig pattern — extract responsibilities into new classes (e.g., `PredictionIOManager` already extracted) while maintaining the existing interface. Do not rewrite in place.
2. **One extraction at a time:** Each extraction (eval logic, reporting, etc.) is a separate branch and PR. Do not combine extractions.
3. **CIC-first for new classes:** Any class extracted from an Evolving component must have a CIC written before the extraction PR is merged.
4. **Deprecation before removal:** Deprecated code paths must log `DeprecationWarning` for at least one release cycle before removal.
5. **No silent legacy retention:** If a module is duplicated (e.g., `ensemble_aggregator/` vs `dataloaders/`), resolve within one release. Do not let both persist.

## Rationale

The project is in an active refactoring phase with multiple planned extractions. Without explicit rules, concurrent refactoring by multiple contributors (human and AI) risks creating inconsistent intermediate states.

## Consequences

### Positive
- Decomposition of god classes has a clear process
- Stability expectations are explicit for downstream consumers
- Prevents "refactor everything at once" anti-pattern

### Negative
- Evolving components require more documentation overhead
- Strangler Fig pattern means temporary duplication during transitions

## References

- [ADR-001: Ontology of the Repository](001_ontology_of_the_repository.md) — stability assignments
- [ADR-006: Intent Contracts](006_intent_contracts_for_non_trivial_classes.md) — CIC-first rule
- [Technical Risk Register](../../reports/technical_risk_register.md) — R1, R5, R9
- [ADR-042: Prediction Frame Adoption](042_prediction_frame_adoption.md) — active Strangler Fig migration

---
*End of ADR-004.*
