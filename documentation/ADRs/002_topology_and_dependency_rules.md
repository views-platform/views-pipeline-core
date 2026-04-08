# ADR-002: Topology and Dependency Rules

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

Repo-assimilation confirmed this codebase has a clean DAG (no circular dependencies). However, coupling patterns show risk areas: `managers/model/model.py` has the highest fan-out (imports from 10+ internal modules), and `data/handlers.py` imports from `modules/statistics/` (a lower-level module reaching up to a higher-level one). Without explicit topology rules, these patterns will degrade over time.

## Decision

We enforce a layered dependency structure. Higher layers may depend on lower layers, but not the reverse.

### Layer Definitions (bottom to top)

```
Layer 0: Exceptions, Constants
         exceptions/, configs/drift_detection.py

Layer 1: Data Representations, CLI
         data/prediction_frame.py, data/handlers.py, data/utils.py
         cli/args.py, configs/pipeline.py

Layer 2: Validators (Sniffers), File I/O
         modules/validation/core_*.py
         files/utils.py

Layer 3: Domain Logic Modules
         modules/statistics/, modules/transformations/,
         modules/dataloaders/, modules/reconciliation/,
         modules/reports/, modules/mapping/, modules/visualizations/

Layer 4: Integration Modules
         modules/wandb/, modules/logging/, modules/appwrite/,
         modules/datastore/

Layer 5: Adapters and Persistence Managers
         modules/validation/adapter.py
         managers/prediction/, managers/configuration/

Layer 6: Orchestrators
         managers/model/, managers/ensemble/,
         managers/extractor/, managers/postprocessor/

Layer 7: Templates and Package Management
         templates/, managers/package/
```

### Rules

1. **Downward only:** A module in layer N may import from layers 0..N-1. It must not import from layers N+1 and above.
2. **No circular dependencies:** If A imports B, B must not import A (directly or transitively).
3. **Cross-layer shortcuts forbidden:** Layer 6 may not bypass Layer 5 to directly manipulate Layer 2 internals (use the Layer 5 interface).
4. **Lazy imports permitted:** `TYPE_CHECKING` imports and lazy `importlib` imports are exempt from layer enforcement for type annotation purposes only.

### Known Deviations

- `data/handlers.py` (Layer 1) imports `modules/statistics/PosteriorDistributionAnalyzer` (Layer 3) for MAP computation. This is a topology violation.
- `configs/pipeline.py` (Layer 1) lazily imports `managers/package/PackageManager` (Layer 7) for version fetching. This uses lazy import to avoid circular dependency but violates the spirit of the rule.
- `ForecastingModelManager` (Layer 6) directly calls sniffer methods (Layer 2) — this is acceptable as orchestrators coordinate all layers.
- `modules/dataloaders/dataloaders.py` (Layer 3) imports `ModelPathManager` from `managers.model` (Layer 6) via backward-compat re-export. ADR-045 E6 relocated `ModelPathManager` to `data/model_path.py` (Layer 1); the import path in dataloaders should be updated to use the canonical location. See risk register C-43.

## Rationale

Explicit topology prevents the dependency graph from degrading into a "big ball of mud." The current clean DAG is an asset worth protecting. Layer rules make violations visible in code review.

## Consequences

### Positive
- New imports that violate layer rules are detectable
- Encourages interface-based interaction between layers
- Prevents circular dependency emergence

### Negative
- Known deviations need eventual resolution
- May require introducing adapter patterns for legitimate cross-layer needs

## References

- [ADR-001: Ontology of the Repository](001_ontology_of_the_repository.md)
- [ADR-009: Boundary Contracts and Configuration Validation](009_boundary_contracts_and_configuration_validation.md)

---
*End of ADR-002.*
