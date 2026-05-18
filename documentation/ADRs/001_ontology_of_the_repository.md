# ADR-001: Ontology of the Repository

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

`views-pipeline-core` contains ~40 non-trivial classes spread across `managers/`, `modules/`, `data/`, `configs/`, `cli/`, `exceptions/`, and `templates/`. Without an explicit ontology, contributors (carbon and silicon) must infer what kinds of things are allowed to exist and where they belong. This leads to misplaced logic, duplicated abstractions (e.g., the `ensemble_aggregator/` vs `dataloaders/` overlap), and god classes that accumulate responsibilities.

## Decision

We define explicit ontological categories for all entities in this repository. Every non-trivial class must belong to exactly one category. New classes must be placed in the correct category or the ontology must be updated first.

### Core Ontological Categories

| Category | Purpose | Representative Classes | Location | Stability |
|----------|---------|----------------------|----------|-----------|
| **Orchestrators** | Pipeline lifecycle coordination (train, evaluate, forecast) | `ForecastingModelManager`, `EnsembleManager`, `ExtractorManager`, `PostprocessorManager`, `ModelManager` | `managers/model/`, `managers/ensemble/`, `managers/extractor/`, `managers/postprocessor/` | Evolving |
| **Path Managers** | Centralized path resolution for model artifacts, configs, data | `ModelPathManager`, `EnsemblePathManager`, `ExtractorPathManager`, `PostprocessorPathManager` | Same manager directories | Stable |
| **Configuration** | Multi-source config merge and validation | `ConfigurationManager`, `PipelineConfig` | `managers/configuration/`, `configs/pipeline.py` | Stable |
| **Data Representations** | Canonical data containers for predictions and spatiotemporal datasets | `PredictionFrame`, `_ViewsDataset`, `CMDataset`, `PGMDataset` | `data/prediction_frame.py`, `data/handlers.py` | Stable |
| **Validators (Sniffers)** | Structural auditing of configs, data, and predictions | `CoreConfigSniffer`, `CoreDataSniffer`, `CorePredictionSniffer` | `modules/validation/core_*.py` | Stable |
| **Adapters** | Bridge between internal representations and external libraries | `EvaluationAdapter`, `PredictionFrameConverter` | `modules/validation/adapter.py`, `managers/prediction/prediction_frame_converter.py` | Evolving |
| **Persistence** | Prediction I/O, file storage, cloud upload | `PredictionIOManager`, `DatastoreModule`, `AppWriteFileModule`, `CacheManager` | `managers/prediction/io.py`, `modules/datastore/`, `modules/appwrite/` | Evolving |
| **Aggregation** | Ensemble prediction pooling and hierarchical reconciliation | `AggregationManager`, `ReconciliationModule`, `ForecastReconciler` | `modules/dataloaders/`, `modules/reconciliation/`, `modules/statistics/` | Stable |
| **Analysis** | Statistical computation (MAP, HDI, posterior distributions) | `PosteriorDistributionAnalyzer` | `modules/statistics/statistics.py` | Stable |
| **Transformations** | Data transformations with undo capability | `DatasetTransformationModule` | `modules/transformations/` | Stable |
| **Reporting** | HTML report generation, geographic mapping, visualization | `ReportModule`, `MappingModule`, `PlotDistribution`, `HistoricalLineGraph`, `EvaluationReportTemplate`, `ForecastReportTemplate` | `modules/reports/`, `modules/mapping/`, `modules/visualizations/`, `templates/reports/` | Stable |
| **Integration** | External service wrappers | `WandBModule`, `LoggingModule` | `modules/wandb/`, `modules/logging/` | Stable |
| **CLI** | Argument parsing and validation | `ForecastingModelArgs`, `ModelArgs` | `cli/args.py` | Stable |
| **Package Management** | Poetry package scaffolding | `PackageManager` | `managers/package/` | Stable |
| **Exceptions** | Custom error hierarchy with WandB alerting | `PipelineException` + 6 subclasses | `exceptions/` | Stable |

### Stability Definitions

| Level | Meaning |
|-------|---------|
| **Stable** | API and responsibilities are settled; changes are additive only |
| **Evolving** | Active refactoring in progress; API may change across releases |
| **Experimental** | May be removed or fundamentally redesigned |

## Rationale

An explicit ontology prevents category drift (e.g., orchestrators accumulating validation logic) and makes it possible to enforce topology rules (ADR-002). The categories above were derived from repo-assimilation analysis of the actual codebase, not aspirational design.

## Consequences

### Positive
- New classes have a clear placement rule
- God-class growth becomes visible (class doesn't fit one category)
- AI assistants can validate placement against the ontology

### Negative
- Some existing classes span categories (e.g., `ForecastingModelManager` is an Orchestrator that also contains Persistence and Adapter logic — this is a known deviation, not a reason to change the ontology)
- Ontology must be updated when genuinely new categories emerge

## References

- [ADR-002: Topology and Dependency Rules](002_topology_and_dependency_rules.md)
- [ADR-006: Intent Contracts for Non-Trivial Classes](006_intent_contracts_for_non_trivial_classes.md)
- [CICs directory](../CICs/) — one contract per non-trivial class

---
*End of ADR-001.*
