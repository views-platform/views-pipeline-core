# Intent Contracts (CICs)

This directory contains Component Intent Contracts (CICs) for non-trivial classes in `views-pipeline-core`.

---

## What Is an Intent Contract?

An Intent Contract declares:
- **What a class is meant to do** (purpose and guarantees)
- **What it is NOT responsible for** (non-goals)
- **How it fails** (failure modes and loudness)
- **What it interacts with** (boundaries)
- **Where the code deviates from ideal** (known deviations)

Intent Contracts are governed by [ADR-006](../ADRs/006_intent_contracts_for_non_trivial_classes.md).

---

## When Is an Intent Contract Required?

A class requires a CIC if it:
- Is a core domain class
- Sits at an architectural boundary
- Orchestrates other components
- Owns state
- Enforces invariants
- Modifies semantics or transformation

---

## Structure of an Intent Contract

Each CIC has 12 sections:

1. **Purpose** — What the class does
2. **Non-Goals** — What it explicitly does not do
3. **Responsibilities and Guarantees** — Observable behavior commitments
4. **Inputs and Assumptions** — Constructor params, preconditions
5. **Outputs and Side Effects** — What it produces, state changes
6. **Failure Modes and Loudness** — How it fails (raise vs log vs silent)
7. **Boundaries and Interactions** — Dependencies and consumers
8. **Examples of Correct Usage** — Real usage patterns
9. **Examples of Incorrect Usage** — Anti-patterns
10. **Test Alignment** — Test file references and coverage
11. **Evolution Notes** — What is likely to change
12. **Known Deviations** — Where behavior doesn't match ideal

---

## Active Contracts

Every `.md` in this directory except this index and the template is listed here — a file on
disk that is not in this list is a defect (#506 found thirteen, 2026-09-19).

### Orchestrators
- `ForecastingModelManager.md` — Central pipeline orchestrator (train/evaluate/forecast/report)
- `EnsembleManager.md` — Multi-model ensemble orchestration (DataFrame path, template default)
- `DataFrameEnsembleManager.md` — Composition-based DataFrame ensemble manager
- `PredictionFrameEnsembleManager.md` — PredictionFrame-native ensemble manager

### Stages
- `EvaluationStage.md` — Build EvaluationFrame, score, persist the MetricFrame, log to WandB
- `ForecastingStage.md` — Predict, sniff, save on the forecasting path

### Path Managers
- `ModelPathManager.md` — Centralized path resolution for model artifacts; queryset loader

### Configuration
- `ConfigurationManager.md` — 5-source priority config merge
- `PipelineConfig.md` — Global singleton (format, version, org name)
- `ForecastingModelArgs.md` — Validated CLI argument dataclass

### Data Representations and Loading
- `PredictionFrame.md` — The views-frames leaf type, re-exported
- `_ViewsDataset.md` — MultiIndex DataFrame with tensor conversion
- `ViewsDataLoader.md` — Fetch, cache and validate a model's input
- `IDataSource.md` — Protocol between framework and engine data sources

### Validators (Sniffers)
- `CoreConfigSniffer.md` — Config contract enforcement
- `CoreDataSniffer.md` — Data structural auditing
- `CoreFrameSniffer.md` — FeatureFrame structural auditing
- `CorePredictionSniffer.md` — Prediction output validation, incl. entity coverage (ADR-064)

### Adapters
- `EvaluationAdapter.md` — DataFrame/PF to EvaluationFrame bridge
- `PredictionFrameConverter.md` — PF to DataFrame/Arrow conversion

### Persistence and External Services
- `PredictionIOManager.md` — Prediction persistence orchestration
- `DatastoreModule.md` — Appwrite file storage interface
- `AppWriteFileModule.md` — Appwrite file operations
- `AppwriteProvisioner.md` — Appwrite collection provisioning
- `WandBModule.md` — Weights & Biases runs, metrics, tables, alerts, artifacts

### Aggregation and Reconciliation
- `AggregationModule.md` — Ensemble prediction pooling; the joint-draw contract (ADR-064)
- `ReconcileFrames.md` — Cross-level reconciliation through the `Reconciler` port

### Reporting
- `PerModelMetricFrameSource.md` — Evaluation-of-record source, one root per model (#485)

## Retired Contracts

Kept for history; the class no longer exists in this repository. Each file's header says
where it went.

- `AggregationManager.md` — replaced by AggregationModule (2026-09-19, #506)
- `ReconciliationModule.md` — the class is `views_frames_reconcile.ReconciliationModule` (ADR-054)
- `PosteriorDistributionAnalyzer.md` — removed with `ForecastReconciler`; summaries live in `views_frames_summarize`
- `ReportModule.md` — lives in views-reporting (ADR-054)

---

## Template

See [cic_template.md](cic_template.md) for the standard CIC format.

---

## Governance

- Changes to class behavior that violate CIC intent are bugs (ADR-006).
- Changes to intent must update the CIC first.
- New non-trivial classes must have a CIC before merging (ADR-004).
- Known Deviations sections must be honest about current state.
