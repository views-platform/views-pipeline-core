# R&D Roadmap: Ensemble Architecture

**Date:** 2026-03-17
**Branch:** `feature/samples_for_fao`
**Scope:** `views_pipeline_core/managers/ensemble/`, `views_pipeline_core/modules/ensemble_aggregator/`, `views_pipeline_core/modules/reconciliation/`

---

## 1. Current State

### What exists

`EnsembleManager` (ensemble.py, 827 lines) orchestrates multi-model ensembles: training member models via subprocess, loading their predictions, aggregating results via `AggregationManager` (Polars-based), and optionally reconciling country-grid hierarchies via `ReconciliationModule`.

PredictionFrame is invisible to the ensemble. Member models that use PredictionFrame convert to parquet before the ensemble loads them. This is architecturally correct (ADR-033) and the parquet format is the permanent cross-model contract.

### SOLID violations inventory

| Principle | Location | Issue | Severity |
|-----------|----------|-------|----------|
| **SRP** | `EnsembleManager` | 6 responsibilities in one class: orchestration, subprocess mgmt, file I/O, aggregation coordination, reconciliation coordination, arg construction | Critical |
| **OCP** | `ensemble.py:605-634` | Reconciliation type hardcoded via `if reconciliation_type == "pgm_cm_point"` | High |
| **OCP** | `ensemble.py:543-592` | Prediction loading has no backend abstraction (if/else for store vs local) | High |
| **OCP** | `aggregator.py:199` | Aggregation methods are a closed set `{"mean", "median", "min", "max"}` | Medium |
| **LSP** | `ensemble.py:368-445` | `_train_model_artifact(model_name)` adds parameter not in parent signature | Medium |
| **DIP** | `ensemble.py:729-735, 772` | Hardcoded instantiation of `EnsemblePathManager`, `_CDataset`, `ReconciliationModule`, `AggregationManager` | High |
| **DIP** | `reconciliation.py:68` | Hardcoded `ForecastReconciler` with no injection point | Medium |
| **DRY** | `ensemble.py:186-276` | WandB execution boilerplate (init, try/except, alert, finally) duplicated 3x identically | High |

---

## 2. Target State

### Architectural principles

1. **EnsembleManager is a thin coordinator.** It delegates to injected collaborators, not hardcoded implementations.
2. **Every extension point is a protocol.** New reconciliation strategies, aggregation methods, and storage backends are addable without modifying existing code.
3. **No duplication.** Shared execution patterns are extracted once.
4. **Parquet remains the contract.** PredictionFrame-native aggregation is an opt-in extension, not a replacement.

### Target class diagram

```
EnsembleManager (thin coordinator)
  ├── MemberPredictionLoader     [protocol]
  │     ├── LocalFilePredictionLoader
  │     └── PredictionStorePredictionLoader
  ├── ReconciliationStrategy     [protocol]
  │     ├── PgmCmPointReconciliation
  │     └── (future strategies)
  ├── AggregationStrategy        [protocol]
  │     ├── PolarsAggregationStrategy (current AggregationManager logic)
  │     └── (future PredictionFrame-native strategy)
  └── ExecutionContext           [shared WandB boilerplate]
```

---

## 3. Phased Milestones

### Phase 1: Extract shared patterns (DRY, low risk)

**Goal:** Eliminate the 3x duplicated WandB execution boilerplate.

**What:**
- Create `ExecutionContext` context manager (or decorator) that wraps stage execution with WandB init, error handling, alerting, and finalize.
- Replace the 3 identical try/except/finally blocks in `_execute_model_training()`, `_execute_model_evaluation()`, `_execute_model_forecasting()`.

**Acceptance criteria:**
- 3 blocks replaced with 3 `with` statements
- All existing ensemble tests pass
- No behavioral change (WandB run init, alerts, and error wrapping identical)

**Risk:** Low. Pure extraction — same code, less duplication.

---

### Phase 2: Extract `MemberPredictionLoader` (SRP + DIP + OCP)

**Goal:** Decouple prediction loading from EnsembleManager.

**What:**
- Define `MemberPredictionLoader` protocol with method: `load(model_name, run_type, ...) -> pd.DataFrame`
- Extract `_load_or_generate_prediction()` (ensemble.py:514-592) into `LocalFilePredictionLoader`
- Create `PredictionStorePredictionLoader` for prediction store path
- Inject loader into EnsembleManager via constructor

**Acceptance criteria:**
- EnsembleManager no longer contains file I/O logic
- `use_prediction_store` flag selects loader at construction, not at runtime branching
- New backend (e.g., S3) addable by implementing protocol without modifying EnsembleManager

**Risk:** Medium. Changes constructor signature; tests that mock `_load_or_generate_prediction` need updating.

---

### Phase 3: Extract `ReconciliationStrategy` (OCP + DIP)

**Goal:** Make reconciliation pluggable.

**What:**
- Define `ReconciliationStrategy` protocol with method: `reconcile(df_prediction, configs) -> pd.DataFrame`
- Extract `_apply_reconciliation()` + `__reconcile_pg_with_c()` (ensemble.py:594-741, 148 lines) into `PgmCmPointReconciliation`
- Create `ReconciliationStrategyFactory.create(reconciliation_type)` that returns the right strategy
- Inject strategy into EnsembleManager; `None` means no reconciliation

**Acceptance criteria:**
- EnsembleManager reconciliation logic reduced to: `df = self._reconciliation_strategy.reconcile(df, self.configs)`
- New reconciliation type addable by implementing protocol + registering in factory
- 148 lines of domain logic moved out of EnsembleManager

**Risk:** Medium. Reconciliation logic is complex with dataset loading; must preserve exact behavior.

---

### Phase 4: Inject `AggregationManager` via factory (DIP)

**Goal:** Remove direct instantiation of AggregationManager.

**What:**
- Create `AggregationStrategy` protocol with methods: `add_model()`, `aggregate()`
- Current `AggregationManager` becomes the default implementation
- EnsembleManager receives factory/strategy via constructor, not hardcoded instantiation
- `_get_aggregated_df()` delegates to injected strategy

**Acceptance criteria:**
- `AggregationManager` is no longer instantiated inside EnsembleManager
- Future PredictionFrame-native aggregation strategy can be added without modifying EnsembleManager (OCP)
- Aggregation method extensibility: custom callables accepted, not just string names

**Risk:** Low-medium. AggregationManager API is already clean; wrapping it in a protocol is mechanical.

---

### Phase 5: PredictionFrame-native aggregation (OCP extension, future)

**Goal:** Enable ensembles to aggregate PredictionFrame objects directly, without parquet roundtrip.

**What:**
- Implement `PredictionFrameAggregationStrategy` that operates on `Dict[str, List[PredictionFrame]]`
- Aggregate samples directly in numpy (no pandas/polars conversion)
- Register in factory keyed by `prediction_format` config

**Acceptance criteria:**
- Ensemble with all-PF members can aggregate without writing intermediate parquet
- Memory usage reduced (no materialization of list-in-cell DataFrames)
- All-DF and mixed ensembles continue to work unchanged

**Risk:** High. Requires defining the PF ensemble contract. Deferred until demand exists.

---

## 4. Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Low | Pure extraction; test coverage exists for WandB integration |
| 2 | Medium | Mock tests must be updated; integration test with subprocess needed |
| 3 | Medium | Reconciliation logic is stateful (dataset caches); must preserve cache behavior |
| 4 | Low-Medium | AggregationManager API is stable; protocol is thin wrapper |
| 5 | High | New contract; requires multi-repo coordination; defer until demand |

---

## 5. What NOT to change

- The parquet on-disk format (universal cross-model contract)
- The subprocess delegation pattern for member model training
- AggregationManager's internal Polars logic (proven, performant)
- ReconciliationModule's core algorithm (torch-based proportional scaling)
- The EnsembleManager ↔ ForecastingModelManager inheritance (LSP fix is low-priority and high-risk)
