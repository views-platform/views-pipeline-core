# Master Development Plan: views-pipeline-core Architecture Improvement

**Date:** 2026-03-17
**Branch:** `feature/samples_for_fao`

---

## Overview

Five efforts have been identified to address SOLID violations across the codebase. This document ranks them into a single execution sequence with dependencies, rationale, and clear deferral indicators.

### The five efforts

| # | Effort | Target | Lines affected | Status |
|---|--------|--------|---------------|--------|
| E1 | Prediction I/O Extraction | model.py → PredictionIOManager | ~236 moved | **DO NOW** |
| E2 | Forecast Shipping Refactoring | PredictionIOManager → PredictionSaver protocols | ~96 decomposed | **DO NOW** |
| E3 | Evaluation Logic Extraction | model.py → EvaluationOrchestrator | ~350 moved | **DO NEXT** |
| E4 | Ensemble Architecture Refactoring | ensemble.py → protocols + strategies | ~827 decomposed | **DO NEXT** |
| E5 | Reporting Extraction | model.py → ReportingOrchestrator | ~199 moved | **DEFER** |

### Supporting documents

| Effort | R&D Roadmap | Product Development Plan |
|--------|-------------|--------------------------|
| E1 | `reports/rd_roadmap_prediction_io_extraction.md` | `reports/product_development_plan_prediction_io_extraction.md` |
| E2 | `reports/rd_roadmap_forecast_shipping.md` | `reports/product_development_plan_forecast_shipping.md` |
| E3 | `reports/rd_roadmap_evaluation_logic_extraction.md` | `reports/product_development_plan_evaluation_logic_extraction.md` |
| E4 | `reports/rd_roadmap_ensemble.md` | `reports/product_development_plan_ensemble.md` |
| E5 | `reports/rd_roadmap_reporting_extraction.md` | `reports/product_development_plan_reporting_extraction.md` |

---

## Execution Sequence

### Phase 1: Foundation — Extract I/O from the God Class

```
E1: Prediction I/O Extraction
├── Task 1: Create PredictionIOManager class (move 4 methods verbatim)
├── Task 2: Wire into ForecastingModelManager (delegation)
└── Task 3: Update test mocks
```

**Why first:**
- Lowest risk of all efforts (leaf nodes only, no call graph changes)
- Directly enables E2 (forecast shipping operates on 236-line module instead of 3,436-line god class)
- Prerequisite for E3 (evaluation logic calls I/O methods; must be extracted first to avoid circular dep)
- No subclass impact, no ensemble impact

**Result:** model.py drops from ~3,436 to ~3,200 lines. `PredictionIOManager` is a focused 236-line module.

**Estimated effort:** 1 session
**Verification:** 899 tests pass, ruff clean

---

### Phase 2: The Core Project — Forecast Shipping SOLID Compliance

```
E2: Forecast Shipping Refactoring (operates on PredictionIOManager, NOT model.py)
├── Task 1: Extract PredictionFileNamer
├── Task 2: Create PredictionSaver protocol + LocalParquetSaver
├── Task 3: Create PredictionStoreConfig with fail-loud env var validation
├── Task 4: Create AppwriteSaver + ViewsForecastsSaver
└── Task 5: Compose savers in save_predictions()
```

**Why second:**
- Fixes the highest-severity production risk (unvalidated env vars → fail-loud at startup)
- Resolves the Arrow NotImplementedError functional blocker
- Creates the PredictionSaver protocol that the ensemble project (E4) will reuse
- Now operates on PredictionIOManager (236 lines) thanks to E1 — massively reduced blast radius

**Result:** `save_predictions()` goes from 96 lines to ~15 lines. Each saver has exactly one responsibility. Arrow Tables can upload to prediction store.

**Estimated effort:** 2-3 sessions
**Verification:** 899 tests pass, 0 `os.getenv()` in model.py, 0 `NotImplementedError` in save path

---

### Phase 3: Structural Improvement — Extract Evaluation Logic

```
E3: Evaluation Logic Extraction
├── Task 1: Create EvaluationContext dataclass
├── Task 2: Create EvaluationOrchestrator class (move 3 methods)
└── Task 3: Wire + update callers and tests
```

**Why third:**
- Depends on E1 (evaluator needs `PredictionIOManager` reference, not raw `_save_evaluations`)
- Removes 350 lines from model.py (the largest single extraction)
- Makes evaluation logic independently testable — important for future metrics work
- Does not block E2 or E4, but makes model.py significantly more navigable for E4 work

**Result:** model.py drops from ~3,200 to ~2,850 lines. Evaluation logic is a focused module.

**Estimated effort:** 2 sessions
**Verification:** 899 tests pass, EnsembleManager's `_evaluate_prediction_dataframe()` still works via delegation

---

### Phase 4: The Second Project — Ensemble Architecture SOLID Compliance

```
E4: Ensemble Architecture Refactoring
├── Task 1: Extract WandB execution context manager (DRY)
├── Task 2: Create MemberPredictionLoader protocol (SRP, DIP, OCP)
├── Task 3: Create ReconciliationStrategy protocol (OCP, DIP)
├── Task 4: Inject AggregationManager via factory (DIP)
└── Task 5: Extend AggregationManager to accept callable methods (OCP)
```

**Why fourth:**
- Benefits from PredictionSaver protocol patterns established in E2
- Benefits from model.py being ~2,850 lines instead of 3,436 (less code to reason about when tracing inheritance)
- The WandB context manager (Task 1) is independent and can start anytime as a quick win
- MemberPredictionLoader mirrors the PredictionSaver pattern from E2

**Result:** ensemble.py drops from 827 to ~500 lines. Reconciliation, aggregation, and prediction loading are pluggable strategies.

**Estimated effort:** 3-4 sessions
**Verification:** 899 tests pass, ensemble line count < 500

---

### Phase 5 (DEFERRED): Reporting Extraction + Advanced Dispatch

```
E5: Reporting Extraction (DEFER until reporting needs change)
├── Task 1: Create ReportingOrchestrator
├── Task 2: Wire into ForecastingModelManager
└── Task 3: Update tests

E2-Task 6 (DEFERRED): PredictionFormatHandler protocol (DEFER until Phase 4 stable)
├── Eliminates isinstance() dispatch in _execute_model_forecasting/evaluation
└── High risk — touches 2 most complex methods (~230 lines each)
```

**Why deferred:**

E5 (Reporting):
- Does not unblock any planned work
- Methods are called once each — no duplication pressure
- Self-contained — doesn't make model.py harder to navigate
- **Do when:** reporting requirements change (new report type, new viz framework, FAO-specific reports)

E2-Task 6 (FormatHandler):
- Touches the core dispatch logic in `_execute_model_evaluation()` (225 lines) and `_execute_model_forecasting()` (107 lines)
- Requires comprehensive integration tests before safe to refactor
- **Do when:** a third prediction format is needed, or after E1-E4 are stable

---

## Dependency Graph (Complete)

```
E1 (Prediction I/O Extraction)
  │
  ├──→ E2 (Forecast Shipping)          — E2 operates on PredictionIOManager
  │      │
  │      └──→ E4 (Ensemble Refactoring) — reuses PredictionSaver pattern
  │
  └──→ E3 (Evaluation Logic Extraction) — evaluator references PredictionIOManager
         │
         └──→ E4 (Ensemble Refactoring) — model.py smaller = easier to trace inheritance

E5 (Reporting Extraction) — INDEPENDENT, DEFERRED
E2-Task 6 (FormatHandler) — depends on E2 Tasks 1-5, DEFERRED
```

---

## model.py Line Count Trajectory

| After | Lines | Reduction |
|-------|-------|-----------|
| Current | 3,436 | — |
| E1 (I/O extraction) | ~3,200 | -236 |
| E2 (forecast shipping) | ~3,200 | (refactors PredictionIOManager, not model.py) |
| E3 (evaluation extraction) | ~2,850 | -350 |
| E5 (reporting extraction, if done) | ~2,650 | -199 |
| **Total possible reduction** | **~2,650** | **-786 lines (23%)** |

The remaining ~2,650 lines are orchestration core + abstract contracts + entry points — the irreducible skeleton of ForecastingModelManager.

---

## Risk Summary

| Phase | Risk | What could go wrong | Reversibility |
|-------|------|--------------------|--------------|
| E1 | **Low** | Test mocks need updating | Fully reversible (move methods back) |
| E2 | **Medium** | Prediction store upload behavior changes | Each saver independently testable |
| E3 | **Medium** | prepare_actuals_df() override breaks | Delegation method preserves compat |
| E4 | **Medium** | Subprocess/reconciliation behavior changes | Each strategy independently testable |
| E5 | **Low** | N/A (deferred) | N/A |

---

## Definition of Done (per phase)

- [ ] All 899+ tests pass
- [ ] Ruff clean on all modified files
- [ ] No behavioral change (identical outputs for identical inputs)
- [ ] No subclass contract changes (hydranet, stepshifter, baseline unaffected)
- [ ] No on-disk format changes (parquet contract preserved)
- [ ] No API changes for downstream consumers (faoapi, ensemble)
