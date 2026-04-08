# Architecture Decomposition Report: views-pipeline-core

**Date:** 2026-04-07
**Branch:** `feature/samples_for_fao`
**Author:** Simon (with Claude Code assistance)

---

## TL;DR

`ForecastingModelManager` was a 3,049-line "god class" that did everything: training, evaluation, forecasting, reporting, persistence, path resolution, and WandB lifecycle management. Over the past development cycle, it has been decomposed into a **~1,960-line thin facade** that delegates to **5 independently testable stage classes** and a **relocated path manager**. All existing import paths and APIs are backward-compatible. Downstream model repos (`views-models`) require **zero changes**.

---

## What Changed

### Before (single monolith)

```
ForecastingModelManager (3,049 LOC)
├── Path resolution (ModelPathManager)
├── Training orchestration
├── Evaluation orchestration (226 LOC, dual DF/PF paths)
├── Forecasting orchestration
├── Report generation
├── Prediction persistence
├── WandB lifecycle management
└── Configuration merging
```

Every piece of logic was a private method on one class. Testing required mocking the entire manager. Adding a feature to evaluation risked breaking forecasting.

### After (facade + stages)

```
ForecastingModelManager (~1,960 LOC, thin facade)
├── Constructs frozen context objects
├── Calls abstract methods (subclass-specific)
├── Delegates results to stage classes
└── Manages WandB run lifecycle

Independently testable stages:
├── PredictionIOManager    (E1) — save/load predictions & evaluations
├── EvaluationStage        (E2) — metrics orchestration, step mappings
├── ReportingStage         (E3) — HTML report generation
├── ForecastingStage       (E4) — forecast validation & persistence
└── TrainingStage          (E5) — training log creation & alerts

Relocated:
└── ModelPathManager       (E6) — moved from managers/ to data/
    (re-export shim preserves all existing imports)
```

---

## Why This Was Done

Five structural problems were identified (documented in [ADR-045](../documentation/ADRs/045_pipeline_stage_architecture.md)):

1. **Inverted dependencies** — `ModelPathManager` lived in `managers/` but lower layers (`data/`, `modules/`) imported upward from it, creating architectural coupling.
2. **No pipeline abstraction** — The seven `_execute_*` methods were private methods, not independently callable or testable units.
3. **No stage context** — Each method pulled what it needed from `self`, with no declared contract for what a stage requires.
4. **Lifecycle entanglement** — Every method interleaved WandB lifecycle with business logic.
5. **Push-based abstract contracts** — Subclass abstract methods received no parameters and had to reach into the god class internals.

Root causes #1, #2, and #3 are now resolved. #4 and #5 are documented for future work.

---

## What This Means For You

### If you work on `views-models` (downstream model repos)

**Nothing changes.** All existing imports work. Your `main.py`, your `MyModelManager` subclass, your `ModelPathManager("purple_alien")` calls — all identical. The re-export shim ensures backward compatibility.

### If you work on `views-pipeline-core`

- **New stage logic** goes into the appropriate stage class (`managers/evaluation/stage.py`, `managers/reporting/stage.py`, etc.), not into `model.py`.
- **Each stage has its own test file** with focused unit tests (e.g., `test_evaluation_stage.py` has 17 tests).
- **Frozen context objects** (`EvaluationContext`, `ReportingContext`, etc.) are immutable dataclasses — stages cannot mutate pipeline state.
- **`model.py` is still ~1,960 LOC** because `_execute_model_evaluation()` (226 LOC, dual DF/PF paths) remains in the facade. It has 10 characterization tests and is tracked for future extraction.

### If you review PRs

The pattern for each extraction is consistent:
1. Frozen context dataclass created
2. Stage class extracted with explicit dependencies
3. Facade method becomes a thin delegate (constructs context, calls stage)
4. Existing tests pass without modification
5. New stage-specific tests added

---

## Extraction Summary

| ID | Stage | LOC | Tests | Commit |
|----|-------|-----|-------|--------|
| E1 | PredictionIOManager | ~200 | Existing | `017c85a` |
| E2 | EvaluationStage | ~225 | 17 + 10 characterization | See branch |
| E3 | ReportingStage | ~220 | 11 | See branch |
| E4 | ForecastingStage | ~148 | 10 | See branch |
| E5 | TrainingStage | ~71 | 7 | See branch |
| E6 | ModelPathManager relocation | ~875 (moved) | Existing (993 pass) | See branch |

**Total new test coverage:** 55+ tests across stage files.

---

## File Locations

| Component | Path |
|-----------|------|
| Facade | `views_pipeline_core/managers/model/model.py` |
| ModelPathManager (canonical) | `views_pipeline_core/data/model_path.py` |
| EvaluationStage | `views_pipeline_core/managers/evaluation/stage.py` |
| ReportingStage | `views_pipeline_core/managers/reporting/stage.py` |
| ForecastingStage | `views_pipeline_core/managers/forecasting/stage.py` |
| TrainingStage | `views_pipeline_core/managers/training/stage.py` |
| PredictionIOManager | `views_pipeline_core/managers/prediction/io.py` |
| Shared types | `views_pipeline_core/types.py` (`BaseStageContext`, `ModelPathProtocol`) |
| Architecture decision | `documentation/ADRs/045_pipeline_stage_architecture.md` |
| Risk register | `reports/technical_risk_register.md` |

---

## Known Remaining Work

- **`_execute_model_evaluation()`** (226 LOC) is the most complex unextracted method. It has 10 characterization tests and is tracked as C-34 in the risk register.
- **Abstract method context parameters** — subclass abstract methods still receive no explicit context. Deferred until downstream repos are ready to update.
- **Pipeline composition container** — a `Pipeline` class that composes stages is now feasible but not yet prioritized.
- **DF path retirement** — the legacy DataFrame evaluation path duplicates the PredictionFrame path. No retirement timeline set.

---

## Questions?

- **Design rationale:** [ADR-045](../documentation/ADRs/045_pipeline_stage_architecture.md)
- **Class contracts:** `documentation/CICs/ForecastingModelManager.md`, `documentation/CICs/ModelPathManager.md`
- **Risk tracking:** `reports/technical_risk_register.md`
- **Test files:** `tests/test_managers/test_*_stage.py`
