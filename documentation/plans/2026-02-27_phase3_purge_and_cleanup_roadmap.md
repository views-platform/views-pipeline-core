# Roadmap: Phase 3 — The Purge & Cleanup

**Date**: 2026-02-27
**Branch context**: `feature/samples_for_fao` (Phase 2 complete, ready for PR)
**Owner**: Simon Polichinel von der Maase

---

## Context

The evaluation transport refactor (`feature/samples_for_fao`) has completed Phase 2:
the orchestrator now owns alignment, runs a shadow parity audit, and all correctness
invariants are proven. 708/708 tests pass, ruff clean.

This document captures the remaining work across three repositories before the
"Pure Math Engine" vision is fully realised.

---

## Migration Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dual-Entry Support in `views-evaluation` (EvaluationManager accepts `ef=`) | ✅ Done |
| 2 | Orchestrator Integration in `views-pipeline-core` (shadow run + parity audit) | ✅ Done |
| 3 | The Purge (remove legacy Pandas path; remove dual-track; migrate HydraNet) | ❌ TODO |

---

## TODO: Phase 3

### [ ] 3A — Remove dual-track from pipeline-core
**Repo**: `views-pipeline-core`
**File**: `views_pipeline_core/managers/model/model.py`
**What**: Once production parity is confirmed across all models, remove `_audit_parity()`
and collapse `_evaluate_prediction_dataframe` to a single (shadow) path. Pass only `ef`
to `EvaluationManager.evaluate()`.
**Dependency**: Production parity confirmed (observational — requires live runs).

---

### [ ] 3B — Library purge in views-evaluation
**Repo**: `views-evaluation`
**What**: Delete the legacy Pandas alignment path from `EvaluationManager.evaluate()`.
Remove `pandas` as a hard dependency from the package.
**Dependency**: Phase 3A complete.

---

### [ ] 3C — HydraNet migration in views-models
**Repo**: `views-models` (purple_alien / HydraNet)
**What**: Update HydraNet inference to return a `PredictionFrame` directly, bypassing
the "Pandas Sandwich" (dense tensor → list-in-cell → DataFrame → re-extract to numpy).
**Dependency**: Can proceed independently of 3A/3B.

---

## TODO: Pre-existing Technical Debt (pipeline-core)

These issues predate this refactor and belong to future phases. They are not blocking
any Phase 3 work but should be tracked.

### [ ] Fix audit_suite.py (stale artifact)
**File**: `audit_suite.py` (repo root — ad-hoc script, not pytest)
**What**: Tests G1, G2, R1 call `ForecastingModelManager._get_conflict_type()`, a method
removed in the "Target Name Only" refactor. Script crashes; `AUDIT_REPORT.md` is stale.
**Action**: Replace G1/G2/R1 with tests that exercise the new explicit-config path.

### [ ] Resolve ADR-031 violations — ensemble aggregator
**Files**: `views_pipeline_core/modules/ensemble_aggregator/aggregator.py`,
`views_pipeline_core/managers/ensemble/ensemble.py`
**What**: `_detect_prediction_shape()` infers "point vs distribution" from list-in-cell
length at runtime. Violates ADR-031 (No-Sniffing Rule).
**Action**: Add explicit `prediction_type` field to ensemble config; remove runtime detection.

### [ ] Resolve ADR-031 violations — handlers.py
**File**: `views_pipeline_core/data/handlers.py`
**What**: Tensor semantics (n_time, n_entities, n_samples, n_vars) inferred from `.shape`
at lines 531, 678, 950, 1220, 1368, 1460 rather than declared through config.
**Action**: Pass tensor shape metadata explicitly through the call chain from config.
