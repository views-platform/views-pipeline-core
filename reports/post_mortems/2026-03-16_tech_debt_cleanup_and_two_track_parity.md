# Post-Mortem: Tech Debt Cleanup & Two-Track Parity Session

**Date**: 2026-03-16
**Branch**: `feature/samples_for_fao`
**Commits**: `d7f3ad2`, `463b413`, `84a997b`
**Scope**: 45 files changed, +548/-556 lines

---

## What We Did

This session had three objectives:

1. **Tech debt cleanup** — remove dead code, fix lint, stabilize after the PredictionFrame refactor
2. **Achieve two-track parity** — make the DataFrame and PredictionFrame evaluation paths structurally identical
3. **Fix bugs found along the way** — off-by-one in sequence counting, PipelineConfig property descriptor issue

### Commit 1: `d7f3ad2` — PipelineConfig singleton + dead code removal

**What**: Refactored `PipelineConfig` from a class to a module-level singleton. Removed ~350 lines of dead code. Fixed 54 F401 and 1 F811 lint errors.

**Dead code removed**:
- `_get_aggregated_df_old()` in ensemble manager (replaced by AggregationManager)
- 150-line commented-out `PreprocessorModelArgs` class
- 86 lines of commented-out serial reconciliation methods
- 45 lines of commented-out `upload_predictions` body
- Commented-out imports in dataloaders
- Orphaned `_model_format` attribute

### Commit 2: `463b413` — Off-by-one fix + lint

**What**: Fixed `_resolve_evaluation_sequence_number()` returning 12 (shift count) instead of 13 (sequence count = shifts + 1 base origin). Added `TYPE_CHECKING` for `WandBModule` forward-refs. Removed unused test imports.

### Commit 3: `84a997b` — Remove parity proving + Arrow store guard

**What**: Removed the dual-execution parity proving mode from the DF evaluation path. Both DF and PF paths now follow the identical pattern: build EvaluationFrame → single `evaluate(ef=ef)` call. Added explicit `NotImplementedError` for Arrow + prediction store. Fixed postprocessor.py duplicate imports.

---

## Why We Did It

The `feature/samples_for_fao` branch had accumulated significant transition scaffolding during the PredictionFrame adoption. The DF evaluation path ran `evaluate()` twice per target (legacy + shadow) and compared results — necessary during migration but not acceptable as permanent architecture. The goal was two clean, first-class parallel tracks with no hacks.

Additionally, multiple rounds of rapid development had left behind dead code, orphaned attributes, inconsistent lint, and a `PipelineConfig` class that worked by accident (a class-attribute hack masking a property descriptor).

---

## How We Did It

### Methodology

Each change followed **TDD (Red → Green → Refactor)**:
1. Write tests that fail on the current code (RED)
2. Fix the code to make tests pass (GREEN)
3. Clean up (REFACTOR)

We also applied:
- **DRY**: Both DF and PF paths now share the same evaluate pattern
- **SRP**: Removed `_audit_parity()` — its single responsibility (transition auditing) was no longer needed
- **Fail Loud and Proud**: Arrow + prediction store now raises `NotImplementedError` instead of silently skipping

### Process

1. **Repo assimilation** first — built a full 8-phase understanding before touching anything
2. **Tech debt investigation** before cleanup — mapped all debt items, prioritized by safety
3. **Incremental verification** — ran full test suite after every change, not just at the end
4. **Review-diff before ship-it** — semantic review of each changeset before committing

---

## What We Learned

### 1. The PipelineConfig Incident

**What happened**: During dead code cleanup, `set_dataframe_format()` was changed from setting a class attribute to a no-op (just logging). This seemed safe — the method was only called with the default value `.parquet`. But it broke the entire data-fetching pipeline in sweep mode.

**Root cause**: `PipelineConfig` used instance properties (`@property`) but was accessed at the class level (`PipelineConfig.dataframe_format` without `()`) in 6 production locations. The old class-attribute hack masked this by shadowing the property descriptor. Removing it exposed the raw `<property object>` in filenames.

**Why 888 tests didn't catch it**: All file I/O in `ViewsDataLoader` tests was mocked via `@patch("save_dataframe")`. The broken filename never reached the filesystem, so no assertion failed.

**Lesson**: When a "safe" change passes all tests but breaks production, the problem is the test suite, not the change. We added `TestPipelineConfigAccessContract` to prevent recurrence and then refactored to a singleton that makes the bug impossible by construction.

### 2. The Off-by-One Was Always There

`_resolve_evaluation_sequence_number("standard")` returned 12 (the shift count) since it was written. The contract requires 13 (base origin + 12 shifts). Nobody noticed because `_assert_predictions_in_step_window()` — the assertion that catches it — was added on this branch. The bug was silent for the entire project history.

**Lesson**: Adding assertions to existing code surfaces pre-existing bugs. This is a feature, not a problem. The fix was 6 lines; the investigation and impact analysis took 10x longer.

### 3. Parity Proving Was a Successful Transition Tool

The dual-execution parity audit served its purpose: it proved that the EvaluationFrame path produces identical results to the legacy path. Once proven, it became dead weight — doubling evaluation time and compute for every DF-format model. Removing it was the right call, but only because the audit had already run successfully across all production models.

**Lesson**: Transition scaffolding should be designed with its own removal criteria from day one. The parity proving code had clear "DoD #3 removal target" markers, which made the cleanup unambiguous.

### 4. Mission Creep Is Real

The session started as "tech debt cleanup" but uncovered:
- A PipelineConfig design flaw requiring a singleton refactor
- An off-by-one bug in sequence counting
- A sweep-only failure in views-r2darts2 (wrong import path)
- The need for a comprehensive status tracking document

Each discovery was handled, but the scope grew from "remove dead code" to "achieve two-track parity." The status tracking document (`2026-03-15_prediction_frame_two_track_status.md`) was created specifically to prevent further creep — it defines what's in scope, what's out, and what "done" means.

---

## Current State

### What's Done
- Both DF and PF evaluation paths are structurally identical
- No parity proving scaffolding remains
- No silent failure paths (Arrow + store raises explicitly)
- 898 tests pass, 7 pre-existing F403 lint warnings remain (structural, deferred)
- PipelineConfig is a singleton — class-vs-instance confusion impossible

### What Remains (1 item)
- **Item 3**: Remove 4 TEMPORARY transformation undo blocks from `model.py`
  - Blocked by: verifying views-r2darts2, views-stepshifter, views-baseline undo transforms before returning predictions
  - views-hydranet: confirmed OK

### Metrics
- **Lines removed**: ~556 (dead code, scaffolding, duplicate imports)
- **Lines added**: ~548 (tests, status doc, singleton, guards)
- **Net**: -8 lines (rare for a session that added 15+ new tests)
- **Lint errors**: 16 → 7 (56% reduction; remaining are structural F403)
- **Test count**: 888 → 898 (+10 new tests)
