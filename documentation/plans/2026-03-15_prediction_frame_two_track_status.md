# PredictionFrame Adoption: Refactor Status & Remaining Work

## Purpose of This Document

Track exactly what remains in the PredictionFrame adoption refactor. Prevent stalling, forgotten cleanup, and mission creep. This is the single source of truth for "how far are we from done."

## Goal

Two clean, first-class parallel tracks from model output → evaluation/forecasting:
- **Track A (DataFrame)**: Models return `List[pd.DataFrame]` / `pd.DataFrame`
- **Track B (PredictionFrame)**: Models return `Dict[str, List[PredictionFrame]]` / `Dict[str, PredictionFrame]`

Both must be equally clean, well-tested, and production-ready. Neither is deprecated — both are permanent (until a future decision says otherwise).

---

## Completed Work

| Item | Status | Commit/Evidence |
|------|--------|----------------|
| PredictionFrame class with save/load/collapse | Done | `c8130c3` |
| EvaluationAdapter with from_prediction_frames() | Done | `b55336a` |
| PredictionFrameConverter (PF↔DF/Arrow adapter) | Done | `7e57c3f`, `4b57eca` |
| CoreConfigSniffer: prediction_format validation | Done | `bd9e25b` |
| Streaming evaluation sink (bounded memory) | Done | `c8130c3` |
| Zero-copy Arrow parquet path | Done | `4afb0ba` |
| Type enforcement guards at all dispatch points | Done | `18a89ee` |
| Multi-target PF dict interface | Done | `18a89ee` |
| PipelineConfig singleton refactor | Done | `d7f3ad2` |
| Dead code removal (350+ lines) | Done | `d7f3ad2` |
| F401/F811 lint cleanup | Done | `d7f3ad2` |
| DoD Issues 3, 5, 8 closed | Done | `d7f3ad2` |
| Off-by-one fix (_resolve_evaluation_sequence_number) | Done | `463b413` |
| TYPE_CHECKING for WandBModule forward-refs | Done | `463b413` |
| **Item 1: Remove parity proving mode** | **Done** | DF path now uses single `evaluate(ef=ef)` call. `_audit_parity()` method deleted. 6 test assertions updated. |
| **Item 2: Arrow + prediction store guard** | **Done** | Explicit `NotImplementedError` raised when PF + prediction store requested. TODO comment removed. |

---

## Remaining Work (2 items)

### Item 3: Remove TEMPORARY transformation undo from pipeline-core — BLOCKED

**What**: Both DF and PF forecast paths undo log-transforms before saving. This is marked TEMPORARY because **transformations are now the model repo's responsibility** — models must return predictions in the original scale. views-hydranet already does this correctly. views-r2darts2, views-stepshifter, and views-baseline should follow.

**Where**: `views_pipeline_core/managers/model/model.py`
- Lines ~2326-2336: PF forecast path — `# TEMPORARY: Undo transformations before saving per target.`
- Lines ~2351-2361: DF forecast path — `# TEMPORARY: Undo transformations before saving.`
- Lines ~2573-2579: Forecast reporting — commented-out undo block
- Lines ~2596-2610: Forecast reporting — active undo + target name update

**Change**: Remove all 4 TEMPORARY blocks. Models must return untransformed predictions.

**Risk**: Medium — if any model repo still relies on pipeline-core to undo transforms, removing this will break its forecasts.
**Blocked by**: Verification that views-r2darts2, views-stepshifter, and views-baseline undo transforms before returning predictions. views-hydranet confirmed OK.

---

### Item 4: Ensemble PF awareness (optional, Tier 2) — DEFERRED

**What**: `EnsembleManager` only produces/consumes DataFrames. No `prediction_format` config, no PF dispatch.

**Why it works today**: Ensembles consume saved parquet files (list-in-cell format). Both DF and PF models save parquet.

**Blocked by**: Design decision on how ensembles should handle multi-sample PredictionFrames.

---

## Scope Boundaries (Mission Creep Guard)

**IN SCOPE for this refactor:**
- Item 3 above (once unblocked)
- Any bug fixes discovered during that change
- Test updates for changed behavior

**OUT OF SCOPE (do not start):**
- Migrating models from DF to PF format (that's views-models work)
- Redesigning the transformation architecture
- Adding PF support to EnsembleManager
- Removing `from_dataframes()` from EvaluationAdapter (both tracks are permanent)
- Removing CorePredictionSniffer (DF track still needs it)
- Removing `to_prediction_df()` from PredictionFrameConverter (permanent API)

## How to Verify "Done"

When Item 3 is complete:
- Both tracks have identical structure: model → validate → save → EvaluationFrame → metrics
- No TEMPORARY markers remain in forecast/evaluation paths
- No dual-execution audit code remains (already done)
- No silent failure paths for prediction store (already done)
- All tests pass
- A DF-format model (e.g., fancy_feline) works end-to-end
- A PF-format model (e.g., purple_alien) works end-to-end
