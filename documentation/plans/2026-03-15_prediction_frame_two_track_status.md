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

---

## Remaining Work (4 items)

### Item 1: Remove parity proving mode

**What**: The DF path in `_evaluate_prediction_dataframe()` runs evaluation TWICE — once via legacy path (`ef=None`), once via EvaluationFrame (`ef=ef`) — then compares. This is transition scaffolding, not first-class behavior.

**Where**: `views_pipeline_core/managers/model/model.py`
- Lines ~3053-3068: Dual execution + audit call
- Lines ~3321-3362: `_audit_parity()` method

**Change**: DF path should use EvaluationFrame directly (single execution), matching PF path. Delete `_audit_parity()`.

**Risk**: Low — parity has been proven across all 7 PF models.
**Effort**: ~30 lines to remove, update affected tests.
**Blocked by**: Nothing — can do now.

---

### Item 2: Handle prediction store + Arrow path explicitly

**What**: PF evaluation saves via Arrow `pq.write_table()` but skips Appwrite upload with a TODO comment (line ~2876). Silent failure if PF + prediction store requested.

**Where**: `views_pipeline_core/managers/model/model.py`, line ~2876

**Change**: Add explicit guard — if `self._use_prediction_store` and prediction is `pa.Table`, raise `NotImplementedError` with clear message. Or implement upload if needed.

**Risk**: Very low — no production model uses PF + prediction store today.
**Effort**: ~10 lines.
**Blocked by**: Nothing — can do now.

---

### Item 3: Remove TEMPORARY transformation undo from pipeline-core

**What**: Both DF and PF forecast paths undo log-transforms before saving. This is marked TEMPORARY because **transformations are now the model repo's responsibility** — models must return predictions in the original scale. views-hydranet already does this correctly. views-r2darts2, views-stepshifter, and views-baseline should follow.

**Where**: `views_pipeline_core/managers/model/model.py`
- Lines ~2330-2336: PF forecast path — `# TEMPORARY: Undo transformations before saving per target.`
- Lines ~2355-2361: DF forecast path — `# TEMPORARY: Undo transformations before saving.`
- Lines ~2577-2583: Forecast reporting — commented-out undo block
- Lines ~2600-2610: Forecast reporting — active undo + target name update

**Change**: Remove all 4 TEMPORARY blocks. Models must return untransformed predictions. If a model returns transformed predictions, that's a model bug, not a pipeline-core responsibility.

**Risk**: Medium — if any model repo still relies on pipeline-core to undo transforms, removing this will break its forecasts. Must verify each engine repo first.
**Effort**: ~40 lines to remove.
**Blocked by**: Verification that views-r2darts2, views-stepshifter, and views-baseline undo transforms before returning predictions. views-hydranet confirmed OK.

---

### Item 4: Ensemble PF awareness (optional, Tier 2)

**What**: `EnsembleManager` only produces/consumes DataFrames. No `prediction_format` config, no PF dispatch.

**Why it works today**: Ensembles consume saved parquet files (list-in-cell format). Both DF and PF models save parquet. Functional compatibility is fine.

**Why it's not urgent**: The parquet contract is the integration point, not the in-memory format. Ensembles don't need to know about PF to aggregate predictions.

**When to do**: When/if an ensemble needs to produce PredictionFrame output (e.g., for a PF-native downstream consumer). Not now.

**Blocked by**: Design decision on how ensembles should handle multi-sample PredictionFrames (concatenate samples? Average? This is an open research question).

---

## Dependency Graph

```
Item 1 (parity proving)      ← blocked by nothing, do now
Item 2 (store + Arrow)       ← blocked by nothing, do now
Item 3 (transform undo)      ← blocked by verifying r2darts2, stepshifter, baseline
Item 4 (ensemble PF)         ← blocked by design decision, defer
```

## Scope Boundaries (Mission Creep Guard)

**IN SCOPE for this refactor:**
- Items 1-3 above
- Any bug fixes discovered during these changes
- Test updates for changed behavior

**OUT OF SCOPE (do not start):**
- Migrating models from DF to PF format (that's views-models work)
- Redesigning the transformation architecture
- Adding PF support to EnsembleManager
- Removing `from_dataframes()` from EvaluationAdapter (both tracks are permanent)
- Removing CorePredictionSniffer (DF track still needs it)
- Removing `to_prediction_df()` from PredictionFrameConverter (permanent API)

## How to Verify "Done"

When Items 1-3 are complete:
- Both tracks have identical structure: model → validate → save → EvaluationFrame → metrics
- No TEMPORARY markers remain in forecast/evaluation paths
- No dual-execution audit code remains
- No silent failure paths for prediction store
- All tests pass
- A DF-format model (e.g., fancy_feline) works end-to-end
- A PF-format model (e.g., purple_alien) works end-to-end
