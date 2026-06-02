# PredictionFrameEnsembleManager Production Roadmap

**Date**: 2026-06-01
**Author**: views-pipeline-core team
**Branch**: `development` (pipeline-core); `feature/golden_hour_ensemble` (views-models)
**Supersedes**: `2026-03-03_prediction_frame_open_issues_remediation.md`, `2026-03-15_prediction_frame_two_track_status.md`

This is the single source of truth for getting PredictionFrameEnsembleManager (PFE)
into production use with HydraNet ensembles. It consolidates all investigation results,
open issues, risk register entries, and GitHub issue cross-references.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State of Play](#2-current-state-of-play)
3. [Repository Landscape](#3-repository-landscape)
4. [Architecture: How PFE Works End-to-End](#4-architecture-how-pfe-works-end-to-end)
5. [Status of the 8 Original Open Issues](#5-status-of-the-8-original-open-issues)
6. [Step 2: Single-Model PredictionFrame Probe](#6-step-2-single-model-predictionframe-probe)
7. [Step 3: Transform Undo Verification](#7-step-3-transform-undo-verification)
8. [Step 4: Ensemble Dry-Run](#8-step-4-ensemble-dry-run)
9. [Step 5: Production Ship](#9-step-5-production-ship)
10. [Risk Register Cross-References](#10-risk-register-cross-references)
11. [GitHub Issue Tracker](#11-github-issue-tracker)
12. [Superseded Documents](#12-superseded-documents)

---

## 1. Executive Summary

The goal is to run the **golden_hour** HydraNet ensemble using
`PredictionFrameEnsembleManager` in production forecast mode. This is the first
PredictionFrame-native ensemble to enter production.

**What has been completed:**
- ADR-054 extraction to views-reporting (11 PRs, merged to development via PR #103)
- All CI failures from the extraction fixed (PRs #129, #132, #133)
- Risk register entries C-140 through C-144 and D-22 registered (PR #135)
- PredictionFrameEnsembleManager exists and has tests (PR #82)
- 6 of 8 original open issues from the 2026-03-03 plan are FIXED
- golden_hour ensemble already configured with PFE in views-models
- All three constituent models (purple_alien, blue_stranger, violet_visitor) already
  declare `prediction_format: "prediction_frame"`

**What remains (4 steps):**
- Step 2: Run a single HydraNet constituent model with `prediction_format: "prediction_frame"` — verify it produces valid `.npy` output
- Step 3: Verify transform undo responsibility — confirm no double-undo or missing-undo
- Step 4: Run golden_hour ensemble with PFE in `--evaluate` mode — compare metrics to a legacy EnsembleManager baseline
- Step 5: Run with `--forecast` and ship

**Important: We are NOT retiring EnsembleManager or DataFrameEnsembleManager.** All three
ensemble managers coexist permanently. PFE is an addition, not a replacement.

---

## 2. Current State of Play

### Completed milestones

| Milestone | Date | Evidence |
|-----------|------|----------|
| PredictionFrame class (save/load/collapse) | 2026-02 | `c8130c3` |
| EvaluationAdapter with from_prediction_frames() | 2026-02 | `b55336a` |
| PredictionFrameConverter (PF↔DF/Arrow) | 2026-02 | `7e57c3f`, `4b57eca` |
| CoreConfigSniffer: prediction_format validation | 2026-02 | `bd9e25b` |
| PredictionFrameEnsembleManager implementation | 2026-03 | PR #82 |
| model.py PF dispatch (eval + forecast paths) | 2026-03 | `18a89ee` |
| Off-by-one fix in _resolve_evaluation_sequence_number | 2026-03 | `463b413` |
| ADR-054 extraction to views-reporting (11 PRs) | 2026-05 | PR #103 |
| CI fix: importorskip guards on all test files | 2026-05 | PRs #129, #132, #133 |
| Risk register entries C-140..C-144, D-22 | 2026-05 | PR #135 |
| TEMPORARY transform undo blocks removed | 2026-04 | Memory note: `project_transform_undo_investigation.md` |

### What is blocked and why

The production path is NOT blocked by any external dependency. The remaining steps
are sequential validation: probe → verify → dry-run → ship. Each step takes hours,
not days.

The only prerequisite is that at least one HydraNet constituent model has completed
training. The user has confirmed one is currently training.

---

## 3. Repository Landscape

Sixteen repositories exist under views-platform. Seven are relevant to PFE production.

### Directly affected repos

| Repository | Role in PFE | What needs to happen | Splash zone |
|------------|-------------|----------------------|-------------|
| **views-pipeline-core** | Defines PredictionFrame, all 3 ensemble managers, model.py dispatch | Nothing — already done. PFE code is merged on `development`. | N/A (already landed) |
| **views-models** | Ensemble configs + main.py, model configs | golden_hour already uses PFE on `feature/golden_hour_ensemble`. Merge to main after Step 5. | `ensembles/golden_hour/main.py`, `models/purple_alien/configs/config_meta.py`, `models/blue_stranger/configs/config_meta.py`, `models/violet_visitor/configs/config_meta.py` |
| **views-hydranet** | HydraNet model implementation, produces PredictionFrames | Nothing — already returns `Dict[str, List[PredictionFrame]]` from `HydranetManager`. Transform undo happens in `FeatureScaler.inverse_transform_volume()`. | Verify only: `views_hydranet/utils/feature_scaler.py:196-259` |

### Indirectly affected repos (watch for regressions)

| Repository | Why it matters | Regression risk |
|------------|---------------|-----------------|
| **views-postprocessing** | References EnsembleManager but NOT PredictionFrame. If postprocessing scripts hardcode EnsembleManager assumptions, PFE output format may not be consumable. | LOW — PFE saves `.npy` files alongside parquet (Track B). Postprocessing reads parquet. As long as `skip_predictions_delivery=False`, parquet exists. |
| **views-baseline** | Has 11 PredictionFrame references, produces PFs. If baseline models join PFE ensembles in the future, they need compatible PF output. | LOW — not part of golden_hour. Future concern only. |
| **views-stepshifter** | Has 2 PredictionFrame references. Legacy step-shift models may eventually produce PFs. | NEGLIGIBLE — not part of golden_hour. |
| **views-r2darts2** | Zero PredictionFrame references, imports pipeline-core. | NEGLIGIBLE — pure DataFrame path. |

### Not affected

| Repository | Why not |
|------------|---------|
| views-reporting | Extraction target. PFE's core path has zero views-reporting dependency. Reporting is optional post-hoc. |
| views-evaluation | Consumed via EvaluationStage composition. PFE delegates; no changes needed. |
| views-dataloader | Upstream data fetch. Format-agnostic. |
| views-transformations | Transform definitions. Models (not pipeline-core) call these. |
| views-apps | Web frontend. Consumes final outputs, not intermediate PFs. |
| views-dataviz | Visualization. Downstream only. |
| views-deploy | Deployment scripts. Needs no PF awareness. |
| views-infra | Infrastructure. No code changes. |
| views-docs | Documentation site. Update after production is proven. |

---

## 4. Architecture: How PFE Works End-to-End

### 4.1 Single-model PredictionFrame flow (what Step 2 tests)

This is what happens when a HydraNet model runs with `prediction_format: "prediction_frame"`:

```
┌─────────────────────────────────────────────────────────────────┐
│ ModelManager.execute_single_run(args)                           │
│                                                                 │
│  1. CoreConfigSniffer.sniff_all(run_type)                       │
│     - validates prediction_format ∈ {"dataframe","prediction_   │
│       frame"}                                                   │
│     - validates skip_predictions_delivery is bool               │
│                                                                 │
│  2. _execute_data_fetching()                                    │
│     - ViewsDataLoader fetches VIEWSER data                      │
│     - CoreDataSniffer validates the DataFrame                   │
│                                                                 │
│  3. _execute_model_evaluation()                                 │
│     ┌──────────────────────────────────────────────────────┐    │
│     │ self._prediction_format == "prediction_frame"         │    │
│     │                                                       │    │
│     │ _evaluate_model_artifact_streaming(eval_type, ...)    │    │
│     │   → calls model's _evaluate_model_artifact()          │    │
│     │   → model returns Dict[str, List[PredictionFrame]]    │    │
│     │   → origin_sink() saves per-origin:                   │    │
│     │       Track A:  _pf_staging/origin_i/target/ (.npy)   │    │
│     │       Track A+: data_generated/predictions_*/origin_  │    │
│     │                 i/target/ (.npy) — permanent           │    │
│     │       Track B:  predictions_*.parquet (list-in-cell)  │    │
│     │                 — only if skip_predictions_delivery    │    │
│     │                   = False                              │    │
│     │   → gc.collect() after each origin                    │    │
│     │                                                       │    │
│     │ Metrics reload:                                       │    │
│     │   PredictionFrame.load(staging, mmap=True) per target │    │
│     │   → _evaluate_prediction_dataframe(pf_dict, ...)      │    │
│     │   → EvaluationStage computes metrics                  │    │
│     │   → shutil.rmtree(staging_path)  # cleanup Track A    │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                 │
│  4. _execute_model_forecasting()                                │
│     ┌──────────────────────────────────────────────────────┐    │
│     │ predictions = _forecast_model_artifact(artifact_name) │    │
│     │ → model returns Dict[str, PredictionFrame]            │    │
│     │ → pf.save(data_generated/predictions_*/target/)       │    │
│     │ → ForecastingStage.process_and_save_forecast(...)     │    │
│     └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Repo affected**: views-pipeline-core (model.py dispatch), views-hydranet (model implementation)

### 4.2 HydraNet model internals (what produces the PredictionFrames)

```
┌─────────────────────────────────────────────────────────────────┐
│ HydranetManager._evaluate_model_artifact()                      │
│   (views-hydranet/views_hydranet/manager/hydranet_manager.py)   │
│                                                                 │
│  1. InferenceOrchestrator runs forward pass                     │
│     → produces raw inference volumes (numpy arrays)             │
│                                                                 │
│  2. FeatureScaler.inverse_transform_volume()                    │
│     (views_hydranet/utils/feature_scaler.py:196-259)            │
│     → undoes log1p via np.expm1                                 │
│     → predictions are now in ORIGINAL SCALE                     │
│                                                                 │
│  3. PredictionFrameAssembler.assemble_evaluation()              │
│     (views_hydranet/utils/prediction_frame_assembler.py)        │
│     → constructs PredictionFrame objects from numpy volumes     │
│     → sets identifiers: {"time": [...], "unit": [...]}          │
│                                                                 │
│  4. Returns: Dict[str, List[PredictionFrame]]                   │
│     key = target name (e.g., "lr_sb_best")                      │
│     value = list of PFs, one per rolling-origin sequence         │
└─────────────────────────────────────────────────────────────────┘
```

**Repo affected**: views-hydranet (read-only verification in Steps 2-3; no changes needed)

### 4.3 PredictionFrameEnsembleManager flow (what Step 4 tests)

```
┌─────────────────────────────────────────────────────────────────┐
│ PredictionFrameEnsembleManager.execute_single_run(args)         │
│   (views-pipeline-core/.../ensemble/prediction_frame_ensemble   │
│   .py)                                                          │
│                                                                 │
│  1. CoreConfigSniffer.sniff_all(run_type)                       │
│                                                                 │
│  2. _build_context() → frozen EnsembleContext                   │
│     models: [purple_alien, blue_stranger, violet_visitor]       │
│     aggregation: "concat"                                       │
│     reconciliation: None                                        │
│                                                                 │
│  3. _execute_model_tasks(ctx)                                   │
│     ├── _evaluate_ensemble(ctx)                                 │
│     │   For each seq_idx in range(n_sequences):                 │
│     │     For each model in ctx.models:                         │
│     │       _load_or_generate_pf(model, target, seq_idx)        │
│     │       → checks data_generated/predictions_*/origin_{idx}  │
│     │         /target/ for existing .npy                        │
│     │       → if missing: _execute_shell_script() to run model  │
│     │       → PredictionFrame.load(path, mmap=True|False)       │
│     │     _aggregate_prediction_frames(frames, "concat")        │
│     │       → np.concatenate([pf.y_pred for pf in frames],     │
│     │         axis=1)                                           │
│     │       → 3 models × 64 samples = 192 concatenated samples │
│     │     Save aggregated PF                                    │
│     │   _evaluate_predictions(aggregated, ctx)                  │
│     │     → EvaluationStage computes metrics                    │
│     │                                                           │
│     ├── _forecast_ensemble(ctx)                                 │
│     │   For each model in ctx.models:                           │
│     │     _load_or_generate_pf(model, target)                   │
│     │   _aggregate_prediction_frames(frames, "concat")          │
│     │   Save aggregated PF                                      │
│     │                                                           │
│     └── _execute_forecast_reporting(ctx) (optional)             │
│         → ReportingStage (requires views-reporting)             │
└─────────────────────────────────────────────────────────────────┘
```

**Repos affected**: views-pipeline-core (PFE code), views-models (golden_hour config + main.py)

### 4.4 The three output tracks

PredictionFrame models produce up to three output artifacts:

| Track | Format | Path pattern | Purpose | Lifetime |
|-------|--------|-------------|---------|----------|
| **A** (staging) | `.npy` + `.npz` | `_pf_staging/origin_i/target/` | Metrics mmap reload | Deleted after metrics computation |
| **A+** (permanent) | `.npy` + `.npz` | `data_generated/predictions_*/origin_i/target/` | PFE ensemble consumption | Permanent — this is what PFE reads |
| **B** (parquet) | Arrow parquet (list-in-cell) | `data_generated/predictions_*.parquet` | Legacy consumption (postprocessing, DF ensembles) | Permanent if `skip_predictions_delivery=False`; skipped if `True` |

**`skip_predictions_delivery`** controls only Track B. It is checked in exactly ONE
place: `model.py:1290`. When `True`, the expensive Arrow conversion + parquet write
is skipped — appropriate when the PFE ensemble is the only consumer.

**Transitional scaffolding**: Once PFE is proven in production and no downstream
consumer needs Track B parquet, `skip_predictions_delivery` can be removed entirely.
GitHub Issue #134 tracks this.

---

## 5. Status of the 8 Original Open Issues

These were identified in `2026-03-03_prediction_frame_open_issues_remediation.md`.

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 1 | `_assert_predictions_in_step_window()` crashes on PF | **FIXED** | `Union[List[pd.DataFrame], List[PredictionFrame]]` type hint + `isinstance` dispatch at model.py:1862-1865 |
| 2 | Sweep path has no PF dispatch | **FIXED** | Sweep path has PF dispatch (model.py sweep section) |
| 3 | Multi-target PF saves only first target | **FIXED** | PF save loop iterates all targets via `pf_dict.items()` in origin_sink |
| 4 | No parity audit in forecast PF path | **FIXED** | Structural audit exists in forecast PF path |
| 5 | No validation that PF's y_pred corresponds to requested target | **NOT FIXED** | Missing `logger.debug` about target correspondence. Trivial — convention-based, model author's responsibility per ADR-042 |
| 6 | `from_prediction_frame()` lacks I3 window integrity | **FIXED** | I3 window integrity check in place |
| 7 | Asymmetric `prediction_format` access | **FIXED** | Consistent `self._prediction_format` property (model.py:627-634) used everywhere |
| 8 | `_audit_parity_ef()` never tested end-to-end | **REMOVED** | Method was refactored into PredictionFrameConverter; parity audit removed (Item 1 in two-track status doc) |

**Issue 5 is the only unfixed item.** It's a single `logger.debug()` line and does not
block production. The convention is documented: model authors are responsible for ensuring
their PF's `y_pred` corresponds to the declared target.

---

## 6. Step 2: Single-Model PredictionFrame Probe

### Goal

Run one HydraNet constituent model (e.g., purple_alien) with
`prediction_format: "prediction_frame"` through evaluation. Verify it produces valid
`.npy` output files at the expected paths.

### Prerequisites

- Model has completed training (artifact exists)
- views-pipeline-core `development` branch is up to date (PR #103 merged)
- views-models `feature/golden_hour_ensemble` branch is available

### Execution

```bash
cd /path/to/views-models/models/purple_alien
python main.py --evaluate --run_type calibration
```

### What to verify

1. **Output files exist**:
   ```
   data_generated/predictions_calibration_<timestamp>/
     origin_0/lr_sb_best/y_pred.npy
     origin_0/lr_sb_best/identifiers.npz
     origin_0/lr_ns_best/y_pred.npy
     ...
     origin_12/lr_os_best/y_pred.npy
     origin_12/lr_os_best/identifiers.npz
   ```
   Expected: 13 origins (0..12) × 3 targets = 39 PredictionFrame directories.

2. **Array shapes are correct**:
   ```python
   import numpy as np
   y = np.load("origin_0/lr_sb_best/y_pred.npy")
   assert y.ndim == 2
   assert y.shape[1] == 64  # 64 posterior samples for HydraNet
   ```

3. **Identifiers are correct**:
   ```python
   ids = np.load("origin_0/lr_sb_best/identifiers.npz")
   assert "time" in ids
   assert "unit" in ids
   assert len(ids["time"]) == y.shape[0]
   ```

4. **Track B parquet exists** (if `skip_predictions_delivery=False`):
   ```
   data_generated/predictions_calibration_<timestamp>.parquet
   ```

5. **Metrics were computed** (check WandB or stdout for evaluation metrics).

### Splash zone (what could break)

| Component | Risk | Symptom |
|-----------|------|---------|
| `_evaluate_model_artifact_streaming()` in model.py | Model returns unexpected format | `TypeError` or `ValueError` at origin_sink |
| `PredictionFrame.save()` / `.load()` | Path construction wrong | `FileNotFoundError` at load time |
| `PredictionFrameConverter` (Track B) | Arrow conversion fails for HydraNet-specific data | `ArrowInvalid` or schema mismatch |
| `EvaluationStage` | PF metrics path untested with real HydraNet data | Metrics are NaN or missing |
| `CoreConfigSniffer` | Config missing required key | `KeyError` with descriptive message (Fail Loud) |

### Regression watch

- **DF models must still work.** After running the PF probe, run a DF model (e.g., `fancy_feline` if available) to confirm the DF path is unaffected.
- **WandB logging**: Verify that WandB run metadata captures `prediction_format` correctly.

---

## 7. Step 3: Transform Undo Verification

### Goal

Confirm that PredictionFrame predictions from HydraNet models are in the correct
(original, untransformed) scale. This resolves risk register entry C-140.

### Background

HydraNet models apply `log1p` transformation during training. The question is: who
undoes it?

**Answer (confirmed by investigation):** HydraNet undoes its own transforms.
`FeatureScaler.inverse_transform_volume()` in
`views-hydranet/views_hydranet/utils/feature_scaler.py:196-259` calls `np.expm1()`
(inverse of `log1p`) on the inference output BEFORE constructing PredictionFrames.

The TEMPORARY transform undo blocks that previously existed in pipeline-core's
model.py have been removed (confirmed in session memory:
`project_transform_undo_investigation.md`).

### Execution

After Step 2 produces PF output:

```python
import numpy as np
y = np.load("origin_0/lr_sb_best/y_pred.npy")

# Values should be in original scale (count-like, non-negative, not log-scale)
print(f"min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

# If values are in log scale (e.g., max < 5, mean around 1-2), something is wrong.
# If values are in original scale (e.g., max > 10, typical fatality counts), correct.
```

### What "correct" looks like

For UCDP best-estimate fatality counts (lr_sb_best):
- Values should be non-negative (fatality counts)
- Typical range: 0 to several hundred for high-conflict cells
- Mean across all cells: single digits (most cells have zero conflict)

If values look like `log(1 + count)` — i.e., compressed range 0–6ish — then the
inverse transform is NOT being applied, and we have a missing-undo problem.

### Repos affected

- **views-hydranet**: Read-only verification of `feature_scaler.py`
- **views-pipeline-core**: Confirm TEMPORARY blocks are gone (already done)

### If verification fails

If double-undo: HydraNet undoes + pipeline-core undoes → values are double-expm1'd.
Symptom: values are extremely large (exponential blowup). Fix: confirm pipeline-core
blocks are truly removed.

If missing-undo: Neither undoes → values are in log scale. Fix: verify
`FeatureScaler.inverse_transform_volume()` is actually called in the execution path.
Check `hydranet_manager.py` call chain.

---

## 8. Step 4: Ensemble Dry-Run

### Goal

Run the golden_hour ensemble with PredictionFrameEnsembleManager in `--evaluate`
mode. Compare metrics to a legacy EnsembleManager baseline (if available) or to
known-good single-model metrics.

### Prerequisites

- Step 2 passed (constituent models produce valid PF output)
- Step 3 passed (values are in correct scale)
- All three constituent models have been evaluated (PF output exists at expected paths)

### Execution

```bash
cd /path/to/views-models/ensembles/golden_hour
python main.py --evaluate --run_type calibration
```

### What to verify

1. **PFE loads constituent PFs successfully**: No `FileNotFoundError` or
   `PipelineException` from `_load_or_generate_pf()`.

2. **Aggregation produces correct shape**:
   ```python
   # golden_hour uses "concat" aggregation
   # 3 models × 64 samples = 192 samples per target
   import numpy as np
   y = np.load("data_generated/predictions_calibration_<ts>/origin_0/lr_sb_best/y_pred.npy")
   assert y.shape[1] == 192  # 3 × 64
   ```

3. **Evaluation metrics are computed**: Check WandB dashboard or stdout for CRPS,
   calibration metrics, etc.

4. **Metrics are reasonable**: Compare to individual model metrics. Ensemble should
   be equal or better than the worst constituent model.

### Splash zone

| Component | Risk | Symptom |
|-----------|------|---------|
| `_load_or_generate_pf()` | Path mismatch between where model saved and where PFE looks | `FileNotFoundError` → `PipelineException` |
| `_aggregate_prediction_frames()` | Identifier mismatch between models (different grid cells or time steps) | `ValueError: "identifiers do not match"` |
| `_aggregate_prediction_frames()` | Row count mismatch between models | `ValueError: "n_rows mismatch"` |
| `EvaluationStage` | Metrics computation fails on 192-sample PF | OOM or unexpected metric behavior |
| `CoreConfigSniffer` | Ensemble config missing required key | `KeyError` (Fail Loud) |
| `_execute_shell_script()` | Subprocess timeout (7200s) for model execution | `PipelineException` with timeout details |

### Regression watch

- **Other ensembles must still work.** After the PFE dry-run, spot-check that a
  legacy EnsembleManager ensemble (e.g., `cruel_summer`) still runs correctly.
- **PFE has no reconciliation support.** If reconciliation is needed in the future,
  it must be added to PFE or handled externally. Currently golden_hour has
  `reconciliation: None`.

### What if metrics diverge from expected?

If metrics are significantly worse than individual models:
1. Check aggregation method: `concat` preserves all samples; verify no samples are
   dropped or duplicated.
2. Check identifier alignment: all three models must produce PFs with identical
   `time` and `unit` identifier arrays.
3. Check that evaluation targets match ensemble config: golden_hour expects
   `lr_sb_best`, `lr_ns_best`, `lr_os_best`.

---

## 9. Step 5: Production Ship

### Goal

Run golden_hour ensemble with `--forecast` and deliver production predictions.

### Prerequisites

- Step 4 passed (evaluation metrics are acceptable)
- Deployment status updated from `"shadow"` to `"deployed"` in golden_hour's
  `config_deployment.py`

### Execution

```bash
cd /path/to/views-models/ensembles/golden_hour
python main.py --forecast
```

### What to verify

1. **Forecast PFs saved**:
   ```
   data_generated/predictions_forecasting_<ts>/lr_sb_best/y_pred.npy
   data_generated/predictions_forecasting_<ts>/lr_ns_best/y_pred.npy
   data_generated/predictions_forecasting_<ts>/lr_os_best/y_pred.npy
   ```

2. **Forecast PF shapes**:
   ```python
   y = np.load("lr_sb_best/y_pred.npy")
   assert y.shape[1] == 192  # 3 models × 64 samples
   assert y.shape[0] > 0     # spatial observations exist
   ```

3. **WandB run completes** with forecast metadata.

4. **Downstream consumers** (if any) can read the output. If
   `skip_predictions_delivery=False`, verify Track B parquet exists for legacy
   consumers.

### Post-ship actions

1. **Merge `feature/golden_hour_ensemble` to main in views-models.**
2. **Update golden_hour deployment_status** to `"deployed"`.
3. **Close GitHub Issues**: #123 (single-model probe), #124 (transform undo), #125
   (dry-run), #126 (production ship).
4. **Update risk register**: Resolve C-140 if transform undo is confirmed correct.
5. **Consider closing Issue #134** (skip_predictions_delivery removal) or scheduling
   it for the next cleanup sprint.

### Repos with post-ship changes

| Repository | Change | Why |
|------------|--------|-----|
| **views-models** | Merge `feature/golden_hour_ensemble` to main | golden_hour configs + main.py become the production version |
| **views-models** | Update `config_deployment.py`: `shadow` → `deployed` | Production status |
| **views-pipeline-core** | Update risk register (C-140 resolved) | Governance hygiene |
| **views-pipeline-core** | Update this roadmap (mark steps complete) | Single source of truth |

---

## 10. Risk Register Cross-References

Active risk register entries relevant to PFE production:

| ID | Tier | Title | Step that resolves it |
|----|------|-------|-----------------------|
| C-140 | 2 | Transform undo ambiguity — PF models may double-undo or miss undo | Step 3 (per-model verification) |
| C-141 | 3 | Concat ordering in `_aggregate_prediction_frames()` — model order not guaranteed | Non-blocking; fix post-production |
| C-142 | 3 | Resolved — importorskip guards added | Already resolved (PRs #129, #132, #133) |
| C-143 | 4 | Mitigated — reconciliation pre-flight check reordered | Already mitigated (PR #133) |
| C-144 | 3 | Undeclared views-reporting runtime dependency in pyproject.toml | Non-blocking for PFE (PFE has zero views-reporting dependency). GitHub Issue #130 tracks fix. |
| D-22 | — | Resolved — speed vs knowledge disagreement about extraction sequencing | Resolved |

### Risks NOT in the register but worth watching

- **OOM during 192-sample evaluation**: golden_hour concatenates 3 × 64 = 192 posterior
  samples. Evaluation metrics on 192-sample PFs have not been tested at scale. If OOM
  occurs, the fix is to compute metrics per-model then aggregate, or use mmap loading.
  This is what C-66 (OOM on DF aggregation) was about — PFE was built to solve it via
  numpy-native operations, but the 192-sample case has not been profiled.

- **Subprocess timeout**: PFE runs constituent models via subprocess with a 7200s (2hr)
  timeout. If a HydraNet model takes longer than 2 hours, the ensemble will fail with
  a timeout error. This is a deployment concern, not a code bug.

---

## 11. GitHub Issue Tracker

Open issues relevant to PFE production, in execution order:

| Issue | Title | Blocked by | Status |
|-------|-------|------------|--------|
| #122 | Merge PR #103: land extraction on development | — | **DONE** (PR merged; issue needs closing) |
| #123 | Probe: run single HydraNet model with prediction_format: prediction_frame | Model training completion | **NEXT** |
| #124 | Verify transform undo responsibility for PF models (C-140) | #123 | Pending |
| #125 | Dry-run: HydraNet ensemble with PredictionFrameEnsembleManager | #124 | Pending |
| #126 | Ship: PredictionFrameEnsembleManager in production forecast | #125 | Pending |
| #130 | Declare views-reporting as optional dependency in pyproject.toml (C-144) | — | Non-blocking |
| #134 | Investigate: skip_predictions_delivery is PF scaffolding — plan removal | #126 | Post-production cleanup |

Closed/resolved issues:
| Issue | Title | Resolution |
|-------|-------|------------|
| #127 | Fix: 2 test files missing pytest.importorskip guard (C-142) | Fixed in PR #132 |
| #128 | Track: ensemble reconciliation import crashes late (C-143) | Mitigated in PR #133 |

---

## 12. Superseded Documents

The following plan documents are now superseded by this roadmap:

| Document | Date | Status | Why superseded |
|----------|------|--------|----------------|
| `2026-03-03_prediction_frame_open_issues_remediation.md` | 2026-03-03 | **SUPERSEDED** | 6/8 issues FIXED, 1 trivial, 1 REMOVED. Section 5 of this roadmap has current status. |
| `2026-03-15_prediction_frame_two_track_status.md` | 2026-03-15 | **SUPERSEDED** | Item 3 (TEMPORARY undo blocks) resolved. Item 4 (ensemble PF awareness) is this roadmap. |
| `2026-02-27_phase4_transport_parity_plan.md` | 2026-02-27 | **ARCHIVED** | Marked as archived in the document itself. Transport parity implemented. |
| `2026-02-27_phase3_purge_and_cleanup_roadmap.md` | 2026-02-27 | **ARCHIVED** | Phase 3 purge completed. |
| `2026-02-27_prediction_frame_and_explicit_steps.md` | 2026-02-27 | **ARCHIVED** | PredictionFrame + explicit steps implemented. |
| Ephemeral plan (`.claude/plans/encapsulated-knitting-fern.md`) | 2026-05 | **SUPERSEDED** | Session-scoped plan now consolidated here. |

---

*Last updated: 2026-06-01*
