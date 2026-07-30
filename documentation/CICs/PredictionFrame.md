# Class Intent Contract: PredictionFrame

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-06-24
**Related ADRs:** ADR-003 (Authority of Declarations), ADR-009 (Boundary Contracts), ADR-042 (PredictionFrame Adoption), ADR-054 (extraction), views-frames ADR-018 (frozen leaf contract)

> **#188 — the type now lives in the leaf.** `views_pipeline_core.data.prediction_frame.PredictionFrame` **re-exports** `views_frames.PredictionFrame` (the published, API-frozen leaf). The leaf **owns the type contract** (construction, validation, `.values`, save/load); pipeline-core owns only its *use* and its `y_pred.npy` *persistence layout*. This CIC documents pipeline-core's contract with the leaf — the authoritative type spec is the leaf's own docs/tests.

---

## 1. Purpose

The canonical, framework-agnostic transport object for model inference output:
a dense `(N, S)` float32 matrix of posterior-sample predictions plus an explicit
`SpatioTemporalIndex` (`time`, `unit`, `level`). The universal handoff between a
model and the pipeline's evaluation and persistence layers, with no Pandas coupling.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** carry ground truth (`y_true`) — that is the `EvaluationFrame`'s job.
- Does **not** perform alignment/intersection/index-join with actuals.
- Does **not** depend on Pandas/Polars/Xarray — numpy-only (the leaf is numpy-only).
- Does **not** infer identifiers from position or content — they are explicit.
- Does **not** perform lead-time (step) assignment — the orchestrator's job.

---

## 3. Responsibilities and Guarantees (as provided by the leaf)

- **Shape**: `.values` is 2D (`N × S`), **float32** (the leaf coerces on construction).
- **Index**: construction takes a `SpatioTemporalIndex(time, unit, level: SpatialLevel)`;
  identifiers are exposed as `.identifiers == {"time", "unit"}` (integer arrays of length `N`).
- **Integer identifiers**: the leaf requires integer-dtype `time`/`unit` (so NaN in an
  identifier is impossible by construction — stricter than the retired local NaN check).
- **Read-only container**: `.values`, `.n_rows`, `.sample_count`, `.index`, `.metadata`.
- **Sample-axis reduction is NOT a method here** — use `views_frames_summarize`
  (`collapse(frame, np.mean)`, `map_estimate`, `hdi`, `quantiles`).

> **Weakened vs the retired local class:** the leaf does **not** reject `N == 0` / `S == 0`
> at construction (it lets numpy reject them downstream). pipeline-core re-establishes the
> non-empty guarantee at its **load boundary** — see §6.

---

## 4. Inputs and Assumptions

- `y_pred: np.ndarray` — 2D `(N, S)`; coerced to float32. Provided by the model.
- `index: SpatioTemporalIndex` — `time` (`month_id`), `unit` (`priogrid_id` for pgm /
  `country_id` for cm), both integer length-`N`; `level: SpatialLevel` (cm/pgm).
- `metadata: FrameMetadata | None` — optional provenance (model/run_type/timestamp/seed).

---

## 5. Outputs and Side Effects

- Properties: `.values` (the `(N, S)` float32 array — **note: not `.y_pred`**),
  `.identifiers` (`{"time","unit"}`), `.n_rows`, `.sample_count`, `.index`, `.metadata`.
- **Persistence is pipeline-core's, not the leaf's `.save`.** Use
  `views_pipeline_core.managers.prediction.prediction_frame_io`:
  - **`save_pf(frame, dir)`** → writes `{dir}/y_pred.npy` + `{dir}/identifiers.npz`
    (the cross-repo layout views-reporting's loader reads — deliberately NOT the leaf's
    `values.npy`+`header.json`).
  - **`load_pf(dir, level, mmap=False)`** → reads that layout into a leaf `PredictionFrame`;
    `level` is supplied (the layout does not persist it); `mmap` preserves the memmap.

---

## 6. Failure Modes and Loudness

- **Leaf construction** raises on: non-2D values; row count ≠ index length; non-integer
  identifiers; missing/short identifiers (the leaf's `_validation`).
- **`load_pf` (pipeline-core)** raises `ValueError` on an empty/corrupted saved frame
  (`ndim != 2`, `N == 0`, or `S == 0`) — re-establishing the retired local class's
  non-empty guarantee at the load boundary (Fail Loud and Proud), so an empty frame never
  propagates silently into eval/ensemble.

---

## 7. Boundaries and Interactions

- **Created by**: engine repos (e.g. views-hydranet, views-baseline) returning
  `Dict[str, (List[)]PredictionFrame(])` from `_forecast_/_evaluate_model_artifact`, and by
  `PredictionFrameEnsembleManager._aggregate_prediction_frames` (reuses the reference
  frame's index). Construction uses the leaf constructor (`SpatioTemporalIndex`+`level`).
- **Persisted by**: `prediction_frame_io.save_pf` (and `NpzSaver`, which delegates to it);
  reloaded via `load_pf` (incl. the mmap metrics-reload path).
- **Consumed by**: `EvaluationAdapter.from_prediction_frame[s]()` (reads `.values` +
  `.identifiers`), `PredictionFrameConverter` (`.values` → Arrow/list-in-cell), and
  views-reporting's `PredictionFrameLoader` (reads `y_pred.npy`).
- **Not consumed by**: `CorePredictionSniffer` (audits `pd.DataFrame` outputs only).

---

## 8. Examples of Correct Usage

```python
from views_frames import PredictionFrame, SpatioTemporalIndex, SpatialLevel

index = SpatioTemporalIndex(
    time=time_vals.astype(np.int64),   # month_id values from X.index
    unit=unit_vals.astype(np.int64),   # priogrid_id / country_id from X.index
    level=SpatialLevel.PGM,
)
pf = PredictionFrame(np.stack([draw_1, draw_2], axis=1), index)   # (N, S)

from views_pipeline_core.managers.prediction.prediction_frame_io import save_pf, load_pf
save_pf(pf, out_dir)
pf2 = load_pf(out_dir, level="pgm", mmap=True)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: the old kwargs constructor — removed in #188 (TypeError).
pf = PredictionFrame(y_pred=preds, identifiers={"time": t, "unit": u})

# WRONG: reading .y_pred — the leaf exposes .values.
arr = pf.y_pred

# WRONG: inferring time from position (time must be real month_id values).
PredictionFrame(preds, SpatioTemporalIndex(np.arange(len(preds)), u, SpatialLevel.PGM))
```

---

## 10. Test Alignment

- `tests/test_data/test_prediction_frame.py` — re-export identity + collapse-via-summarize.
- `tests/test_data/test_prediction_frame_persistence.py` — `save_pf`/`load_pf` round-trip + mmap.
- `tests/test_modules/test_evaluation_adapter_golden.py` — golden-output net for the adapter
  hot path (pins eval numerics across the migration; register C-189).

---

## 11. Evolution Notes

- **#188 complete:** the local class is retired; pipeline-core re-exports the leaf
  (`views_frames.PredictionFrame`). The Strangler-Fig migration (ADR-042) for the *type*
  is done; the leaf is the single canonical frame.
- **On-disk format:** pipeline-core keeps `y_pred.npy` + `identifiers.npz` (`prediction_frame_io`)
  because it is a cross-repo contract (views-reporting reads `y_pred.npy`). Moving to the
  leaf's `values.npy`+`header.json` is a separate, coordinated change (needs a views-reporting
  update) — out of scope for #188.
- **Producer coupling (C-193):** engine repos construct PredictionFrames; the leaf-constructor
  change is breaking → pipeline-core 3.0.0 + lockstep engine migration (views-hydranet#137,
  views-baseline#21). A leaf-level `build_prediction_frame` factory is proposed (views-frames#113, D-38).
- **`CorePredictionSniffer` extension** (level-range validation of `unit`) remains future work.

## 12. Known Deviations

- **Empty frames not rejected at construction:** the leaf accepts `N == 0` / `S == 0`;
  pipeline-core re-guards only at `load_pf` (§6), not at every construction site.
- **`level` not persisted:** the `y_pred.npy` layout stores only `time`/`unit`; `level` is
  supplied to `load_pf` by the caller (from `config["level"]`).
- **No `.y_pred` / `.collapse` / `.identifier_keys`:** these existed on the retired local
  class; callers use `.values`, `views_frames_summarize.collapse`, and `set(.identifiers)`.

---

## End of Contract

This document defines the **intended meaning** of pipeline-core's use of the leaf
`PredictionFrame`. Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
