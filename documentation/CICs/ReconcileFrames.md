# Class Intent Contract: reconcile_frames

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-06-26
**Related ADRs:** ADR-042 (PredictionFrame), ADR-051 (Composition-Based Ensemble Architecture)
**Related epics:** #233 (frames-native ensemble reconciliation, #234)

> Module: `views_pipeline_core/modules/reconciliation/reconcile_frames.py`. Two free
> functions (`align_country_to_grid`, `reconcile_frames`) — not a class — but governed by
> this contract.

---

## 1. Purpose

Reconcile a grid (pgm) `PredictionFrame` to country (cm) totals via an injected `Reconciler`
port, frames-in/frames-out. It is the frames-native orchestration the
`PredictionFrameEnsembleManager` calls, replacing the DataFrame path's dataset↔frame adapter.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** implement the reconciliation math — that is the injected `Reconciler`
  (`views_frames_reconcile.ReconciliationModule`).
- Does **not** build geography (`(time, priogrid) -> country`) — baked into the injected reconciler.
- Does **not** touch pandas, datasets, or the DataFrame path's `reconcile_datasets` adapter
  (`_stack_cells`/`_align_to_dataframe`). The frames path holds frames; no bridge is needed.
- Does **not** load the country forecast (that is `cm_forecast_loader.load_cm_frame`).
- Does **not** persist anything; it returns a new frame.

---

## 3. Responsibilities and Guarantees

- Guarantees the country frame is aligned to the grid's draw count before reconciling:
  `point-broadcast` (cm `sample_count == 1` → tiled across S) or `aligned-draws`
  (cm `sample_count == S` → passthrough); any other count **fails loud**.
- Guarantees bounded memory via chunk-by-time (default): reconciliation runs one time step
  at a time, never holding a global × S frame whole (C-200a).
- Guarantees the result is realigned to the input grid's `(time, unit)` index — the port's
  row order is never trusted.
- Guarantees the input `pgm_frame` is **not mutated** (a new frame is returned), and its
  `FrameMetadata` is carried through.
- Guarantees the reconciliation **mode** is logged (`aligned-draws` flagged as the per-draw
  approximation, C-200b).

---

## 4. Inputs and Assumptions

- `reconciler`: an object satisfying the `Reconciler` port (`reconcile(cm, pgm) -> frame`).
- `cm_frame`: a `SpatialLevel.CM` frame whose `sample_count` is 1 or equals the grid's.
- `pgm_frame`: a `SpatialLevel.PGM` frame with **unique** `(time, unit)` rows (asserted;
  duplicates break the index realignment — C-21) and at least one row.

---

## 5. Outputs and Side Effects

- Returns a new pgm `PredictionFrame`, same index/shape as the input grid, reconciled to
  country totals, with the input's metadata. Output is uncollapsed `(N, S)`.
- Side effect: one `logger.info` recording the mode.

---

## 6. Failure Modes and Loudness

- `cm_frame.sample_count` neither 1 nor the grid's draw count → `ValueError`.
- `pgm_frame` empty or with duplicate `(time, unit)` rows → `ValueError`.
- A reconciler that drops/duplicates rows → the final `reindex` fails loud (not a superset).
- No silent fallback: a misconfigured or misbehaving reconciler raises.

---

## 7. Boundaries and Interactions

- Imports **only** `views_frames` + the `Reconciler` port. Must not import the dataset
  god-class, pandas, `views_reporting`, `views_postprocessing`, or `reconcile_datasets`
  (ADP/CRP — the frames path must not inherit the pandas-era write-side orchestration).
- Trusts the injected `Reconciler` as opaque (geography + math live behind it).

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.modules.reconciliation.reconcile_frames import reconcile_frames
# reconciler: an injected views_frames_reconcile.ReconciliationModule(map_keys, map_vals)
reconciled = reconcile_frames(reconciler, cm_frame, pgm_frame)  # chunk-by-time by default
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: a country frame with an incompatible sample count (not 1, not S)
reconcile_frames(reconciler, cm_with_2_draws, pgm_with_1024_draws)  # -> ValueError

# WRONG: reusing the DataFrame adapter for the frames path
from views_pipeline_core.modules.reconciliation.adapter import reconcile_datasets  # not here
# -> couples the frames path to pandas list-in-cell datasets (C-200c).
```

---

## 10. Test Alignment

- `tests/test_modules/test_reconcile_frames.py`: align modes + fail-loud; reorder-safety;
  empty + duplicate-row fail-loud; no-mutation; metadata pass-through; mode log; and
  real-substrate parity (`views_frames_reconcile`) — point & versions sum to country totals
  per draw, draws preserved, zeros preserved, non-negative, **chunk-by-time == whole-frame
  bit-exact**.
- End-to-end through the manager: `tests/test_managers/test_prediction_frame_ensemble_manager.py`
  (`TestPredictionFrameReconciliationEndToEnd`).

---

## 11. Evolution Notes

- The `point-broadcast` tiling lives here WET; its DRY home is a native broadcast in
  `views_frames_reconcile` (views-platform/views-frames#143), after which this helper can
  collapse to a direct call.
- `aligned-draws` is a per-draw approximation (C-200b); the principled joint reconciliation
  is design-gated in views-platform/views-frames#145.

---

## End of Contract

This document defines the **intended meaning** of `reconcile_frames`.
Changes to behavior that violate this intent are bugs. Changes to intent must update this contract.
