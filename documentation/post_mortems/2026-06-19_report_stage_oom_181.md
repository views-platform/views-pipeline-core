# Investigation & durable-fix design — #181 report-stage host-RAM OOM

**Date:** 2026-06-19  **Issue:** views-pipeline-core #181 (cross-ref views-hydranet C-116 / #124)
**Status:** root cause measured; durable fix designed; no fix landed yet (per "investigate, then log").

## 1. Observed failure
`python main.py -r calibration -t -e -re` (HydraNet `violet_visitor`, 32 GB box) is OOM-killed
(exit 137, ~16–20 GB) in the **post-eval report/publish tail**. Dropping `-re` → 2.4 GB, exit 0
(~7× drop). Peak **scales with `n_posterior_samples`** (S=3 completes, S=8 OOMs), is
wandb-mode-independent, and the model side is ruled out (the inverse-transform on the real
forecast volume is ~0.4 GB). Trajectory: train/eval flat ~2.7 GB (GPU) → PF assembly ~9 GB →
report tail ("Publishing pg_metadata" → "Processing features") **doubles 9 → ~18 GB in ~60 s → OOM**.

## 2. Root cause — object-dtype representations at full grid×time scale (measured)
The report path materializes the prediction and historical data as **pandas object-dtype
DataFrames** — `pred_{target}` cells holding Python lists of S samples, and historical scalar cells
each wrapped in a size-1 `np.array`. Object dtype is ~50–160× heavier per cell than the equivalent
dense `float32` array, and the report does it over the **full grid × full time span**.

Amplifications (file:line):
| # | Stage | Where | Carries S? | Applies to |
|---|-------|-------|-----------|-----------|
| A | PF → **list-in-cell** DataFrame (`[list(row) for row in pf.y_pred]`) | `managers/prediction/prediction_frame_converter.py:73` | **yes** | forecast (~1.17M rows) |
| B | dataset **densification** + per-cell `np.array(float32)` | `data/handlers.py:_preprocess_dataframe` (47-76), `_validate_prediction_structure` (210-247), `_init_dataframe` (103-104) | yes | forecast |
| **H** | **scalar→per-cell `np.array` object-ification** of historical actuals | `data/handlers.py:_init_dataframe:136-152` (non-broadcast path) | no (S=1) | **historical (~10.5M rows)** |
| C | float64 tensor + `np.stack` 2nd copy, then MAP `np.sort` over samples | `data/handlers.py:_prediction_to_tensor:457-486`, views-reporting `statistics/dataset_statistics.py:calculate_map` | yes | forecast |

The two-part signature maps cleanly: **(H) historical object-ification is the S-independent ~9 GB
base** (the full calibration span is ~324 months × 32 400 cells ≈ 10.5M cells — ~9× the forecast
volume), and **(C) the MAP `np.sort` over the S-axis tensor is the doubling 9 → 18 GB** ("Processing
features"). This is why peak scales with S even though the report is "post-collapse" — the collapse
(MAP) is the step that first materializes the full-S tensor. `skip_predictions_delivery=True` does
**not** gate this path (it gates only Track-B parquet, `model.py:1334-1342`); `-re` is an independent
flag (`args.py:266`) so eval does not require the report.

## 3. Measurement (synthetic micro-benchmark, `reports/investigations/report_stage_oom_181.py`)
Quarter-PGM scale measured here (this box has ~6–12 GB free; full scale would OOM it too), per-row
bytes extrapolated to full production volumes. tracemalloc captures Python-heap (object cells); RSS
captures numpy. **Numbers are order-of-magnitude (small-scale RSS is noisy); the ratios are the point.**

| Stage (1 target) | B/row | full forecast GB | full historical GB |
|---|---:|---:|---:|
| dense float32 baseline (per sample) | 4 | — | — |
| A list-in-cell  (S=8) | ~385 | ~0.42 | (n/a) |
| B densify+arrays (S=8) | ~364 | ~0.40 | (n/a) |
| **H historical obj-ify (S=1)** | ~220 | (n/a) | **~2.2** |
| C float64 tensor | ~32–63 | ~0.03 | ~0.3 |

Each object stage is ~**50–160× the dense per-sample cost**. With `n_targets` (the real run has
several) multiplying the forecast side, the historical base + forecast frames + the float64 tensor +
MAP sort copy alive concurrently plausibly sums to the observed ~16–18 GB. **The dense numpy compute
is small (C ≈ 0.03/0.3 GB); the cost is the object-dtype DataFrames** — exactly the "list-in-cell /
object dtype is the non-scaler; dense arrays scale" thesis (`views-frames`, C-40/C-66).

## 4. Durable fix design
**Principle: the report must consume dense arrays, never rebuild object-dtype DataFrames over the
full grid.** It needs point/MAP + a few quantiles for a *sample* of entities — not all S, not every
cell, not pandas.

### Durable (the views-frames / Cluster-A direction)
- The report **receives a typed, collapsed array frame** (a collapsed `PredictionFrame` / the
  anticipated `MetricFrame` from the `views-frames` README §4.2) — point estimate + selected
  quantiles already reduced — instead of loading a full-S list-in-cell parquet and re-deriving.
  Kills A, B, C at the source. This is the keystone-track fix; gated on `views-frames` existing.
- Historical actuals likewise flow as a **dense (time × entity) float32 array** (or are loaded
  column-pruned + chunked), never object-ified per cell. Kills H.

### Standalone quick wins (no views-frames dependency; can land first, low risk)
1. **float32 in `_prediction_to_tensor`** (drop hardcoded `np.float64` at `handlers.py:465`) — halves C. Smallest, safest.
2. **Collapse-before-materialize** — reduce S to point + quantiles *before* building the dense
   tensor / before `to_prediction_df`, so nothing downstream carries S. Removes the S-scaling.
3. **Entity sampling for graphs** — the historical line graph needs a handful of entities, not all
   ~32 400 cells; load/object-ify only the sampled subset. (Converges with views-reporting **C-26**,
   the render-scale sibling.)
4. **Lazy / skipped densification in the report path** — the report needs per-entity series, not a
   dense grid; don't fill 10.5M zero cells just to render.
5. **Decouple the report** — make `-re` opt-out so an eval pass is never gated by report RAM
   (issue open-question 5); a 32 GB box can then always complete `-t -e`.

### Ownership / sequencing
- **pipeline-core:** the converter (A), `handlers.py` densify/object-ify/float64 (B, H, C), a
  collapse-to-array API, the `-re` decoupling. **views-reporting:** `calculate_map`/templates
  consuming the collapsed array, entity sampling (its C-26). The durable frame contract is **views-frames**.
- **Recommended order:** quick wins 1+2 first (unblock 32 GB boxes now) → entity sampling (3) →
  durable array-frame input via views-frames (kills the class). Relationship: **C-40, C-66** (same
  list-in-cell mechanism), **C-182** (immutable frames), **C-165** (no abstractions), views-reporting **C-26**.

## 5. Regression guard
`tests/test_managers/test_report_stage_memory.py` — drives the amplification chain at reduced scale
and asserts the object-dtype overhead ratio stays bounded, so a future regression (or the fix) is
measured by the test, not by OOM-killing a box.

## 6. Register (next step)
Register #181 as **Cluster A's first observed-in-production member** — distinct locus from C-40
(persistence) and views-reporting C-26 (render): the report-stage `_load_historical_data` +
PF-list-in-cell + per-cell object-ification + float64-tensor OOM. Cross-ref C-40/C-66/C-182/C-26,
views-frames, views-hydranet C-116/#124.
