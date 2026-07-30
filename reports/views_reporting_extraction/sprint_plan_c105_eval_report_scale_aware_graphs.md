# Sprint Plan: C-105 — Scale-Aware Eval Report Sample Graphs

> **SUPERSEDED (2026-05-27):** This sprint plan was written before the architectural
> misplacement investigation concluded. The investigation (see
> `architectural_misplacement_investigation.md` Section 11) determined that extraction
> PRs 0-6 must complete before Sprint 4. After extraction, the files referenced below
> (`views_pipeline_core/templates/reports/evaluation.py`,
> `views_pipeline_core/modules/visualizations/historical.py`) will live in the
> `views-reporting` package. **This plan must be rewritten against `views-reporting`
> file paths before Sprint 4 begins.** The design approach (entity sampling, numpy
> fallback, dual-path dispatch) remains valid — only the file locations change.

**Risk register entry:** C-105 (Tier 2)
**Target branch:** `fix/eval-report-scale-aware-graphs`
**Base branch:** `development`
**Estimated effort:** 4–6 hours
**Priority score:** 4.0 (Memory & PGM scalability cluster)

---

## 1. Problem Statement

`EvaluationReportTemplate._add_prediction_sample_graphs()` (`evaluation.py:302-428`)
generates per-target prediction-vs-historical line graphs for the eval HTML report. The
method was designed for CM-scale data (~50 entities, ~1800 rows) and breaks at PGM
scale (172k grid cells, 6.2M rows) in two independent ways:

### Scaling Failure 1: PGMDataset Materialization

The method wraps prediction and historical DataFrames in `PGMDataset` (or `CMDataset`),
which calls `_init_dataframe()` (`handlers.py:86-163`). This eagerly materializes ALL
rows into numpy arrays via multiple `.apply()` passes. At PGM scale with 64 posterior
samples: ~4 GB memory, several minutes of wall time. The class was designed for CM
analysis (~1800 rows) and has no scale guard, no sampling parameter, and no lazy
initialization (C-106).

### Scaling Failure 2: HistoricalLineGraph Plotly Traces

`HistoricalLineGraph._plot_interactive()` (`historical.py:141-273`) creates one
Plotly trace per entity. At PGM scale: 172k traces, each with historical + forecast
data. The resulting HTML is multi-GB. The dropdown menu for entity selection has 172k
entries. The browser cannot render it.

### Current State After PR #87

With `skip_predictions_delivery: True` (the new default for PF models), Track B
eval-path parquets are no longer produced. The `_add_prediction_sample_graphs()` method
reads Track B parquets via `_get_generated_predictions_data_file_paths()` — when no
parquets exist, the method returns an empty list and the entire graph section is
silently skipped (the `try/except` at `evaluation.py:293-300` catches the resulting
IndexError or empty-list case).

This means PF model eval reports currently have **no sample graphs at all**. For CM
models (which use the DF path and always write parquets), graphs continue to work as
before.

### Data Sources Available

PF models write two alternative data formats that could back the graphs:

| Track | Format | Location | Content |
|-------|--------|----------|---------|
| A (staging) | `.npy` + `.npz` | `_pf_staging/origin_{i}/{target}/` | Ephemeral — deleted after eval |
| A+ (permanent) | `.npy` + `.npz` | `predictions_{run_type}_{ts}/origin_{i}/{target}/` | Persists after eval |

Track A+ is the natural replacement. It contains `y_pred.npy` (the prediction tensor)
and `identifiers.npz` (with `time` and `unit` arrays). `PredictionFrame.load()` can
reconstruct the full PredictionFrame from these files.

---

## 2. Design

### Core Approach: Entity-Sampled Numpy Fallback

When Track B parquets are absent (PF models with `skip_predictions_delivery: True`),
fall back to Track A+ numpy data with entity sampling:

1. **Discover** Track A+ directories via
   `model_path._get_generated_pf_prediction_paths(run_type)`
2. **Load** PredictionFrame from the selected sequence directories via
   `PredictionFrame.load(path)`
3. **Sample** a fixed number of entities (e.g., 10) instead of loading all 172k
4. **Reconstruct** a lightweight DataFrame with only sampled entities for
   `HistoricalLineGraph`

### Entity Sampling Strategy

For CM (~50 entities): sample all — no change from current behavior.

For PGM (172k entities): sample N entities (default N=10, configurable). Sampling
should be deterministic (seeded by run_type + target for reproducibility) and
stratified if possible (pick entities from different value ranges to show variation,
not just the first 10).

Proposed approach:
```python
import numpy as np

def _sample_entities(unit_ids: np.ndarray, n: int = 10, seed: int = 42) -> np.ndarray:
    if len(np.unique(unit_ids)) <= n:
        return np.unique(unit_ids)
    rng = np.random.default_rng(seed)
    return rng.choice(np.unique(unit_ids), size=n, replace=False)
```

### HistoricalLineGraph Changes

`_plot_interactive()` already supports a subset of entities via the `entity_ids`
parameter (`historical.py:44-139`). No changes needed to the graph class itself — the
sampling happens before data is passed to the graph.

### Dual-Path Dispatch

The method needs to handle three cases:

1. **Track B parquets exist** (DF models, or PF models with
   `skip_predictions_delivery: False`): Use existing parquet-based path. No change.
2. **Track A+ numpy exists** (PF models with `skip_predictions_delivery: True`): Use
   numpy fallback with entity sampling.
3. **Neither exists**: Skip graphs silently (existing behavior).

---

## 3. Implementation Steps

### Step 1: Add Entity Sampling Helper

**File:** `views_pipeline_core/templates/reports/evaluation.py`

Add a module-level helper function `_sample_entity_ids(unit_ids, n, seed)` that
returns a deterministic sample of entity IDs. This is a pure function — no state, no
side effects.

### Step 2: Add Numpy Prediction Loading Helper

**File:** `views_pipeline_core/templates/reports/evaluation.py`

Add a helper method `_load_pf_predictions_as_dataframe(pf_paths, target, entity_ids)`
that:

1. Takes a list of Track A+ directory paths (from
   `_get_generated_pf_prediction_paths`)
2. Picks first/middle/last sequences (matching existing Track B logic)
3. For each selected sequence path, loads `PredictionFrame.load(path / target)` or
   iterates origin subdirectories to find the target
4. Extracts `y_pred` values for the sampled entity IDs only
5. Returns a list of lightweight DataFrames with columns
   `[month_id, {entity_id_col}, pred_{target}]` — compatible with `CMDataset` /
   `PGMDataset` construction

This avoids materializing the full 6.2M-row DataFrame. Only sampled entities are
extracted from the numpy arrays.

### Step 3: Modify `_add_prediction_sample_graphs()`

**File:** `views_pipeline_core/templates/reports/evaluation.py`

Restructure the method body:

```python
def _add_prediction_sample_graphs(self, report_manager, target_identifier):
    # --- Attempt Track B (parquet) path ---
    pred_file_paths = self.model_path._get_generated_predictions_data_file_paths(
        self.run_type
    )
    if pred_file_paths:
        # Existing parquet-based logic (unchanged)
        ...
        return

    # --- Attempt Track A+ (numpy) path ---
    pf_paths = self.model_path._get_generated_pf_prediction_paths(self.run_type)
    if not pf_paths:
        logger.info("No prediction data found for sample graphs (Track B or A+).")
        return

    # Load historical data (reuse existing logic)
    ...

    # Determine entity sample
    level = self.configs.get("level", "cm")
    entity_col = "priogrid_gid" if level == "pgm" else "country_id"
    entity_ids = _sample_entity_ids(
        historical_df.index.get_level_values(entity_col).unique().to_numpy(),
        n=10 if level == "pgm" else 50,
        seed=hash(target_identifier) & 0xFFFFFFFF,
    )

    # Filter historical data to sampled entities
    historical_sampled = historical_df.loc[
        historical_df.index.get_level_values(entity_col).isin(entity_ids)
    ]
    hist_dataset = dataset_cls(historical_sampled, targets=[target_identifier])

    # Load predictions from Track A+ for sampled entities
    pred_dataframes = _load_pf_predictions_as_dataframe(
        pf_paths, target_identifier, entity_ids, level
    )

    # Render graphs (same as existing Track B path)
    for seq_idx, pred_df in pred_dataframes:
        pred_dataset = dataset_cls(pred_df)
        graph = HistoricalLineGraph(
            historical_dataset=hist_dataset,
            forecast_dataset=pred_dataset,
        )
        html = graph.plot_predictions_vs_historical(
            targets=[target_identifier],
            entity_ids=list(entity_ids),
            as_html=True,
            alpha=0.9,
        )
        report_manager.add_html_block(
            f"Prediction Sample (sequence {seq_idx})", html, height=700
        )
```

### Step 4: Handle PredictionFrame → DataFrame Conversion for Sampled Entities

The key technical challenge: `PredictionFrame` stores `y_pred` as a dense numpy array
with shape `(n_rows, n_steps)` or `(n_rows, n_steps, n_samples)`. The identifiers
(`time`, `unit`) map rows to month_id and entity_id. To produce a DataFrame compatible
with `CMDataset`/`PGMDataset`:

```python
def _pf_to_sampled_dataframe(pf, target, entity_ids, level):
    """Convert a PredictionFrame to a sampled DataFrame for visualization."""
    time_ids = pf.identifiers["time"]
    unit_ids = pf.identifiers["unit"]

    # Build mask for sampled entities
    mask = np.isin(unit_ids, entity_ids)

    # Extract point predictions (mean across samples if stochastic)
    y = pf.y_pred[mask]
    if y.ndim == 3:
        y = y.mean(axis=-1)  # (n_sampled, n_steps) → point estimate
    # For visualization, use the first step (step 0) or flatten
    # This depends on how HistoricalLineGraph expects the data

    entity_col = "priogrid_gid" if level == "pgm" else "country_id"
    index = pd.MultiIndex.from_arrays(
        [time_ids[mask], unit_ids[mask]],
        names=["month_id", entity_col],
    )
    return pd.DataFrame({f"pred_{target}": y[:, 0]}, index=index)
```

The exact shape handling depends on how the existing Track B parquets encode
predictions (list-in-cell vs expanded). The Track B path uses `PGMDataset` which
auto-detects `pred_*` columns and computes `sample_size`. The numpy path should
produce the same column structure.

**Important:** This conversion is applied only to the **sampled** subset (10 entities),
not the full 172k. Memory usage is negligible.

### Step 5: Write Tests

**File:** `tests/test_templates/test_evaluation_report.py` (new or existing)

1. `test_sample_entity_ids_cm_returns_all` — CM-scale (<= N) returns all unique IDs
2. `test_sample_entity_ids_pgm_returns_n` — PGM-scale returns exactly N sampled IDs
3. `test_sample_entity_ids_deterministic` — same seed produces same sample
4. `test_add_prediction_sample_graphs_track_b_path` — parquets present → uses Track B
5. `test_add_prediction_sample_graphs_numpy_fallback` — no parquets, Track A+ present
   → uses numpy path
6. `test_add_prediction_sample_graphs_neither` — no data → skips silently

### Step 6: Update CICs and Risk Register

- **CIC:** `EvaluationReportTemplate` — if a CIC does not exist, this is not the
  sprint to create one. Just ensure no existing CIC is violated.
- **Risk register:** Mark C-105 as Resolved. Note that C-106 (PGMDataset scale guard)
  is bypassed — the numpy fallback does not use `PGMDataset` for prediction data; it
  constructs a sampled DataFrame directly.

---

## 4. Files Modified

| File | Change |
|------|--------|
| `views_pipeline_core/templates/reports/evaluation.py` | Dual-path dispatch, numpy fallback, entity sampling |
| `tests/test_templates/test_evaluation_report.py` | 6 new tests |
| `reports/technical_risk_register.md` | Resolve C-105 |

### Files NOT Modified

| File | Why not |
|------|---------|
| `views_pipeline_core/modules/visualizations/historical.py` | Already supports `entity_ids` parameter — no changes needed |
| `views_pipeline_core/data/handlers.py` | PGMDataset not used for prediction data in the numpy path |
| `views_pipeline_core/data/prediction_frame.py` | `PredictionFrame.load()` used as-is |
| `views_pipeline_core/data/model_path.py` | `_get_generated_pf_prediction_paths()` used as-is |

---

## 5. Acceptance Criteria

- [ ] CM models: eval report sample graphs work identically to before (Track B path)
- [ ] PF models with `skip_predictions_delivery: True`: eval report shows sampled
      entity graphs from Track A+ numpy data
- [ ] PF models with `skip_predictions_delivery: False`: eval report uses Track B
      parquets (no change)
- [ ] PGM-scale entity count is sampled to N=10 (not 172k traces)
- [ ] Entity sampling is deterministic (same seed → same entities)
- [ ] No `PGMDataset` construction for prediction data in the numpy path
- [ ] Memory usage for graph generation is bounded (no 4 GB materialization)
- [ ] `ruff check .` clean
- [ ] Full test suite passes
- [ ] C-105 marked Resolved in risk register

---

## 6. Risk Assessment

**Blast radius:** Medium. Changes to `evaluation.py` affect all model eval reports.
The Track B path is untouched (guarded by the `if pred_file_paths:` check), so DF
models are safe. The new numpy path only fires when Track B parquets are absent.

**Data fidelity:** The numpy fallback shows point estimates (mean across samples for
stochastic models). The Track B path shows list-in-cell data which `PGMDataset`
expands into per-sample arrays. The graphs may look slightly different — this is
acceptable because the purpose is visualization, not exact numeric comparison.

**Missing Track A+ data:** If a PF model was evaluated before the C-94 fix (timestamp
agreement), Track A+ directories may not exist at the expected path. The fallback
handles this by returning early with a log message.

**`_get_generated_pf_prediction_paths()` correctness:** The C-96 fix
(2026-05-22) corrected this method to filter by directory naming convention only. It
now correctly discovers Track A+ directories. Verify this with a quick sanity check
against a real model's `data_generated/` directory if possible.

---

## 7. Open Design Questions

### Q1: How Many Steps to Show?

PredictionFrame `y_pred` has shape `(n_rows, n_steps)` or `(n_rows, n_steps, n_samples)`.
The existing Track B parquets contain list-in-cell predictions for all steps. The graph
currently shows predictions for all entities across all time steps.

For the numpy fallback, should we show:
- All 36 steps? (Each step is a separate time point in the forecast horizon)
- Only the first step? (Simpler, less visual noise)
- The mean across steps? (Collapses temporal dimension)

**Recommendation:** Show all 36 steps. The `HistoricalLineGraph` already handles
time-series data — each entity gets one line across all time points. This matches
the Track B behavior.

### Q2: Stochastic Models — Point Estimate or HDI?

`HistoricalLineGraph._plot_interactive()` already supports HDI (Highest Density
Interval) visualization when `sample_size > 1`. For the numpy fallback:
- Option A: Collapse to point estimate (`y.mean(axis=-1)`) — simpler, faster
- Option B: Pass full sample data to `PGMDataset` for HDI visualization — richer but
  requires more memory

**Recommendation:** Start with Option A (point estimate). Add HDI support as a
follow-up if users request it. The primary goal is "some graph" vs "no graph."

### Q3: Entity Sampling — Random or Value-Stratified?

- Random (seeded): Simple, fast, reproducible
- Stratified: Pick entities from quantiles of the prediction distribution — ensures
  the sample shows variation, not just background noise

**Recommendation:** Start with random seeded sampling. Stratified requires loading
the full prediction array to compute quantiles, which partially defeats the purpose
of sampling.

---

## 8. Relationship to C-106

C-106 (`PGMDataset` no scale guard) is related but NOT addressed by this sprint. This
sprint **bypasses** `PGMDataset` for prediction data entirely — the numpy fallback
constructs sampled DataFrames directly. `PGMDataset` is still used for the historical
data, but historical data at CM scale (~1800 rows) is fine.

If historical data is at PGM scale (172k rows), the `PGMDataset` construction for
historical data could still be slow. Consider: (a) sampling historical data to match
the sampled entity IDs before wrapping in `PGMDataset`, or (b) constructing the
historical dataset only for the sampled entities. This is a bounded problem — 10
entities × ~500 months = 5000 rows, well within `PGMDataset` comfort zone.

C-106 remains open for its own sprint (PGMDataset scaling assessment / retire / refactor).
