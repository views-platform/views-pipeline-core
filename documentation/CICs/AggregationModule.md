# Class Intent Contract: AggregationModule

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-09-19 (written from the code after #516/#517; replaces the retired `AggregationManager.md`, #506)
**Related ADRs:** ADR-064 ("The pooling contract"; the named refusals), ADR-044 (register C-324, C-325, C-323)
**Module:** `views_pipeline_core/modules/aggregation/aggregator.py`

---

## 1. Purpose

Pools the predictions of an ensemble's constituent models into one prediction table. Two
kinds of input: **point** predictions (one value per `(time, entity)` row per target) and
**distributions** (a list-in-cell of `S` samples per row per target). Constituents are added
one at a time with an optional weight; `aggregate()` returns one Polars DataFrame with the
ensemble's index and one `pred_<target>` column per target. Constructed by
`EnsembleManager` (`managers/ensemble/ensemble.py`) and `DataFrameEnsembleManager`
(`managers/ensemble/dataframe_ensemble.py`); the PredictionFrame ensemble path does not use it.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** reconcile across levels (that is `reconcile_frames` / the `Reconciler` port).
- Does **not** check entity coverage against the raw input. That is `CorePredictionSniffer`
  on each *constituent* before it reaches the pool (ADR-064 item 1); the pool's own check is
  row-set *equality* between constituents (item 4). An ensemble's pooled output is never
  coverage-checked (register C-323, annotated 2026-09-19).
- Does **not** evaluate, save, or publish. The ensemble manager does.
- Does **not** choose weights; it normalises the ones it is given.

---

## 3. Responsibilities and Guarantees

- `add_model(data, weight=None, name=None)` accepts a Polars/pandas DataFrame or a parquet/csv
  path; the first model sets the pool's `prediction_type` (`"point"` / `"distribution"`),
  `sample_size`, and the canonical index (`_index_signature`, remembered with the model that
  set it). Every later model must match all three, or `add_model` raises.
- **Every constituent forecasts the same `(time, entity)` rows.** `_check_index_consistency`
  refuses a model whose row set differs and **names the rows**: the entities missing and
  extra (sorted, up to `INDEX_MISMATCH_MAX_LISTED = 25`, then "and N more"), their month
  range, and both model names — the canonical model and the offender. A count alone was
  what #509 got (C-324).
- `aggregate(method=None, use_weights=True)`: distributions default to `"concat"`
  (`"vincentization"` is the alternative); points default to `"mean"` (`"median"`, `"min"`,
  `"max"`). An unknown method, or `aggregate()` before any `add_model`, raises `ValueError`;
  for point predictions `use_weights=True` is supported only with `"mean"`.
- **The pooling contract (`concat`; ADR-064).** One `(model, sample)` pick per output
  column, drawn once above the target loop — `chosen_models` by the normalised weights,
  `chosen_samples` uniformly — and reused for **every row and every target**: column *k* of
  the pool is column `chosen_samples[k]` of model `chosen_models[k]` throughout. A
  constituent's sample column is one scenario across all entities and targets, and pooling
  keeps it one. The draw is seeded (`default_rng(42)`): the same pool is reproducible; pooled
  values differ from pre-3.3.0 runs because the draw pattern does (C-325).
- Weights: unspecified → equal; partially specified → the remainder is shared equally by
  the unspecified models; specified weights summing above 1.0, or to exactly 1.0 with some
  unspecified, raise.
- Point aggregation with `use_weights=True` is a weighted mean (`"mean"` only).

---

## 4. Inputs and Assumptions

- `index_cols` default `["month_id", "country_id"]`; the ensemble managers pass their level's
  index columns explicitly.
- A distribution column is `List(Float32)` with the same `S` in every row of every model.
- On the in-memory path (`_load_to_polars`, both production callers), each constituent is
  wrapped in `CMDataset`/`PGMDataset` **before** the index check, which densifies to the
  entities present at the last time step; the index check therefore sees the densified rows,
  not the engine's raw output (ADR-064 consequences; C-324). The direct-parquet path does not
  densify.
- Model order is `add_model` order and fixed for the life of the instance; index *k* in the
  drawn `chosen_models` means the same constituent for every target.

---

## 5. Outputs and Side Effects

- `aggregate()` returns a Polars DataFrame: `index_cols` + one `pred_<target>` per target
  (list-in-cell of `S` for distributions). Also stored on `self.aggregated_df`. No disk I/O.

---

## 6. Failure Modes and Loudness

- Prediction-type or sample-size mismatch between constituents → `ValueError` naming both.
- Row-set mismatch → `ValueError` naming the rows (above). Never filtered (ADR-064 item 5:
  refuse, do not filter — dropping rows would manufacture an absence).
- A column whose samples are not `(n_rows, S)` → `ValueError` naming the column.
- Bad weights → `ValueError` with the sum.
- Every refusal is loud; nothing is coerced.

---

## 7. Boundaries and Interactions

- **Depends on:** `polars`, `numpy`, `pandas` (input acceptance), `data.handlers`
  (`CMDataset`, `PGMDataset` for densification on the in-memory path).
- **Used by:** `EnsembleManager._get_aggregated_df`, `DataFrameEnsembleManager` (same),
  then `reconcile_frames` / savers / `EvaluationStage` downstream.
- **Contract with engines:** every constituent must forecast the entities present at the
  last observed month (ADR-064) — checked per constituent by `CorePredictionSniffer`, and
  agreed between constituents here.

---

## 8. Examples of Correct Usage

```python
agg = AggregationModule(index_cols=["month_id", "country_id"], target_cols=["lr_sb"])
agg.add_model("models/a/data/generated/a_predictions.parquet", weight=0.6, name="a")
agg.add_model("models/b/data/generated/b_predictions.parquet", name="b")   # gets 0.4
pooled = agg.aggregate()            # "concat": one (model, sample) per column, joint across rows/targets
```

---

## 9. Examples of Incorrect Usage

```python
agg.aggregate(method="concat")      # before add_model → ValueError
agg.add_model(df_with_dead_states)  # extra rows vs the canonical model → ValueError naming them
```

---

## 10. Test Alignment

- `tests/test_modules/test_ensemble_aggregator.py`: the joint-draw contract
  (`test_concat_picks_one_model_and_sample_per_column_for_every_row_and_target` encodes every
  cell as `(model, row, sample)` and asserts one pick per column across rows and targets —
  fails against per-row draws, per-target re-draws, and the March block-wise design);
  weights honoured in the draw; index-mismatch refusals name entities, months and models,
  capped at 25; same-count-different-entities and month-only mismatches refused; by-name
  column resolution in `_describe_rows`.

---

## 11. Evolution Notes

- `INDEX_MISMATCH_MAX_LISTED` is one of three identical caps (with the sniffer's
  `ENTITY_COVERAGE_MAX_LISTED` and the PF ensemble's `IDENTIFIER_MISMATCH_MAX_LISTED`);
  extracted on a fourth site, not before (WET before DRY).
- The fixed seed predates this contract; making it configurable is a behaviour change with
  a release note, not a refactor.

---

## End of Contract
