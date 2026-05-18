# Class Intent Contract: AggregationManager

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology), ADR-003 (Authority of Declarations)

---

## 1. Purpose

`AggregationManager` aggregates predictions from multiple ensemble constituent models into a single set of ensemble predictions. It supports both point predictions (scalar per cell) and distributional predictions (list of samples per cell), with weighted or unweighted aggregation.

The workflow is: `add_model()` one or more models, then call `aggregate()` to produce the combined output. All internal computation uses Polars for performance; inputs may be Polars DataFrames, pandas DataFrames, or file paths.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** train models, run inference, or evaluate predictions.
- Does **not** persist results to disk. The caller is responsible for saving the output.
- Does **not** perform hierarchical reconciliation (that is `ReconciliationModule`'s responsibility).
- Does **not** decide which models to include or what weights to assign. Those decisions are made by the orchestrator.
- Does **not** handle missing data imputation. Models must have aligned index rows; mismatches cause `ValueError`.

---

## 3. Responsibilities and Guarantees

- **`add_model(data, weight, name)`**:
  - Accepts `pl.DataFrame`, `pd.DataFrame`, `str`, or `Path` as input.
  - For `.parquet` files, uses `_load_parquet_direct` (zero-copy Arrow read via Polars). For all other inputs, uses `_load_to_polars` which wraps in `CMDataset`/`PGMDataset` based on the entity index name.
  - Validates weight < 1.0 if provided.
  - Validates index consistency against previously added models (exact row-set match via `_check_index_consistency`).
  - Detects prediction type (`"point"` if list length == 1, `"distribution"` otherwise) and sample size via `_detect_prediction_shape`.
  - Validates consistency of prediction type and sample size across all added models via `_check_model_consistency`.
  - Renames target columns to `{target}_{model_name}` to avoid collisions.

- **`aggregate(method, use_weights)`**:
  - Dispatches to `_aggregate_distributions` or `_aggregate_point_predictions` based on `self.prediction_type`.
  - **Distribution methods**: `"concat"` (linear pooling with resampling, default), `"vincentization"` (quantile-weighted average).
  - **Point methods**: `"mean"` (default), `"median"`, `"min"`, `"max"`.
  - Weights are only supported with `method="mean"` for point predictions. Raises `ValueError` if weights are used with other point methods.
  - Returns a `pl.DataFrame` with index columns and aggregated target columns.

- **Weight normalisation** (`_normalize_weights_new`):
  - All `None` weights: equal distribution (`1/n_models`).
  - Mixed specified/unspecified: remaining weight distributed equally among unspecified models.
  - Total specified weight > 1.0: raises `ValueError`.
  - All weights normalised to sum to 1.0.

---

## 4. Inputs and Assumptions

- **Constructor parameters**:
  - `index_cols: List[str]`: Column names forming the row identity (default `["month_id", "country_id"]`).
  - `target_cols: Optional[List[str]]`: Column names for prediction targets (e.g., `["pred_ln_sb_best"]`).

- **`add_model` data requirements**:
  - Must contain all `index_cols` as integer-typed columns.
  - Must contain all `target_cols` as `List`-typed columns (each cell is a list of floats).
  - All models must have the same index rows (same set of `(month_id, entity_id)` tuples).
  - All models must have the same prediction type and sample size.

- **Internal state**:
  - `self.models: List[_ModelSpec]`: Accumulated model data.
  - `self.prediction_type: Optional[str]`: `"point"` or `"distribution"`, set from first model.
  - `self.sample_size: Optional[int]`: Sample count per cell, set from first model.
  - `self._index_signature: Optional[pl.DataFrame]`: Canonical index from first model.

- **`_ModelSpec`** (dataclass): `name: str`, `df: pl.DataFrame`, `weight: Optional[float]`.

---

## 5. Outputs and Side Effects

- **`aggregate()`**: Returns `pl.DataFrame` with `index_cols` + aggregated `target_cols`.
  - For distribution aggregation: each target cell contains a list of `sample_size` floats (concat) or `sample_size` floats (vincentization).
  - For point aggregation: each target cell is a scalar float.
- **`self.aggregated_df`**: Set as a side effect of `_aggregate_distributions`. Not set by `_aggregate_point_predictions`.
- **Helper `_arrow_series_to_numpy`** (module-level function): Extracts `(n, s)` float32 numpy array from a Polars `List(Float32)` series via zero-copy Arrow buffer extraction.
- **Determinism**: `_concatenate_aggregation` uses `np.random.default_rng(42)` for reproducible resampling.
- **No disk I/O**: The class does not write anything to disk.

---

## 6. Failure Modes and Loudness

| Condition | Exception | Message pattern |
|---|---|---|
| Weight >= 1.0 | `ValueError` | "Weight must be less than 1.0, got {weight}" |
| Index mismatch between models | `ValueError` | "Index mismatch for model '{name}'" |
| Prediction type mismatch | `ValueError` | "Model '{name}' has prediction type '{type}', but existing models use '{type}'" |
| Sample size mismatch (distributions) | `ValueError` | "Model '{name}' has sample size {n}, but existing models use {m}" |
| Mixed point/distribution columns | `ValueError` | "Target columns contain a mixture of point and probabilistic predictions" |
| Aggregate before adding models | `ValueError` | "Cannot aggregate: prediction_type is not set" |
| Invalid method for prediction type | `ValueError` | "Invalid method='{method}' for {type} predictions" |
| Weights with non-mean point method | `ValueError` | "Weights can only be used with aggregation_func='mean'" |
| Specified weights sum > 1.0 | `ValueError` | "Specified weights sum to {total}, which exceeds 1.0" |
| Missing index columns in parquet | `ValueError` | "Parquet '{name}' missing index columns: {list}" |
| Non-integer index column | `TypeError` | "Index column '{col}' must be integer, got {dtype}" |
| Non-List target column | `TypeError` | "Target column '{col}' must be List, got {dtype}" |

All failures are loud. No silent fallbacks.

---

## 7. Boundaries and Interactions

- **Upstream**: Receives prediction DataFrames or parquet file paths from the ensemble orchestrator.
- **`_ViewsDataset` / `CMDataset` / `PGMDataset`**: Used internally by `_load_to_polars` to normalise legacy pandas inputs. The `_load_parquet_direct` path bypasses this entirely for Arrow-native files.
- **`PredictionFrameConverter`**: The Arrow parquet files written via `to_arrow_table` are consumed by `_load_parquet_direct`.
- **Downstream**: The output `pl.DataFrame` is consumed by the ensemble pipeline for persistence, reconciliation, or further processing.

---

## 8. Examples of Correct Usage

```python
mgr = AggregationManager(
    index_cols=["month_id", "country_id"],
    target_cols=["pred_ln_sb_best"],
)

# Add models (from parquet files or DataFrames)
mgr.add_model("model_a/predictions.parquet", weight=0.6, name="model_a")
mgr.add_model("model_b/predictions.parquet", weight=0.4, name="model_b")

# Aggregate distributions
result = mgr.aggregate(method="concat", use_weights=True)

# Point predictions
mgr_point = AggregationManager(
    index_cols=["month_id", "country_id"],
    target_cols=["pred_ln_sb_best"],
)
mgr_point.add_model(point_df_a, name="a")
mgr_point.add_model(point_df_b, name="b")
result = mgr_point.aggregate(method="median", use_weights=False)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: aggregating before adding models
mgr = AggregationManager(target_cols=["y"])
mgr.aggregate()  # raises ValueError

# WRONG: mixing point and distribution models
mgr.add_model(point_df)
mgr.add_model(distribution_df)  # raises ValueError

# WRONG: using weights with median
mgr.aggregate(method="median", use_weights=True)  # raises ValueError

# WRONG: models with different index rows
mgr.add_model(df_100_rows, name="a")
mgr.add_model(df_99_rows, name="b")  # raises ValueError
```

---

## 10. Test Alignment

Tests live in `tests/test_modules/test_ensemble_aggregator.py`.

| Test area | Covers |
|---|---|
| `test_check_model_consistency_type_mismatch_raises` | Prediction type mismatch detection |
| `test_check_model_consistency_sample_size_mismatch_raises` | Sample size mismatch detection |
| Point aggregation tests | Mean, median, min, max methods |
| Distribution aggregation tests | Concat and vincentization methods |
| Weight validation tests | Weight normalisation, excess weight detection |
| Arrow parquet tests | `_load_parquet_direct` schema validation |
| `_arrow_series_to_numpy` tests | Zero-copy extraction correctness |

---

## 11. Evolution Notes

- Originally a simpler aggregation utility. Extended to support both point and distributional predictions with the introduction of ensemble distributional forecasting.
- The `_load_parquet_direct` path was added alongside `PredictionFrameConverter.to_arrow_table` to enable zero-copy read of Arrow-native parquet files, bypassing the `CMDataset`/`PGMDataset` preprocessing overhead.
- `_arrow_series_to_numpy` was introduced to eliminate `series.to_list()` + `np.asarray()` which created Python list intermediaries and caused memory pressure on large ensembles.
- Vincentization was added as an alternative to concatenation for distribution pooling, providing a quantile-averaging approach.

---

## 12. Known Deviations

- **Lives in `modules/ensemble_aggregator/`**: The class is in `modules/ensemble_aggregator/aggregator.py`, which is appropriate. The user prompt mentions `modules/dataloaders/dataloaders.py` but the actual `AggregationManager` class lives in the aggregator module. There may be residual duplication or confusion between the two locations (R9).
- **`self.aggregated_df` inconsistency**: Set by `_aggregate_distributions` but not by `_aggregate_point_predictions`. Callers should rely on the return value, not the instance attribute.
- **`_load_to_polars` uses `_ViewsDataset` constructor**: This triggers full preprocessing (zero-filling, array conversion) even when the data is already clean. For large DataFrames this adds unnecessary overhead compared to the direct parquet path.
- **Deterministic seed**: `_concatenate_aggregation` uses a hardcoded seed (`rng(42)`). This ensures reproducibility but means results are not randomised across runs. This is intentional but should be documented for users who expect stochastic behaviour.

---

## End of Contract

This document defines the **intended meaning** of `AggregationManager`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
