# Class Intent Contract: _ViewsDataset

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology), ADR-003 (Authority of Declarations), ADR-009 (Boundary Contracts)

---

## 1. Purpose

`_ViewsDataset` is the base class for all VIEWS prediction and feature datasets. It wraps a MultiIndex pandas DataFrame (with two index levels: a time identifier and an entity identifier) and provides:

- Automatic detection of prediction mode (columns prefixed `pred_`) vs. feature mode (explicit `targets` list required).
- Bidirectional conversion between the DataFrame and a 4D numpy tensor of shape `(time x entity x samples x features/targets)`.
- Distribution statistics: summary quantiles (`compute_statistics`), Highest Density Intervals (`calculate_hdi`), Maximum A Posteriori estimates (`calculate_map`), and combined HDI+MAP (`calculate_hdi_map`).
- Subsetting by time, entity, feature, and sample index, with cached tensor splits.
- Integrity checking via tensor round-trip (`check_integrity`).

Concrete subclasses (`_PGDataset`/`PGMDataset` for PRIO-grid, `_CDataset`/`CMDataset` for country) add entity-specific validation and metadata (e.g., country-to-grid mappings, reconciliation support).

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** fetch data from VIEWSER or any external data source. Data must arrive as a `pd.DataFrame`, file path, or `Path`.
- Does **not** train models, run inference, or evaluate predictions.
- Does **not** enforce naming conventions beyond the `pred_` prefix for prediction mode detection.
- Does **not** perform data transformations (log, lag, spatial) -- that is the responsibility of `UpdateViewser` and the transformation library.
- Does **not** persist data to disk or any store.
- Does **not** own the `PredictionFrame` contract. `_ViewsDataset` operates on `pd.DataFrame`; conversion to/from `PredictionFrame` is handled by `PredictionFrameConverter`.

---

## 3. Responsibilities and Guarantees

- **Index validation**: Guarantees the DataFrame has a 2-level `pd.MultiIndex`. Raises `ValueError` if not.
- **Prediction mode auto-detection**: If any column starts with `pred_`, the dataset enters prediction mode. In prediction mode, `targets` is set from `pred_*` columns, `features` is empty, and `sample_size` is validated for consistency across all prediction columns.
- **Feature mode requires explicit targets**: If no `pred_*` columns exist, `targets` must be provided. Raises `ValueError` if `targets is None`.
- **Array normalisation**: All DataFrame cells are converted to `np.ndarray` on init. Scalars become length-1 arrays. Lists become arrays. Existing arrays are dtype-normalized to `float32` (predictions) or left as-is (features).
- **Consistent sample sizes**: In prediction mode, all prediction columns must have the same array length per cell. Raises `ValueError` on mismatch.
- **Tensor shape**: `to_tensor()` always returns a 4D array `(time, entity, samples, vars)`. Prediction tensors use `_prediction_to_tensor`; feature tensors use `_features_to_tensor`.
- **Tensor caching**: `to_tensor()` caches the result in `_prediction_tensor_cache` or `_features_tensor_cache`. `split_data()` caches results in `_split_tensor_cache` (bounded to `_max_tensor_cache_size=128`).
- **Preprocessing on init**: `_preprocess_dataframe` fills missing `(time, entity)` combinations with zeros, using the entity set from the last time step as the canonical set.
- **Statistics only for predictions**: `compute_statistics`, `calculate_hdi`, `calculate_map`, `calculate_hdi_map`, `sample_predictions`, and `report_hdi` all raise `ValueError` if `is_prediction` is `False`.
- **`_BASE_YEAR = 1980`**: Class-level constant used for time-to-date conversions by subclasses.

---

## 4. Inputs and Assumptions

- **`source`**: `Union[pd.DataFrame, str, Path]`. If a path, the file is read via `read_dataframe()`. Must have a 2-level MultiIndex.
- **`targets`**: `Optional[List[str]]`. Required when the DataFrame has no `pred_*` columns. Ignored (with warning) when `pred_*` columns are present.
- **`broadcast_features`**: `bool`. When `True`, scalar features are broadcast to match the sample size of distributional columns. When `False` (default), scalars are wrapped in length-1 arrays and tensor operations are disabled (`sample_size = None`).
- Assumes all `pred_*` column cells contain arrays of the same length within each column.
- Assumes the DataFrame is complete or can be completed by zero-filling from the last time step's entity set.

---

## 5. Outputs and Side Effects

- **`to_tensor()`**: Returns `np.ndarray` of shape `(time, entity, samples, vars)`.
- **`to_dataframe(tensor)`**: Converts a 4D tensor back to a `pd.DataFrame` with the original MultiIndex and column structure.
- **`split_data()`**: Returns `(X, y)` tuple of 4D tensors. `X` has feature dimensions; `y` has target dimensions.
- **`compute_statistics()`**: Returns a DataFrame with columns `{var}_{stat}` for mean, std, q05, q25, q50, q75, q95, q98, q100.
- **`calculate_hdi(alpha)`**: Returns a DataFrame with `{var}_hdi_lower` and `{var}_hdi_upper` columns.
- **`calculate_map()`**: Returns a DataFrame with `{var}_map` columns. Uses `joblib.Parallel` for batch processing.
- **`calculate_hdi_map()`**: Returns a DataFrame with both HDI and MAP columns in a single pass.
- **`get_subset_tensor()` / `get_subset_dataframe()`**: Return subsetted views by time, entity, sample, and feature.
- **`to_reconciler(feature, time_id)`**: Returns a `torch.Tensor` of shape `(samples, entities)` for use by `ForecastReconciler`. Automatically un-logs `ln_*` features.
- **Side effects**: Mutates `self.dataframe` during init (array conversion, preprocessing, feature broadcasting). Caches tensors on first access.

---

## 6. Failure Modes and Loudness

| Condition | Exception | Message pattern |
|---|---|---|
| Source is not DataFrame/path | `ValueError` | "Invalid input type for ViewsDataset" |
| DataFrame is empty | `ValueError` | "Dataframe is empty or not a valid DataFrame" |
| Index is not MultiIndex | `ValueError` | "DataFrame must have a MultiIndex" |
| Index does not have 2 levels | `ValueError` | "Must have exactly two index levels" |
| `targets=None` in feature mode | `ValueError` | "Targets must be specified for non-prediction dataframes" |
| Missing target columns | `ValueError` | "Missing targets: {set}" |
| Mixed prediction column lengths | `ValueError` | "Inconsistent sample sizes in prediction columns" |
| Features present alongside `pred_*` | `ValueError` | "Prediction dataframe should only contain pred_* columns" |
| Invalid prediction cell type | `TypeError` | "Invalid type ... for prediction column" |
| Tensor ops with `broadcast_features=False` | `ValueError` | "Tensor operations are disabled when broadcast_features=False" |
| Statistics on non-prediction data | `ValueError` | "Statistics only available for prediction dataframes" |
| `split_data()` on prediction data | `ValueError` | "Data splitting not applicable to prediction dataframes" |
| Tensor dimension mismatch | `ValueError` | "Mismatch in number of time steps/entities" |

All failures are loud -- no silent fallbacks, no boolean returns from validators.

---

## 7. Boundaries and Interactions

- **Subclasses**: `_PGDataset` (PRIO-grid), `_CDataset` (country), and their public wrappers `PGMDataset`, `CMDataset`. Subclasses override `validate_indices()` to enforce specific entity index names (`priogrid_id`, `country_id`).
- **`PosteriorDistributionAnalyzer`** (Layer 3, `views_pipeline_core.modules.statistics`): Used for MAP and HDI computation via `self._posterior_distribution_analyser.analyze()`.
- **`PlotDistribution`** (Layer 3, `views_pipeline_core.modules.visualizations`): Lazy-imported for `plot_map()` and `plot_hdi()`.
- **`ModelPathManager`**: Used in `__init__` to check if `source` is a path (via `_is_path`).
- **`ForecastReconciler`**: Receives `torch.Tensor` output from `to_reconciler()`.
- **`AggregationManager`**: Wraps `_ViewsDataset` internally via `CMDataset`/`PGMDataset` in `_load_to_polars()`.

---

## 8. Examples of Correct Usage

```python
# Prediction mode (auto-detected from pred_* columns)
ds = _ViewsDataset(prediction_dataframe)
assert ds.is_prediction
tensor = ds.to_tensor()  # shape: (T, E, S, V)
stats = ds.compute_statistics()
hdi = ds.calculate_hdi(alpha=0.9)
map_df = ds.calculate_map(enforce_non_negative=True)

# Feature mode (explicit targets)
ds = _ViewsDataset(feature_df, targets=["ln_sb_best"], broadcast_features=True)
X, y = ds.split_data()

# Subsetting
subset = ds.get_subset_tensor(time_ids=[500, 501], features=["pred_var1"])

# Round-trip integrity
assert ds.check_integrity()
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: no targets in feature mode
ds = _ViewsDataset(feature_df)  # raises ValueError

# WRONG: calling statistics on feature data
ds = _ViewsDataset(feature_df, targets=["x"])
ds.compute_statistics()  # raises ValueError

# WRONG: tensor ops with broadcast_features=False (default)
ds = _ViewsDataset(feature_df, targets=["x"])
ds.to_tensor()  # raises ValueError

# WRONG: split_data on prediction data
ds = _ViewsDataset(prediction_df)
ds.split_data()  # raises ValueError
```

---

## 10. Test Alignment

Tests live in `tests/test_utils/test_views_dataset.py`.

| Test class | Covers |
|---|---|
| `Test_ViewsDatasetInitialization` | Valid init, prediction mode detection, invalid source, missing targets, None targets |
| `TestTensorConversion` | Features-to-tensor, predictions-to-tensor, round-trip integrity |
| Additional tests | Subset operations, statistics, HDI, MAP |

Key gap: no dedicated tests for `_preprocess_dataframe` zero-fill behaviour or `to_reconciler` log-transform logic.

---

## 11. Evolution Notes

- The class is large (~1600 lines in the base alone) and carries many responsibilities: data loading, validation, tensor conversion, statistical computation, and plotting. A future refactor should consider extracting statistics and plotting into composition-based helpers.
- `broadcast_features` was added to support both scalar and distributional feature datasets without forcing tensor operations on scalar-only data.
- `_split_tensor_cache` was added with a size bound (`_max_tensor_cache_size=128`) to prevent unbounded memory growth.
- `calculate_hdi_map` was added to avoid the redundant double-pass of calling `calculate_hdi` and `calculate_map` separately.

---

## 12. Known Deviations

- **Topology violation**: Imports `PosteriorDistributionAnalyzer` from Layer 3 (`views_pipeline_core.modules.statistics`) directly in the constructor. Per ADR-002, data handlers (Layer 1) should not depend on modules (Layer 3). This creates a circular-ish dependency that complicates testing and import order.
- **Lazy Layer 3 imports**: `PlotDistribution` and `ModelPathManager` are imported inside methods to partially mitigate the topology issue, but `PosteriorDistributionAnalyzer` is still eagerly instantiated.
- **God class**: The class handles data validation, tensor algebra, statistical computation, and visualization dispatch. This violates single-responsibility and makes the class difficult to test in isolation.
- **Mutable init**: `__init__` mutates the input DataFrame in-place (array conversion, preprocessing, feature broadcasting) before assigning to `self.dataframe`. Callers must be aware that the DataFrame they passed may be modified.

---

## End of Contract

This document defines the **intended meaning** of `_ViewsDataset`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
