# Class Intent Contract: DatasetTransformationModule

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology of the Repository), ADR-003 (Authority of Declarations)

---

## 1. Purpose

Manages reversible data transformations for the VIEWS forecasting pipeline. Provides logarithmic transformations (`ln`, `lx`, `lr`) with full undo capability, column name tracking through transformation chains, and transformation history logging. Used primarily in forecast reporting to undo transformations applied during model training, ensuring predictions are presented in interpretable scales.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** modify the original dataset object passed to the constructor.
- Does **not** perform any statistical analysis, model training, or inference.
- Does **not** handle feature engineering beyond logarithmic transformations and prefix-based naming.
- Does **not** validate data quality (NaN handling, outlier detection, range checking).
- Does **not** persist transformations to disk; the history exists only in memory.
- Does **not** support arbitrary user-defined transformations. Only `ln`, `lx`, and `lr` are implemented.

---

## 3. Responsibilities and Guarantees

**Construction:**
- Accepts `CMDataset` or `PGMDataset`. Converts Pandas DataFrames to Polars internally. Raises `TypeError` for unsupported dataframe types.
- Initializes `column_mapping` as identity mapping (`{col: col}` for all columns).
- Initializes `transformation_history` as empty list.
- Records `_temporal_index` and `_spatial_index` from the dataset's `_time_id` and `_entity_id`.

**Forward transformations:**
- **`ln_transform(column_names)`**: Applies `ln(x + 1)`. Renames column with `ln_` prefix (replacing `lr_` or `lx_` prefix if present). Skips columns that already have `ln_` prefix.
- **`lx_transform(column_names, offset=-100)`**: Applies `ln(x + exp(offset))`. Renames column with `lx_` prefix. Stores `offset` in transformation history.
- **`lr_transform(column_names)`**: Identity transform (no value change). Adds `lr_` prefix to mark columns as linear/raw. Refuses to apply to `ln_` or `lx_` prefixed columns.

**Reverse transformations:**
- **`undo_ln_transform(column_names)`**: Applies `exp(x) - 1`. Changes `ln_` prefix to `lr_`. Skips non-`ln_` columns.
- **`undo_lx_transform(column_names, offset=-100)`**: Applies `exp(x) - exp(offset)`. Changes `lx_` prefix to `lr_`. Caller must supply the same offset used in the forward transform.
- **`undo_lr_transform(column_names)`**: Removes `lr_` prefix (identity, no value change). Skips non-`lr_` columns.
- **`undo_all_transformations()`**: Iterates the transformation history in reverse and undoes each transform. Clears the history afterward.

**Data access:**
- **`get_dataframe(as_pandas=True)`**: Returns transformed data. As Pandas: sets `MultiIndex` on `(_temporal_index, _spatial_index)`. As Polars: returns the internal dataframe.
- **`get_current_column_name(original_name)`**: Resolves current column name through the entire transformation chain. Raises `KeyError` if original name was never in the dataframe.
- **`get_all_column_mappings()`**: Returns copy of full `{original: current}` mapping dict.
- **`get_transformed_columns()`**: Returns only columns where original != current.
- **`get_transformation_history()`**: Returns copy of the chronological history list.

**Invariants maintained by all transformations:**
- Every transformation creates a new column and drops the old one.
- `column_mapping` is updated via `_update_column_mapping()` to preserve the chain from original name to current name.
- Each transformation appends a record to `transformation_history` with `operation`, `old_name`, `new_name`, and (for `lx`) `offset`.

---

## 4. Inputs and Assumptions

- Constructor requires a `CMDataset` or `PGMDataset` instance with a `.dataframe` attribute (Polars or Pandas) and `._time_id` / `._entity_id` string attributes.
- Column names use underscore-separated conventions. Transformation prefixes (`ln_`, `lx_`, `lr_`) are detected by splitting on `_` and checking parts. The prefix `pred_` is handled specially (prefix inserted after `pred_`).
- Cells may contain numpy arrays (object dtype in Polars), which the transformations handle via `isinstance(x, np.ndarray)` checks in `map_elements`.
- `undo_lx_transform()` requires the caller to supply the same `offset` used in the original `lx_transform()`. There is no automatic offset recovery from history.

---

## 5. Outputs and Side Effects

- All transformation methods return `None` and mutate `self.dataframe`, `self.column_mapping`, and `self.transformation_history` in place.
- `get_dataframe()` returns a copy (Pandas with MultiIndex or Polars DataFrame).
- `get_all_column_mappings()` and `get_transformation_history()` return copies.
- Extensive logging via `logging.getLogger(__name__)` at INFO and DEBUG levels.

---

## 6. Failure Modes and Loudness

- **Column not found**: `_validate_column_exists()` raises `ValueError` with message listing available columns.
- **Invalid column name after prefix removal**: `_remove_transform_prefix()` raises `ValueError` if only `"pred"` remains.
- **Invalid dataset type**: Constructor raises `TypeError` if `.dataframe` is neither Polars nor Pandas.
- **Original name not in mapping**: `get_current_column_name()` raises `KeyError`.
- **Duplicate transformation**: Silently skips with a `logger.warning()`. Does not raise.
- **Wrong offset on undo**: Completes without error but produces incorrect values. No validation against history.

---

## 7. Boundaries and Interactions

- **Canonical location**: `views_reporting.transformations` (extracted from pipeline-core via ADR-054).
- **Re-export shim**: `views_pipeline_core.modules.transformations` re-exports from `views_reporting` for backwards compatibility. The shim raises `ImportError` if `views-reporting` is not installed.
- **Depends on**: `CMDataset` / `PGMDataset` (from `views_pipeline_core.data.handlers`), `polars`, `numpy`, `pandas`.
- **Used by**: Forecast reporting modules that need to reverse log transformations before presenting results.
- Has no interaction with storage, model training, or external services.
- Operates entirely in memory.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.data.handlers import CMDataset
from views_pipeline_core.modules.transformations import DatasetTransformationModule

dataset = CMDataset(source=my_dataframe, targets=["ged_sb_dep"])
transformer = DatasetTransformationModule(dataset)

# Forward transform
transformer.ln_transform(["ged_sb_dep"])

# Track column names
current_name = transformer.get_current_column_name("ged_sb_dep")
# Returns "ln_ged_sb_dep"

# Get transformed data as Pandas
df = transformer.get_dataframe(as_pandas=True)

# Undo all transformations
transformer.undo_all_transformations()
df_original_scale = transformer.get_dataframe()

# Check what changed
transformed = transformer.get_transformed_columns()
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: Passing raw DataFrame instead of CMDataset/PGMDataset
transformer = DatasetTransformationModule(my_dataframe)  # TypeError

# WRONG: Applying ln to already-ln column (silently skipped)
transformer.ln_transform(["ged_sb_dep"])
transformer.ln_transform(["ln_ged_sb_dep"])  # Skipped with warning

# WRONG: Undoing lx with wrong offset (no error, wrong values)
transformer.lx_transform(["col"], offset=-100)
transformer.undo_lx_transform(["lx_col"], offset=-50)  # Silent but wrong

# WRONG: Looking up a column that never existed
transformer.get_current_column_name("nonexistent")  # KeyError

# WRONG: Applying lr_ to an ln_ column (silently skipped)
transformer.ln_transform(["ged_sb_dep"])
transformer.lr_transform(["ln_ged_sb_dep"])  # Skipped with warning
```

---

## 10. Test Alignment

Tests live in `tests/test_modules/test_transformations.py`. Coverage includes:

- **`TestInitialization`**: Init with Pandas DataFrame, init with Polars DataFrame, column mapping initialization, empty transformation history.
- **`TestLnTransform`**: Single column transform, verifies `ln(x + 1)` correctness via `np.testing.assert_array_almost_equal`, column renaming, old column removal.
- Additional test classes cover `lx_transform`, `lr_transform`, undo operations, column mapping tracking, and `get_dataframe()` output formats.

Tests use `CMDataset` fixtures with random integer data in a `(month_id, country_id)` MultiIndex structure.

---

## 11. Evolution Notes

- The class currently supports only three transformation types (`ln`, `lx`, `lr`). Additional transformations (e.g., Box-Cox, standardization) could follow the same pattern: forward method, undo method, prefix convention, history tracking.
- `undo_all_transformations()` iterates history in reverse. If offset-dependent transforms (`lx`) are used, the offset is stored in the history record and should be recoverable -- but the current undo dispatch logic should be verified for correctness.
- Converting between Polars and Pandas on every `get_dataframe(as_pandas=True)` call may become a performance concern for large datasets.

---

## 12. Known Deviations

- **Polars/Pandas conversion overhead**: The class stores data internally as Polars but `get_dataframe(as_pandas=True)` (the default) converts to Pandas on every call. For large datasets this may cause performance issues.
- **No offset validation on undo**: `undo_lx_transform()` does not cross-reference the `offset` parameter against the value stored in `transformation_history`. A wrong offset produces silently incorrect results.
- **`map_elements` with `return_dtype=pl.Object`**: The transformations store results as Polars Object dtype (numpy arrays in cells) rather than native numeric types. This is necessary for the current data format but means downstream code must handle array-in-cell patterns.

---

## End of Contract

This document defines the **intended meaning** of `DatasetTransformationModule`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
