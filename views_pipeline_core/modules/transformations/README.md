# Dataset Transformation Module

> File: `views_pipeline_core/modules/transformations/transformations.py`  
> Core Component: `DatasetTransformationModule`

## Overview

The `DatasetTransformationModule` manages reversible column-wise value transformations applied to VIEWS country-month (CM) and priogrid-month (PGM) datasets. It standardizes log transformations, preserves provenance through a column mapping system, and offers complete undo functionality for converting model-time features or predictions back to interpretable scales for reporting, evaluation, and artifact export.

This module is used primarily:
- After model inference to revert log-space predictions to raw magnitudes
- During feature engineering to align transformation naming conventions (`ln_`, `lx_`, `lr_`)
- For auditability (full transformation history available)
- For ensemble harmonization (consistent naming state across constituent models)

It supports both Pandas and Polars backends (internally normalizing to Polars for performance).

## Core Principles

| Principle | Implementation |
|----------|----------------|
| Reversibility | Every forward transform has a deterministic inverse |
| Provenance | Original column names retained as keys in `column_mapping` |
| Transparency | Rich structured `transformation_history` logging |
| Safety | Duplicate transforms are prevented (prefix inspection) |
| Consistency | Prefix semantics: `ln_`=log(·+1), `lx_`=log(·+exp(offset)), `lr_`=raw/linear state |

## Transformation Semantics

| Prefix | Meaning | Forward Formula | Inverse Formula | Typical Use |
|--------|---------|-----------------|-----------------|-------------|
| `ln_`  | Natural log with +1 shift | y = ln(x + 1) | x = exp(y) − 1 | Count-like targets/features |
| `lx_`  | Natural log with exponential offset | y = ln(x + exp(offset)) | x = exp(y) − exp(offset) | Handling structural zeros |
| `lr_`  | Linear/raw (identity) | y = x | x = y | Explicit raw marking / normalization state |

All undo operations convert the resulting column to the `lr_` naming state.

## Class: `DatasetTransformationModule`

### Initialization

```python
transformer = DatasetTransformationModule(dataset)
```

Args:
- `dataset`: Instance of `CMDataset` or `PGMDataset` with `.dataframe`, `._time_id`, `._entity_id`.

Behavior:
- Converts Pandas → Polars (resetting index) if needed.
- Captures initial columns in `column_mapping`.
- Records temporal and spatial index labels for MultiIndex reconstruction on Pandas export.

Raises:
- `TypeError` if underlying dataframe is not `pd.DataFrame` or `pl.DataFrame`.

### Public Data Access Methods

#### `get_dataframe(as_pandas: bool = True) -> Union[pl.DataFrame, pd.DataFrame]`
Returns transformed dataframe as:
- Pandas: with MultiIndex (`time_id`, `entity_id`)
- Polars: with indices as regular columns

#### `get_current_column_name(original_name: str) -> str`
Resolves the live (latest) column name given its original pre-transformation identifier.

#### `get_all_column_mappings() -> Dict[str, str]`
Full mapping of original → current names (including unchanged).

#### `get_transformed_columns() -> Dict[str, str]`
Subset of mappings where names differ (i.e., transformed columns).

#### `get_transformation_history() -> List[dict]`
Chronological list of transformation records:
```python
[
  {
    "operation": "ln_transform",
    "old_name": "ged_sb_dep",
    "new_name": "ln_ged_sb_dep"
  },
  {
    "operation": "undo_ln_transform",
    "old_name": "ln_ged_sb_dep",
    "new_name": "lr_ged_sb_dep"
  },
  # ...
]
```

### Forward Transform Methods

#### `ln_transform(column_names: List[str]) -> None`
Applies `ln(x + 1)` to each column (handling scalar or `np.ndarray` cell values). Replaces existing `lr_` / `lx_` prefixed versions if present.

Skips:
- Columns already containing `ln` prefix.

#### `lx_transform(column_names: List[str], offset: float = -100) -> None`
Applies `ln(x + exp(offset))`. The `offset` is stored in history for traceability.

Skips:
- Columns already with `lx` prefix.

Careful:
- Must supply the same offset in manual undo flows.

#### `lr_transform(column_names: List[str]) -> None`
Marks raw / identity state via prefix insertion only (no numeric change). Will not apply if the column already contains any transform prefix.

### Reverse / Undo Methods

All reverse operations:
- Drop transformed column
- Create new column in raw (`lr_`) state
- Update `column_mapping`
- Append an undo record to `transformation_history`

#### `undo_ln_transform(column_names: List[str]) -> None`
Applies inverse of `ln_transform`: `exp(y) - 1`.

#### `undo_lx_transform(column_names: List[str], offset: float = -100) -> None`
Applies inverse of `lx_transform`: `exp(y) - exp(offset)`.

Important:
- You must pass the same offset used during forward transform to recover correct values.

#### `undo_lr_transform(column_names: List[str]) -> None`
Removes `lr_` prefix (pure renaming).

#### `undo_all_transformations() -> None`
Bulk reversible sweep:
- Detects all `ln_` and `lx_` columns
- Converts each to `lr_` using default offset for `lx_` (−100)
- Clears `transformation_history` (irreversible to restore history structure)

Use when:
- Preparing outputs for external publication
- Returning all feature columns to interpretable magnitude

#### `undo_transformations(column_names: List[str]) -> None`
Selective multi-type undo:
- Auto-detects each column’s transform prefix (`ln_`, `lx_`, `lr_`)
- Undoes only those with `ln_` or `lx_`
- Leaves `lr_` columns unchanged
- Does NOT clear global history

### Internal Utility Methods (Private)

| Method | Purpose |
|--------|---------|
| `_update_column_mapping(old_name, new_name)` | Chain-safe name propagation |
| `_validate_column_exists(column)` | Defensive precondition enforcement |
| `_has_transform_prefix(column, prefix)` | Idempotency detection |
| `_add_transform_prefix(column, new_prefix, old_prefix=None)` | Prefix insertion w/ `pred_` preservation |
| `_remove_transform_prefix(column, prefix)` | Clean prefix removal with safety checks |

These internal methods enable safe incremental transformations even when a column undergoes multiple prefix transitions (e.g. `lx_` → `ln_` → `lr_`).

## Example Workflow

```python
from views_pipeline_core.data.handlers import CMDataset
from views_pipeline_core.modules.transformations import DatasetTransformationModule

# Load dataset
dataset = CMDataset(source="country_month.parquet")

# Initialize transformer
tx = DatasetTransformationModule(dataset)

# Apply transformations
tx.ln_transform(["ged_sb_dep"])
tx.lx_transform(["ged_ns_dep"], offset=-50)

# Inspect changes
print(tx.get_transformed_columns())
# {'ged_sb_dep': 'ln_ged_sb_dep', 'ged_ns_dep': 'lx_ged_ns_dep'}

# Undo one
tx.undo_ln_transform(["ln_ged_sb_dep"])

# Undo everything remaining
tx.undo_all_transformations()

# Export for reporting
df_raw = tx.get_dataframe(as_pandas=True)
```

## Common Usage Patterns

| Scenario | Recommended Method |
|----------|--------------------|
| Prepare model input features | `ln_transform` / `lx_transform` |
| Mark raw target columns | `lr_transform` |
| Revert predictions before evaluation | `undo_ln_transform` / `undo_lx_transform` |
| Fully normalize for external API | `undo_all_transformations` |
| Audit transformation lineage | `get_transformation_history` |
| Map original → current | `get_current_column_name` |

## Robustness Guarantees

- No in-place mutation of original dataset object
- All transformation outputs are copies
- Safe handling of array-valued cells (supports probabilistic posterior samples)
- Consistent naming for ensemble interoperability
- History clearing is explicit (only in `undo_all_transformations`)

## Error Handling & Logging

Severity mapping:
- ERROR: Structural invalid column / bad input type
- WARNING: Attempt to re-transform already transformed columns
- INFO: Lifecycle and batch operation summaries
- DEBUG: Low-level mapping and prefix transitions

Integrate logging configuration via the pipeline’s global `logging.yaml` (see project `/documentation`).

## Integration Notes

Upstream:
- Receives dataset objects from `views_pipeline_core.data.handlers` layer

Downstream:
- Supplies transformed frames to:
  - Model training managers
  - Forecast reporting and post-processing
  - Visualization modules (after undo steps)
  - Evaluation metrics (expect raw-scale targets)

## Best Practices

1. Avoid stacking different log transforms (e.g., applying `ln_` to an `lx_` column) unless analytically justified.
2. Always undo transformations before publishing raw magnitudes externally.
3. Store offset values (for `lx_transform`) externally if manual undo is required later with non-default settings.
4. Use `get_transformed_columns()` to drive automated export routines.
5. Prefer `undo_transformations()` over `undo_all_transformations()` when partial reversibility is needed.

## Potential Pitfalls

| Pitfall | Mitigation |
|---------|------------|
| Using wrong offset on undo of `lx_` | Track offset in `transformation_history` |
| Attempting `lr_transform` on already transformed columns | The method warns and skips safely |
| Forgetting to revert predictions before metric calculation | Enforce a pre-evaluation undo step in pipeline |
| Clearing history unintentionally | Avoid calling `undo_all_transformations()` unless intentional |

## Reference Links

- Pipeline Development Guidelines: `documentation/development_guidelines.md`
- Data Handler Interfaces: `views_pipeline_core/data/handlers.py`
- Logging Standards: ADR `025_log _level_standards.md`
- Transformation Naming Conventions: Appears indirectly in artifact and prediction ADRs (`012`, `013`, `004`)

## FAQ

| Question | Answer |
|----------|--------|
| Does it mutate the original dataset? | No, data is copied during initialization. |
| Can I reapply `ln_transform` after undo? | Yes—undo converts to `lr_`, enabling clean reapplication. |
| Are probabilistic arrays supported? | Yes, via `map_elements` with `np.ndarray` handling. |
| How do I know which columns changed? | Call `get_transformed_columns()`. |
| Can I chain ln → lx → ln? | Technically yes; mapping adjusts, but interpretability may suffer—use sparingly. |
