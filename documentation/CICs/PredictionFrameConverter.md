# Class Intent Contract: PredictionFrameConverter

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-003 (Authority of Declarations), ADR-009 (Boundary Contracts), ADR-042 (PredictionFrame Adoption)

---

## 1. Purpose

`PredictionFrameConverter` is the I/O adapter between the `PredictionFrame` object (ADR-042/042) and the list-in-cell DataFrame/Arrow formats used for disk persistence and ensemble consumption. It owns three concerns:

1. **Format conversion**: PredictionFrame to list-in-cell DataFrame (`to_prediction_df`), to legacy DataFrame list (`to_legacy_dfs`), and to zero-copy Arrow table (`to_arrow_table`).
2. **Structural auditing**: Verifying that conversions preserve row counts and column names (`audit_prediction_structure`).
3. **Parity auditing**: Verifying bit-wise equivalence between the PredictionFrame path and the legacy DataFrame path during the Strangler Fig transition (`audit_parity_ef`).

All public methods are stateless. The class exists purely for cohesion and to provide clean patch points in tests.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** create, validate, or modify `PredictionFrame` objects. It receives them as duck-typed inputs.
- Does **not** persist files to disk. Persistence is the responsibility of `PredictionIOManager`.
- Does **not** load data from VIEWSER, stores, or any external source.
- Does **not** perform any statistical computation on predictions.
- Does **not** handle the `EvaluationFrame` construction -- it only audits two pre-built `EvaluationFrame` objects for parity.

---

## 3. Responsibilities and Guarantees

- **`to_prediction_df(pf, target)`**: Converts a single PredictionFrame to a DataFrame with a `pd.MultiIndex` derived from `pf.identifiers["time"]` and `pf.identifiers["unit"]`, and a single column `pred_{target}` where each cell is a Python list of sample floats. This is a permanent cross-repo contract: the ensemble manager reads `pred_{target}` columns from every model's saved parquet files.
- **`to_legacy_dfs(predictions, target)`**: Maps `to_prediction_df` over a list of PredictionFrames. Marked as a parity-bridge method -- will be removed when the legacy DataFrame path is retired.
- **`to_arrow_table(pf, target, level)`**: Converts a PredictionFrame to a `pa.Table` with flat columns (`month_id`, entity column, `pred_{target}` as `List<float32>`). Uses zero-copy Arrow construction (no Python list materialisation). Raises `ValueError` if `level` is not in `_LEVEL_TO_ENTITY_COL`.
- **`audit_prediction_structure(pf, df, target)`**: Verifies row count parity between a PredictionFrame and its converted DataFrame, and that the expected `pred_{target}` column exists. Raises `ValueError` on any mismatch.
- **`audit_parity_ef(ef_pf, ef_leg, target)`**: Compares two EvaluationFrame objects field-by-field (`y_pred`, `y_true`, and all four identifier arrays) using `np.testing.assert_allclose` / `assert_array_equal`. Raises `ValueError` with a message beginning `"Parity Failure"` on any divergence.

---

## 4. Inputs and Assumptions

- **PredictionFrame** (duck-typed): Must have `y_pred` (2D ndarray of shape `(n_rows, n_samples)`) and `identifiers` dict with keys `"time"` and `"unit"` (1D arrays of length `n_rows`).
- **EvaluationFrame** (duck-typed): Must have `y_pred`, `y_true` (ndarrays), and `identifiers` dict with keys `"time"`, `"unit"`, `"origin"`, `"step"`.
- **`target`**: A string naming the target variable. Used to construct column name `pred_{target}`.
- **`level`**: For `to_arrow_table` only. Must be `"cm"` or `"pgm"`. Maps to `_LEVEL_TO_ENTITY_COL` (`"country_id"` or `"priogrid_id"` respectively).
- **Module-level constants**: `_TIME_COL = "month_id"`, `_LEVEL_TO_ENTITY_COL = {"cm": "country_id", "pgm": "priogrid_id"}`.

---

## 5. Outputs and Side Effects

- **`to_prediction_df`**: Returns `pd.DataFrame` with MultiIndex and one `pred_{target}` column (list-in-cell).
- **`to_legacy_dfs`**: Returns `List[pd.DataFrame]`, one per input PredictionFrame.
- **`to_arrow_table`**: Returns `pa.Table` with flat columns (no MultiIndex).
- **`audit_prediction_structure`**: Returns `None`. Logs `"PF STRUCTURAL INTEGRITY OK"` on success.
- **`audit_parity_ef`**: Returns `None`. Logs `"EF PARITY CONFIRMED"` on success.
- **No side effects**: All methods are pure functions (no mutation of inputs, no disk I/O, no network calls).

---

## 6. Failure Modes and Loudness

| Condition | Exception | Message pattern |
|---|---|---|
| `to_arrow_table` with unsupported level | `ValueError` | "Unsupported level '{level}'" |
| Row count mismatch in `audit_prediction_structure` | `ValueError` | "PF->DF conversion: PF has {n} rows but converted DF has {m} rows" |
| Missing column in `audit_prediction_structure` | `ValueError` | "PF->DF conversion: expected column 'pred_{target}' not found" |
| `y_pred` divergence in `audit_parity_ef` | `ValueError` | "Parity Failure (y_pred): ..." |
| `y_true` divergence in `audit_parity_ef` | `ValueError` | "Parity Failure (y_true): ..." |
| Identifier divergence in `audit_parity_ef` | `ValueError` | "Parity Failure (identifiers['{key}']): ..." |

All failures are loud. No silent fallbacks.

---

## 7. Boundaries and Interactions

- **Upstream**: Receives `PredictionFrame` objects from the model inference path (via `ForecastingModelManager`).
- **Downstream**: Produced DataFrames are consumed by `PredictionIOManager` for disk persistence. Produced Arrow tables are written directly via `pyarrow.parquet.write_table`.
- **`ViewsForecastsSaver`**: Calls `to_prediction_df()` internally to convert `PredictionFrame` → `pd.DataFrame` before uploading to the views-forecasts central store (Phase 6 Task 4).
- **`AggregationManager`**: Reads the `pred_{target}` list-in-cell parquet files produced by this converter's output chain.
- **`EvaluationAdapter`**: `to_legacy_dfs` output feeds into `EvaluationAdapter.from_dataframes()` during the parity bridge period.

---

## 8. Examples of Correct Usage

```python
converter = PredictionFrameConverter()

# Convert one PF to list-in-cell DataFrame
df = converter.to_prediction_df(pf, target="ln_sb_best")

# Convert list of PFs (parity bridge)
dfs = converter.to_legacy_dfs(predictions, target="ln_sb_best")

# Zero-copy Arrow table for fast parquet write
table = converter.to_arrow_table(pf, target="ln_sb_best", level="pgm")

# Structural audit after conversion
converter.audit_prediction_structure(pf, df, target="ln_sb_best")

# Parity audit between PF and legacy paths
converter.audit_parity_ef(ef_from_pf, ef_from_legacy, target="ln_sb_best")
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: passing unsupported level to to_arrow_table
converter.to_arrow_table(pf, "ln_sb_best", level="admin1")  # raises ValueError

# WRONG: passing a PredictionFrame to audit_parity_ef (expects EvaluationFrame)
converter.audit_parity_ef(pf1, pf2, "x")  # will fail on missing y_true

# WRONG: expecting audit methods to fix mismatches (they only detect)
converter.audit_prediction_structure(pf, bad_df, "x")  # raises, does not repair
```

---

## 10. Test Alignment

Tests live in `tests/test_managers/test_prediction_frame_converter.py`.

| Test class | Covers |
|---|---|
| `TestToPredictionDf` | `to_prediction_df()` -- correct MultiIndex, column name, cell content |
| `TestToLegacyDfs` | `to_legacy_dfs()` -- list mapping, one DF per PF |
| `TestAuditParityEf` | `audit_parity_ef()` -- success path, y_pred mismatch, y_true mismatch, identifier mismatch |
| `TestAuditPredictionStructure` | `audit_prediction_structure()` -- row count mismatch, missing column, success |
| `TestToArrowTable` | `to_arrow_table()` -- correct schema, zero-copy construction, unsupported level |

---

## 11. Evolution Notes

- `to_arrow_table` (Fix A) was added to eliminate Python list materialisation in the parquet write path. It produces backward-compatible parquet that `pd.read_parquet()` reads as object-dtype list cells.
- `to_legacy_dfs` and `audit_parity_ef` are Strangler Fig bridge methods. They will be removed when the legacy DataFrame path is retired (DoD #3 in the PredictionFrame adoption plan).
- The class is stateless by design. If future conversion needs require configuration (e.g., compression settings), prefer constructor parameters over method arguments.

---

## 12. Known Deviations

- **Parity bridge is temporary**: `to_legacy_dfs` and `audit_parity_ef` exist solely for the ADR-042 Strangler Fig transition. They are annotated with `# DoD #3 removal target` comments and should be deleted once the legacy DataFrame path is deprecated.
- **`AssertionError` typo**: The `audit_parity_ef` method catches `AssertionError` (a typo for `AssertionError` -- note: this is Python's `AssertionError` vs `AssertionError`). In practice, `np.testing.assert_allclose` raises `AssertionError` which the except clause may not catch if the typo is `AssertionError`. This is a latent bug that will surface when an actual parity failure occurs.

---

## End of Contract

This document defines the **intended meaning** of `PredictionFrameConverter`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
