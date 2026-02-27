# Specification: EvaluationFrame Contract

**Status:** Draft
**Owner:** Evaluation Core

## 1. Philosophy
The Evaluation repository is a "Pure Math Engine." It must not depend on high-level data manipulation frameworks (Pandas, Polars, Xarray) for its internal operations. Instead, it operates on a canonical **EvaluationFrame**.

The EvaluationFrame is a collection of synchronized NumPy arrays where all alignment, reindexing, and truth-duplication have already been performed by an external **Adapter**.

## 2. Structure

An `EvaluationFrame` consists of:

### 2.1. Primary Data
*   **`y_true`**: `np.ndarray` of shape `(N,)`.
    *   Type: `float32` or `float64`.
    *   Represents the ground truth observations.
*   **`y_pred`**: `np.ndarray` of shape `(N, S)`.
    *   Type: `float32` or `float64`.
    *   `S` represents the number of samples.
    *   If `S == 1`, it is treated as a point forecast.
    *   If `S > 1`, it is treated as a probabilistic/sample-based forecast.

### 2.2. Identifiers
A mapping of dimension names to `np.ndarray` of shape `(N,)`. These are used for grouping and filtering, but carry no mathematical weight in the core metrics.

Required keys:
*   **`time`**: Temporal marker (usually `month_id`).
*   **`unit`**: Entity marker (usually `country_id`).
*   **`origin`**: Marker for the forecast sequence (e.g., the month the forecast was generated).
*   **`step`**: Marker for lead time (e.g., 1, 2, 3...).

### 2.3. Metadata
A dictionary containing immutable context:
*   `target_name`: str
*   `model_name`: str
*   `task_type`: Literal["regression", "classification"]

## 3. Invariants
1.  **Shape Parity**: `len(y_true) == y_pred.shape[0] == len(identifiers['time']) == ...`
2.  **No NaNs in Identifiers**: All rows must be uniquely and completely identified.
3.  **Pure Arrays**: No list-in-cells. No objects. Only primitive NumPy types.

## 4. Operational Requirements

### 4.1. Grouping (Pure Math)
To reproduce "Month-wise" or "Step-wise" evaluation, the core engine will:
1.  Take an identifier key (e.g., `time`).
2.  Identify unique values and their indices.
3.  Create "Views" (slices) of the `y_true` and `y_pred` arrays for those indices.
4.  Pass those views to the metric functions.

### 4.2. Vectorization
Metric functions should prefer `axis=1` operations on `y_pred` to compute sample-wise statistics, broadcasting `y_true` across the sample dimension.
