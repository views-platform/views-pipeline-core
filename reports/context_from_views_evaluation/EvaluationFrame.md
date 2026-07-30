# Class Intent Contract: EvaluationFrame

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-02-25  
**Related ADRs:** ADR-010 (Ontology), ADR-011 (Topology), ADR-012 (Authority)

---

## 1. Purpose

The canonical, framework-agnostic internal representation of a forecasting evaluation task. It encapsulates synchronized NumPy arrays for observations, predictions, and identifiers.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform data alignment or index intersection.
- This class does **not** handle I/O (loading from or saving to disk).
- This class does **not** depend on Pandas, Polars, or Xarray.
- This class does **not** calculate metrics (that is the role of MetricCalculators).

---

## 3. Responsibilities and Guarantees

- **Shape Integrity**: Guarantees that all internal arrays (`y_true`, `y_pred`, and all identifiers) have the same number of rows ($N$).
- **Sample Consistency**: Guarantees that `y_pred` is a dense 2D array of shape $(N, S)$, where $S \ge 1$.
- **Pure NumPy**: Guarantees that no Python objects (lists, dicts) are stored inside data cells.
- **State Immutability**: Provides methods to select subsets of data by creating *new* instances rather than mutating state.

---

## 4. Inputs and Assumptions

- **Pre-aligned Data**: Assumes that the adapter has already performed necessary joins and truth-duplication.
- **Homogeneous Types**: Assumes that all predictions in a single frame share the same task type and sample count.
- **Required Identifiers**: Expects at least `time`, `unit`, `origin`, and `step` identifiers to be present for regrouping.

---

## 5. Outputs and Side Effects

- **Group Indices**: Produces mappings of unique identifier values to integer row indices.
- **Sub-frames**: Produces new `EvaluationFrame` instances for specific slices of data.

---

## 6. Failure Modes and Loudness

- Raises `ValueError` if input arrays have mismatched lengths during initialization.
- Raises `KeyError` if requested grouping identifiers are missing.
- Fails loud if any identifier contains `NaN` (as per ADR-012).

---

## 7. Boundaries and Interactions

- **Upstream**: Created by **Adapters**.
- **Downstream**: Consumed by **NativeEvaluator** and **MetricCalculators**.
- **Isolation**: Must not import anything outside of `numpy` and standard typing.

---

## 8. Examples of Correct Usage

```python
ef = EvaluationFrame(
    y_true=np.array([0, 1]),
    y_pred=np.array([[0.1, 0.2], [0.8, 0.9]]), # 2 samples
    identifiers={'time': np.array([100, 100]), 'unit': np.array([1, 2])}
)
month_groups = ef.get_group_indices('time')
sub_ef = ef.select_indices(month_groups[100])
```
