# Class Intent Contract: EvaluationAdapter

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-02-27
**Related ADRs:** ADR-030 (Alignment), ADR-031 (Authority)

---

## 1. Purpose

The definitive bridge between high-level data frameworks (Pandas) and the Pure Math Engine. It is responsible for performing "dirty" data wrangling (intersection, truth-duplication, identifier extraction) to produce a "pure" `EvaluationFrame`.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** calculate metrics.
- This class does **not** perform model training or inference.
- This class does **not** handle I/O or persistence.
- This class does **not** modify prediction values; it only aligns and re-packages them.

---

## 3. Responsibilities and Guarantees

- **Bit-wise Parity**: Guarantees that its alignment logic produces results identical to the legacy internal alignment of `views-evaluation`.
- **Structural Integrity**: Guarantees that the resulting `EvaluationFrame` is internally consistent (shape parity across all arrays).
- **Identifier Extraction**: Guarantees that `time` and `unit` identifiers are correctly extracted from MultiIndex levels or dictionary keys.
- **Lead-time Assignment**: Guarantees that lead-times (steps) are assigned correctly based on either explicit mapping (Preferred) or positional inference (Legacy).

---

## 4. Inputs and Assumptions

- **Raw Data**: Accepts a ground-truth DataFrame and a list of prediction objects (DataFrames or `PredictionFrame`).
- **Target Name**: Requires the explicit name of the target column to extract.
- **Explicit Metadata**: Prefers an explicit `step_mapping` but can fallback to positional inference for legacy support.

---

## 5. Outputs and Side Effects

- **EvaluationFrame**: Produces a `views_evaluation.EvaluationFrame` ready for computation.
- **Immutable Operations**: Does not mutate input DataFrames; returns new array copies or views.

---

## 6. Failure Modes and Loudness

- Raises `ValueError` if the intersection of truth and predictions is empty.
- Raises `ValueError` if required identifiers contain `NaN`.
- Raises `ValueError` if inconsistent sample counts are detected across prediction sequences.
- Fails loud if an explicit `step_mapping` is provided but is missing a required month.

---

## 7. Boundaries and Interactions

- **Upstream**: Interacts with `ForecastingModelManager` and the data loading layer.
- **Downstream**: The only component allowed to instantiate an `EvaluationFrame` for production evaluation tasks.
- **Dependencies**: Depends on `pandas`, `numpy`, and `views_evaluation`.

---

## 8. Examples of Correct Usage

```python
# From DataFrames
ef = EvaluationAdapter.from_dataframes(
    actual=df_truth,
    predictions=[df_pred1, df_pred2],
    target="ged_sb",
    step_mapping={500: 1, 501: 2}
)

# From PredictionFrame
ef = EvaluationAdapter.from_prediction_frame(
    actual=df_truth,
    prediction_frame=pf_inference,
    target="ged_sb"
)
```

---

## 9. Test Alignment

- **Parity Verification**: Verified by the `_audit_parity` logic in the orchestrator.
- **Unit Tests**: Covered by `tests/test_modules/test_evaluation_adapter.py`.
- **Invariant Tests**: Ensure that origins and steps are correctly synthesized across multiple sequences.

---

## End of Contract
