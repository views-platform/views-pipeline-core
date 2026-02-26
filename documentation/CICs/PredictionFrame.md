# Class Intent Contract: PredictionFrame

**Status:** Draft
**Owner:** Orchestration Core
**Last reviewed:** 2026-02-25
**Related ADRs:** ADR-012 (Authority over Inference), ADR-013 (EvaluationFrame Parity)

---

## 1. Purpose

The canonical, framework-agnostic representation of a model's inference output. It encapsulates predictions and their associated spatiotemporal metadata, serving as the universal transport object between Models and the Pipeline Core.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** carry ground truth (`y_true`). That is the responsibility of the Orchestrator/EvaluationFrame.
- This class does **not** perform alignment with actuals.
- This class does **not** depend on heavy dataframe libraries (Pandas, Polars, Xarray) for its internal state.
- This class does **not** infer identifiers; they must be provided explicitly by the model.

---

## 3. Responsibilities and Guarantees

- **Shape Integrity**: Guarantees that `y_pred` and all identifier arrays have the same number of rows ($N$).
- **Sample Consistency**: Guarantees that `y_pred` is a dense 2D array of shape $(N, S)$, where $S \ge 1$.
- **Metadata Completeness**: Guarantees that every prediction row has associated `time` and `unit` identifiers.
- **Pure Transport**: Acts as a dumb container; does not perform math or transformation logic.

---

## 4. Inputs and Assumptions

- **Raw Inference**: Accepts raw output from models (NumPy arrays).
- **Explicit Metadata**: Assumes the model knows *where* and *when* it predicted (no "guessing" based on position).
- **No NaNs**: Assumes identifiers are complete and valid (no `NaN` in time/unit).

---

## 5. Outputs and Side Effects

- **Transport**: Provides clean access to `y_pred` and `identifiers` for the Orchestrator to consume.
- **Self-Description**: Exposes properties like `sample_count`, `n_rows`, and `identifier_keys`.

---

## 6. Failure Modes and Loudness

- Raises `ValueError` if `y_pred` shape does not match identifier lengths.
- Raises `ValueError` if required identifiers (`time`, `unit`) are missing.
- Fails loud if `y_pred` is not a numeric array.

---

## 7. Boundaries and Interactions

- **Upstream**: Created by **ModelManager** (specifically subclasses like `ForecastingModelManager`) or directly by Models.
- **Downstream**: Consumed by **EvaluationAdapter** (to build `EvaluationFrame`) or **Persistence Layer** (to save forecasts).
- **Isolation**: Minimal dependencies (NumPy).

---

## 8. Examples of Correct Usage

```python
# Created by a Model
pf = PredictionFrame(
    y_pred=np.array([[0.1, 0.2], [0.3, 0.4]]), # 2 rows, 2 samples
    identifiers={
        'time': np.array([100, 101]),
        'unit': np.array([1, 1])
    }
)

# Consumed by Pipeline
n_samples = pf.sample_count  # 2
prediction_array = pf.y_pred
```

---

## 9. Examples of Incorrect Usage

- **Aligning Data**: Trying to join `PredictionFrame` with another frame inside the class. (Use an Adapter/Orchestrator).
- **Storing Truth**: Adding `y_true` to this class. (Use `EvaluationFrame`).
- **Inferring Time**: Passing only `y_pred` and expecting the frame to guess the months.

---

## 10. Test Alignment

- **Invariant Tests**: Verify shape consistency assertions on initialization.
- **Serialization Tests**: Ensure it can be converted to/from storage formats without loss.
- **Parity Tests**: Ensure it captures all data currently carried by `pd.DataFrame` columns in the legacy pipeline.

---

## 11. Evolution Notes

- Future versions may support `xarray` backends if the ecosystem moves that way, but the public API (attributes) should remain stable.
- May eventually replace `pd.DataFrame` as the return type for `Model.predict()`.

---

## End of Contract
