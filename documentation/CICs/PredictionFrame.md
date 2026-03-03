# Class Intent Contract: PredictionFrame

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-03-03
**Related ADRs:** ADR-033 (PredictionFrame Adoption)

---

## 1. Purpose

The canonical, framework-agnostic transport object for model inference output.
Carries a dense matrix of posterior-sample predictions together with the explicit
spatiotemporal metadata needed to align them with ground truth. Serves as the
universal handoff between a model and the pipeline's evaluation and persistence
layers, without coupling either side to Pandas.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** carry ground truth (`y_true`). That is the
  `EvaluationFrame`'s responsibility, managed by the `EvaluationAdapter`.
- This class does **not** perform alignment with actuals, intersection, or any
  index join logic.
- This class does **not** depend on Pandas, Polars, Xarray, or any DataFrame
  library for its internal state. Its only dependency is NumPy.
- This class does **not** infer identifiers from array position or data content.
  Identifiers must be provided explicitly by the model; no guessing.
- This class does **not** perform lead-time (step) assignment. That is the
  orchestrator's responsibility (ADR-030, ADR-031).

---

## 3. Responsibilities and Guarantees

- **Shape integrity**: Guarantees that `y_pred` is exactly 2D (`N × S`) with
  `N > 0` and `S ≥ 1`.
- **Identifier completeness**: Guarantees that `identifiers` contains at minimum
  `{"time", "unit"}`, and that every identifier array has exactly `N` entries.
- **No NaN in identifiers**: Guarantees that no identifier array contains NaN.
- **Pure transport**: Acts as a dumb, read-only container. No math, no
  transformation, no side effects.

---

## 4. Inputs and Assumptions

- `y_pred: np.ndarray` — 2D array of shape `(N, S)`. Row `i` is the vector of
  `S` posterior samples for observation `i`. Must be provided by the model.
- `identifiers: Dict[str, np.ndarray]` — must contain at minimum:
  - `"time"`: 1D array of `month_id` integer values (length `N`). The model must
    derive these from `X.index` (the input data's time axis); they are not
    inferred by PredictionFrame.
  - `"unit"`: 1D array of `priogrid_gid` (for `level="pgm"`) or `country_id`
    (for `level="cm"`) integer values (length `N`). Same source: `X.index`.
  Additional identifier keys (e.g. `"origin"`) may be present and are preserved.

---

## 5. Outputs and Side Effects

- Provides clean, read-only access via properties:
  - `y_pred` — the raw NumPy prediction array
  - `identifiers` — the identifier dictionary
  - `n_rows` — `y_pred.shape[0]`
  - `sample_count` — `y_pred.shape[1]`
  - `identifier_keys` — `set(identifiers.keys())`
- No mutations, no side effects.
- `__repr__` returns a summary string for logging.

---

## 6. Failure Modes and Loudness

- `ValueError` — `y_pred` is not 2D; `y_pred.shape[0] == 0`; `y_pred.shape[1]
  < 1`; required identifier key missing; identifier array length ≠ `n_rows`;
  NaN present in any identifier array.
- All validation fires at construction time — there is no deferred or lazy check.

---

## 7. Boundaries and Interactions

- **Created by**: `ModelManager` subclasses via `_forecast_model_artifact()`,
  `_evaluate_model_artifact()`, or `_evaluate_sweep()` when
  `configs["prediction_format"] == "prediction_frame"` (ADR-033).
- **Consumed by**: `PandasAdapter.from_prediction_frame()` /
  `from_prediction_frames()` (converts to `EvaluationFrame` for evaluation)
  and by the `ModelManager` persistence shim (converts to DataFrame for storage
  during the migration window).
- **Not consumed by**: `CorePredictionSniffer` — PredictionFrame is
  self-validating at construction; the sniffer audits only `pd.DataFrame`
  outputs (see `CorePredictionSniffer.md` Section 11 for the migration path).

---

## 8. Examples of Correct Usage

```python
# Created by a model returning PredictionFrame
# X is the input DataFrame with MultiIndex (unit, time)
time_vals = X.index.get_level_values("month_id").values
unit_vals = X.index.get_level_values("priogrid_gid").values  # or country_id

pf = PredictionFrame(
    y_pred=np.stack([samples_draw_1, samples_draw_2], axis=1),  # (N, S)
    identifiers={"time": time_vals, "unit": unit_vals},
)

# Consumed by the adapter
ef = PandasAdapter.from_prediction_frames(
    actual=df_actual,
    predictions=[pf],
    target="ged_sb",
    step_mapping=[{base_origin + s: s for s in steps}],
)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: inferring time from position
pf = PredictionFrame(
    y_pred=predictions,
    identifiers={"time": np.arange(len(predictions)), "unit": unit_vals},
)
# time must be actual month_id values from X.index, not positional indices

# WRONG: adding y_true to PredictionFrame
pf = PredictionFrame(
    y_pred=predictions,
    identifiers={"time": time_vals, "unit": unit_vals, "y_true": actuals},
)
# ground truth belongs in EvaluationFrame, not here
```

---

## 10. Test Alignment

- Covered by `tests/test_data/test_prediction_frame.py`.
- Tests must cover: correct construction; shape mismatch raises; missing
  required identifier raises; identifier length mismatch raises; NaN in
  identifier raises; properties return correct values.

---

## 11. Evolution Notes

- **Active migration in progress.** Per ADR-033, `PredictionFrame` is being
  adopted as the primary model output format via the Strangler Fig pattern.
  Migration sequence: forecast → calibration/validation → sweep. See ADR-033
  for the full migration contract and parity requirements.
- **Persistence format.** During the migration window, `ModelManager` converts
  PredictionFrame to DataFrame for storage (downstream compatibility). When
  `views_evaluation` and `views_hydranet` complete their own migration, the
  storage format moves to a dense binary format (e.g. NumPy `.npz`). This is a
  single-point change in `ModelManager`.
- **`CorePredictionSniffer` extension.** When PredictionFrame is fully adopted,
  `CorePredictionSniffer` will add a PF branch with level-range validation
  (`unit` values in the valid `priogrid_gid` / `country_id` range). See
  `CorePredictionSniffer.md` Section 11 for the two-precondition migration path.

---

## End of Contract

This document defines the **intended meaning** of `PredictionFrame`.

Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
