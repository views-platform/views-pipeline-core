# Class Intent Contract: CorePredictionSniffer

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-03-02
**Related ADRs:** ADR-003 (Authority of Declarations), ADR-008 (Observability), ADR-009 (Boundary Contracts), ADR-041 (Sniffer Pattern)

---

## 1. Purpose

Audits prediction DataFrames at the output boundary before they reach the evaluation
or persistence layer, verifying that the declared target columns exist, the index
layout is canonical, and the DataFrame is non-empty.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** yet audit `PredictionFrame` objects. `PredictionFrame`
  support is pending two preconditions: (1) definition of which non-redundant
  invariants the sniffer should enforce beyond `PredictionFrame`'s own constructor
  guarantees, and (2) addition of a `prediction_format` config key that explicitly
  declares the expected format at call time — no `isinstance` inference. See
  Section 11 for the planned migration path.
- This class does **not** evaluate predictions (compute metrics or scores).
- This class does **not** modify the DataFrame, its index, or its columns.
- This class does **not** infer the model's level from the DataFrame's index names;
  `level` is required at construction and must be supplied explicitly by the caller.
- This class does **not** accept legacy formats: `priogrid_id` (use `priogrid_gid`)
  and flat-indexed CM DataFrames are rejected outright.
- This class does **not** validate evaluation-specific semantics (step correctness,
  origin alignment, or metric compatibility). Those checks require knowledge that
  lives in the evaluation repository (`views-evaluation`) and must be performed by a
  sniffer that lives there, not here.

---

## 3. Responsibilities and Guarantees

- Guarantees the DataFrame is non-empty.
- Guarantees `targets` is a `str` or `list`.
- Guarantees that every required `pred_{target}` column is present.
- Guarantees the DataFrame has a valid `pd.MultiIndex` matching **exactly** the
  layout declared for `level` (`pgm`: `(priogrid_gid, month_id)` or `cm`:
  `(country_id, month_id)`).

---

## 4. Inputs and Assumptions

- `level: str` — required at construction; `"pgm"` or `"cm"`. There is no permissive
  mode. By the time this sniffer is called, `CoreConfigSniffer` has already guaranteed
  `level` is a known value; it is never legitimately unknown.
- `df: pd.DataFrame` — the prediction DataFrame produced by model inference.
- `targets: Union[str, list]` — the target name(s) declared in the model config.

---

## 5. Outputs and Side Effects

- Produces no return value and no mutations.
- On a clean pass: emits `logger.info("CorePredictionSniffer: prediction DataFrame
  audited.")`.
- On violation: raises immediately with a self-identifying message.

---

## 6. Failure Modes and Loudness

- `ValueError` — empty DataFrame; invalid `targets` type; missing `pred_{target}`
  columns; flat (non-Multi) index; unrecognised or wrong-level index names.
- `NotImplementedError` — `level` is not in `EXPECTED_INDEX_NAMES`.
- Legacy index name `priogrid_id` raises `ValueError` — `priogrid_gid` is the
  canonical name.

---

## 7. Boundaries and Interactions

- **Called from**: `ModelManager` (evaluation, forecasting, and sweep paths) after
  model inference, before evaluation metric computation or prediction saving.
- The sniffer is instantiated per call, not stored on the manager.
- Delegates `_check_multiindex_structure()` to the shared `_check_multiindex()`
  utility function in `core_data_sniffer.py`.

---

## 8. Examples of Correct Usage

```python
CorePredictionSniffer(
    level=self.configs["level"]
).sniff_predictions(df, targets=self.configs["targets"])
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: passing a PredictionFrame — not yet supported; PredictionFrame support
# requires a prediction_format config key and a dedicated check branch (see Section 11)
CorePredictionSniffer(level="pgm").sniff_predictions(prediction_frame, targets="ged_sb")

# WRONG: omitting level — level is required; calling without it raises TypeError
CorePredictionSniffer().sniff_predictions(df, targets="ged_sb")
```

---

## 10. Test Alignment

- Covered by `tests/test_modules/test_core_prediction_sniffer.py` (7 test classes,
  22 test methods).
- `TestSniffPredictionsPass` — happy-path validation: valid pgm and cm DataFrames
  pass without raising.
- `TestBehaviorChanges` — regressions from refactor: flat-index rejection,
  `priogrid_id` rejection, wrong-level index rejection (old "pass" → new "fail").
- `TestEmptyDataFrame` — `ValueError` on zero-row input.
- `TestInvalidTargetType` — `TypeError` when `targets` is not a list.
- `TestMissingPredictionColumns` — `ValueError` when no `pred_` columns found.
- `TestMultiIndexStructure` — flat-index rejection, wrong index names, missing
  level values.
- `TestStrictLevel` — pgm-level rejects cm index; cm-level rejects pgm index.

---

## 11. Evolution Notes

- **PredictionFrame migration path.** `PredictionFrame` will eventually be a parallel
  output format returned by the same inference methods that currently return
  `pd.DataFrame` (models that do not depend on Pandas — e.g. PyTorch/TF — and models
  operating at subnational resolution where DataFrame sample storage does not scale).
  Adding support requires three steps, in order:

  1. **Define the PredictionFrame check contract.** Determine which invariants the
     sniffer should enforce for `PredictionFrame` that `PredictionFrame.__init__()`
     does NOT already guarantee. `PredictionFrame` already enforces non-empty shape,
     required identifiers (`"time"`, `"unit"`), shape consistency, and no NaN.
     Candidates for non-redundant checks: level-appropriate unit validation (for
     `level="pgm"`, are `identifiers["unit"]` values in the priogrid_gid range? for
     `level="cm"`, in the country_id range?) and time coverage against a declared
     partition. If no non-redundant checks can be defined, PredictionFrame validation
     remains entirely at construction time and this sniffer is not extended.

  2. **Add `prediction_format` to `CoreConfigSniffer.MANDATORY_KEYS`.** The value
     must be an explicit declaration (`"dataframe"` or `"prediction_frame"`) — no
     `isinstance` type inference at the sniffer boundary. `CoreConfigSniffer` validates
     the key is a known value before any inference begins.

  3. **Add the PredictionFrame branch.** Update the constructor to accept
     `prediction_format: str`; dispatch to `_check_dataframe_contract()` or
     `_check_prediction_frame_contract()` based on the declared format. Update this
     CIC: move PredictionFrame from Non-Goals to Responsibilities; define new
     invariants; update Sections 4, 6, 9, and 10.

- Canonical index names are defined in `EXPECTED_INDEX_NAMES` in
  `core_data_sniffer.py`. No inline checks; update the constant there.

## 12. Known Deviations

- **PredictionFrame not yet supported:** Currently validates `pd.DataFrame` outputs only. `PredictionFrame` validation is handled by `PredictionFrame._validate_input()` at construction time. CIC Section 11 documents the planned migration.
- **No value range validation:** Validates structural properties (non-empty, correct columns, correct index) but does not validate that prediction values are in reasonable ranges (e.g., non-negative for conflict counts).

---

## End of Contract

This document defines the **intended meaning** of `CorePredictionSniffer`.

Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
