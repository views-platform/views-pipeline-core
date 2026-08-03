# Class Intent Contract: CoreDataSniffer

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-03-02
**Related ADRs:** ADR-003 (Authority of Declarations), ADR-008 (Observability), ADR-009 (Boundary Contracts), ADR-041 (Sniffer Pattern)

---

## 1. Purpose

Audits DataFrames loaded from VIEWSER against the expected partition contract —
verifying MultiIndex layout and exact month range — after data is fetched and before
the model sees it. It is the structural gatekeeper between the data-loading layer and
the training/evaluation layer.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** load, fetch, or transform data.
- This class does **not** validate column values, NaN counts, or feature completeness;
  those are the responsibility of the upstream data-preparation pipeline.
- This class does **not** decide which partition to use — that is `ModelManager`'s
  responsibility, passed in at construction.
- This class does **not** modify the DataFrame, its index, or any stored state.
- This class does **not** validate feature columns, queryset structure, or
  model-specific data expectations. A model that requires specific features or
  structural guarantees beyond the universal MultiIndex contract should have its
  own data sniffer living in the model repository, not here.

---

## 3. Responsibilities and Guarantees

- Guarantees that the DataFrame has a valid `pd.MultiIndex` matching **exactly** the
  layout declared for `level` (`pgm`: `(priogrid_gid, month_id)` or `cm`:
  `(country_id, month_id)`).
- Guarantees that the month range in the DataFrame (`min(month_id)`, `max(month_id)`)
  exactly matches the bounds pre-computed from the partition dict at construction.
- Pre-computed bounds (`_first_expected`, `_last_expected`) are immutable after
  construction.

---

## 4. Inputs and Assumptions

- `partition_dict: Dict` — contains `_PARTITION_TRAIN` and `_PARTITION_TEST` sub-dicts,
  each a 2-tuple `(first_month, last_month)`.
- `partition: str` — `"calibration"`, `"validation"`, or `"forecasting"`.
- `level: str` — required; `"pgm"` or `"cm"`. There is no permissive mode. By the
  time this sniffer is called, `CoreConfigSniffer` has already guaranteed `level` is
  a known value; it is never legitimately unknown.
- `override_month: Optional[int]` — when provided, replaces the expected last month
  for forecasting partitions (used when predicting beyond the stored partition end).
- The DataFrame passed to `sniff_loaded_data()` must have a `month_id` MultiIndex
  level; if it does not, the MultiIndex structure check will fail first.

---

## 5. Outputs and Side Effects

- Produces no return value and no mutations.
- On a clean pass: emits `logger.info("CoreDataSniffer: Loaded data audited
  (partition='%s', level='%s').", ...)`.
- On violation: raises immediately with a self-identifying message.

---

## 6. Failure Modes and Loudness

- `ValueError` — flat (non-Multi) index; wrong index names; incomplete month
  coverage (every month in the expected range must be present — interior holes
  fail, #288); at **construction**: unknown partition string (the legacy silent
  forecasting fallback was tightened to fail loud, #288).
- `NotImplementedError` — `level` is not in `EXPECTED_INDEX_NAMES` (raised at
  construction, and again defensively at sniff time via `_check_multiindex`).
- A bool-returning or "soft failure" path does not exist; replace any code that checks
  the return value of this sniffer.

---

## 7. Boundaries and Interactions

- **Called from**: `ViewsDataLoader.get_data()` immediately after the DataFrame is
  fetched, before returning to `ModelManager`.
- `ViewsDataLoader` does not store `level`; it receives it as a parameter of
  `get_data()` and forwards it to the sniffer constructor.
- **Must not** be used to validate prediction output (use `CorePredictionSniffer`).

---

## 8. Examples of Correct Usage

```python
CoreDataSniffer(
    partition_dict=self.partition_dict,
    partition=self.partition,
    level=self.configs["level"],   # always explicit; level is required
    override_month=self.override_month,
).sniff_loaded_data(df)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: checking a return value (sniff_loaded_data returns None)
if CoreDataSniffer(...).sniff_loaded_data(df):
    return df

# WRONG: using on prediction output
CoreDataSniffer(...).sniff_loaded_data(df_predictions)  # use CorePredictionSniffer
```

---

## 10. Test Alignment

- Covered by `tests/test_modules/test_core_data_sniffer.py`.
- Tests cover: both pgm and cm layouts, flat-index rejection, wrong-level rejection,
  calibration / validation / forecasting partition bounds, override_month behaviour,
  and first/last month mismatch detection.

---

## 11. Evolution Notes

- `EXPECTED_INDEX_NAMES` in `core_data_sniffer.py` defines all recognised layouts.
  Add new levels there — not via inline checks. Updating this constant is the only
  change required to support a new spatial resolution.
- `_check_multiindex()` is a module-level utility function in `core_data_sniffer.py`
  shared with `CorePredictionSniffer`. It must remain in that module (composition
  over inheritance).
- Partition keys and run-type spellings live in `data/constants.py`; the
  month-range rule itself is `data/partitions.resolve_month_range` — the single
  implementation shared by the fetch path, this sniffer, and `CoreFrameSniffer`
  (C-209, #288). If the partition contract ever changes, change it THERE — this
  module no longer carries its own copy.

## 12. Known Deviations

- **No schema validation beyond MultiIndex:** Validates index structure and partition compatibility but does not validate column names, dtypes, or value ranges of feature columns.
- **viewser API drift not detected:** If viewser returns a DataFrame with correct structure but changed semantics (renamed features, different units), the sniffer will pass it.

---

## End of Contract

This document defines the **intended meaning** of `CoreDataSniffer`.

Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
