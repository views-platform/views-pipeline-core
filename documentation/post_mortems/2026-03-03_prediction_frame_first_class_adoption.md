# Post-Mortem: PredictionFrame First-Class Adoption (Phases 0–5)

**Date**: 2026-03-03
**Branch**: `feature/samples_for_fao`
**Owner**: Simon Polichinel von der Maase & Claude Code Agent

---

## 1. What was done?

We extended the transport-parity foundation from the previous phase into a
systematic, TDD-driven migration of `PredictionFrame` as a first-class model
output format throughout `ForecastingModelManager`.  Six phases were executed:

### Phase 0 — Documentation anchors
- **ADR-033** (`documentation/ADRs/033_prediction_frame_adoption.md`): Formalised
  the Strangler Fig pattern for the DF → PF migration.  Defined `prediction_format`
  as the sole dispatch authority (no `isinstance` checks).  Established the parity
  contract and migration sequence (forecast → calibration/validation → sweep).
- Updated CICs for `PredictionFrame`, `EvaluationAdapter`, and `CoreConfigSniffer`
  to reflect their evolving responsibilities.

### Phase 1 — `prediction_format` in CoreConfigSniffer (TDD)
- Added `SUPPORTED_PREDICTION_FORMATS = frozenset({"dataframe", "prediction_frame"})`
  to `core_config_sniffer.py`.
- Added `"prediction_format"` to `MANDATORY_KEYS`.
- Added `_check_prediction_format()` private method called from `sniff_config()`.
- 4 new tests in `tests/test_modules/test_core_config_sniffer.py` (RED → GREEN).

### Phase 2 — `PandasAdapter.from_prediction_frames()` (TDD)
- Implemented `PandasAdapter.from_prediction_frames(actual, List[PF], target,
  List[Dict[int,int]])` in `adapter.py`.
- Mirrors `from_dataframes()` exactly: I2 (mapping list length) and I3 (window
  integrity pre-intersection) invariants both enforced.
- Builds a temporary pandas MultiIndex from PF identifiers for alignment;
  extracts matched rows via `get_indexer()`; no list-in-cell explosion needed.
- 7 new tests in `tests/test_modules/test_evaluation_adapter.py` (RED → GREEN).

### Phase 3 — `_pf_to_legacy_dfs()` parity bridge (TDD)
- Implemented `_pf_to_legacy_dfs(List[PF], target) -> List[pd.DataFrame]` as a
  module-level function in `adapter.py`.
- Converts each PF's `y_pred` rows to `list(row)` cells in a list-in-cell DF —
  the format that `from_dataframes()` expects.
- Marked as `PARITY-BRIDGE ONLY` in docstring; will be removed when DF path retires.
- 5 new tests in `TestPfToLegacyDfs`, including `test_parity_closure` which
  asserts `from_prediction_frames([pf]) == from_dataframes(_pf_to_legacy_dfs([pf]))`.

### Phase 4A — Forecast dispatch + `_audit_parity_ef()` (TDD)
- Added dispatch in `_execute_model_forecasting()`:
  - PF path: bypass `CorePredictionSniffer`, convert via `_pf_to_legacy_dfs()`,
    pass the resulting legacy DF downstream unchanged.
  - DF path: existing behaviour (sniffer + DF downstream).
- Added `_audit_parity_ef(ef_pf, ef_df, target)` to `ForecastingModelManager`:
  compares `y_pred`, `y_true`, and all four identifier arrays between two
  EvaluationFrames using `np.testing.assert_allclose` / `assert_array_equal`;
  raises `ValueError("Parity Failure …")` on any mismatch.
- Created `tests/test_managers/test_model_manager_prediction_format.py` (new file)
  with `_ForecastStub`, `_make_stub()`, and `TestForecastDispatch` /
  `TestAuditParityEf` (6 tests total, RED → GREEN).

**Key debugging insight**: `test_explicit_tasks.py` patches
`sys.modules['views_evaluation.evaluation.evaluation_frame'] = MagicMock()` at
module level, which contaminates any test file that imports `EvaluationFrame`
after it is collected.  Fixed by using `SimpleNamespace` with real numpy arrays
as the duck-typed EvaluationFrame substitute throughout `TestAuditParityEf`.

### Phase 4b — Evaluation dispatch (TDD)
- Added dispatch in `_execute_model_evaluation()` (validate-and-save loop):
  - PF path: skip `CorePredictionSniffer`; convert each PF to a legacy DF via
    `_pf_to_legacy_dfs([pf], primary_target)`; call `_save_predictions()`.
  - DF path: existing `validate_and_save` closure with ThreadPoolExecutor.
- Added dispatch in `_evaluate_prediction_dataframe()` (per-target metric loop):
  - PF path: call `from_prediction_frames()` → `_audit_parity_ef()` → pass
    converted legacy DFs to `evaluation_manager.evaluate()`.
  - DF path: existing `from_dataframes()` path.
- Both paths still run the dual `evaluation_manager.evaluate()` →
  `_audit_parity()` metric-level parity loop (unchanged).
- `TestEvalDispatch` + `TestEvalMetricsDispatch` (4 tests, RED → GREEN).

**Note on fallback**: Evaluation dispatch sites use
`self.configs.get("prediction_format", "dataframe")` rather than direct key
access, to maintain backward compatibility for test stubs that pre-date the
mandatory `prediction_format` config key.  Forecast dispatch uses direct key
access (the inconsistency is a known open issue).

### Phase 5 — Abstract method signatures
- `_forecast_model_artifact()`: `-> Union[pd.DataFrame, PredictionFrame]`
- `_evaluate_model_artifact()`: `-> Union[List[pd.DataFrame], List[PredictionFrame]]`
- `_evaluate_sweep()`: `-> Union[List[pd.DataFrame], List[PredictionFrame]]`
- Docstrings for all three now state the identifier contract explicitly:
  `identifiers["time"]` must be `month_id` values from `X.index` level 0;
  `identifiers["unit"]` must be `priogrid_gid` / `country_id` from level 1.
  This is the model author's responsibility.

---

## 2. Why was it done?

- **ADR-033** formalised a transition away from Pandas-backed predictions that
  couples model inference to the Pandas library and forces list-in-cell sample
  storage that does not scale to subnational resolution.
- `PredictionFrame` eliminates the "Pandas sandwich": model authors writing
  PyTorch / TensorFlow / JAX models no longer need to wrap outputs in DataFrames.
- The Strangler Fig pattern with mandatory parity auditing de-risks the transition
  by running both paths in parallel and comparing outputs bit-for-bit, rather than
  doing a hard cutover.

---

## 3. How was it done?

### TDD discipline
Every implementation phase followed strict RED → GREEN:
1. Write failing tests that describe the new behaviour.
2. Confirm RED (tests fail for the right reason).
3. Implement the minimum code to go GREEN.
4. Ruff clean.
5. Full suite passes.
6. Commit.

### Dispatch authority
`prediction_format` is the sole dispatch key — no `isinstance` checks anywhere
in the dispatch logic.  This follows ADR-031 (no semantic inference) and was
enforced by `CoreConfigSniffer` from Phase 1 onwards.

### Parity bridge pattern
`_pf_to_legacy_dfs()` is the single conversion function that allows PF outputs
to feed into existing DF-centric infrastructure unchanged.  It is explicitly
marked as a bridge and will be deleted when the DF path retires.

### Test isolation strategy
Tests in `test_managers/` use `object.__new__(ForecastingModelManager)` to bypass
`__init__` and wire only the attributes actually read by the method under test.
This keeps stubs minimal and prevents cascade failures when unrelated infrastructure
is missing.

---

## 4. What was learned?

### 4.1. Strangler Fig requires a working pre-flight check on both paths
`_assert_predictions_in_step_window()` was written for DataFrames and was not
updated during the migration.  Tests pass because the test helpers patch it out,
but the production path crashes with `AttributeError` when PF objects are passed.
Lesson: every method that touches `list_df_predictions` needs to be audited for
type assumptions when the list type changes.

### 4.2. Module-level sys.modules patches in older test files contaminate later test runs
`test_explicit_tasks.py` replaces `views_evaluation.evaluation.evaluation_frame`
with a MagicMock at module level.  Any test collected after it that imports
`EvaluationFrame` gets a MagicMock, making real assertions on EvaluationFrame
fields impossible.  The fix (`SimpleNamespace` with real numpy arrays) is correct
but is a code smell that points to a broader need to isolate test session setup.

### 4.3. Parity audits must be end-to-end, not unit-level only
`_audit_parity_ef()` is tested at unit level with SimpleNamespace stubs.  The
invocation of `_audit_parity_ef()` inside `_evaluate_prediction_dataframe()` is
patched out in all manager-level tests.  This means no test ever runs the full
chain: model returns PF → adapters build EFs → parity audit fires on real data.
End-to-end integration tests remain a gap.

### 4.4. Multi-target PF is architecturally underspecified
`PredictionFrame` stores a single `y_pred` array.  For multi-target models, it
is unclear whether one PF per target should be returned, or a single PF covering
all targets (with targets indexed somehow in `y_pred`).  Current code assumes
single-target models and silently drops all but the first target when saving.
This needs an explicit decision before multi-target PF models are onboarded.

### 4.5. `.get("prediction_format", "dataframe")` is the right fallback for the bridge period
Forcing a `KeyError` on missing `prediction_format` inside evaluation methods
breaks all pre-Phase-1 test stubs.  The `CoreConfigSniffer` is the right
enforcement point; manager dispatch methods should tolerate the absence of the
key during the transition with a DF fallback.  The inconsistency between the
forecast site (direct key access) and evaluation sites (`.get()`) should be
harmonised.

---

## 5. Impact Assessment

| Metric | Before | After |
|--------|--------|-------|
| Test count | 699 | 761 (62 added) |
| Failing tests | 0 | 0 |
| PF-path files modified | 0 | 2 (`model.py`, `adapter.py`) |
| Abstract method return types corrected | 0 | 3 |
| Parity audits active | 1 (metric-level) | 2 (EF-level + metric-level) for PF path |
| Known production-blocking bugs | 0 (PF not wired) | 1 (`_assert_predictions_in_step_window`) |

---

## 6. Open Issues and Next Steps

See `documentation/plans/2026-03-03_prediction_frame_open_issues_remediation.md`
for the prioritised remediation plan covering all 8 known gaps.
