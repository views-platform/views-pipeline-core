# Remediation Plan: PredictionFrame Open Issues (Post Phase 0–5)

**Date**: 2026-03-03
**Branch**: `feature/samples_for_fao`
**Prerequisite**: Post-mortem `2026-03-03_prediction_frame_first_class_adoption.md`

This plan addresses all 8 open issues identified in the Phase 0–5 state audit,
in priority order.  Each issue includes full context, a precise implementation
spec, required TDD steps, and verification criteria.

All work continues on `feature/samples_for_fao`.  TDD discipline (RED → GREEN →
ruff → full suite) is mandatory for every change.

---

## Issue 1 — CRITICAL: `_assert_predictions_in_step_window()` crashes on PF

### Context

`_execute_model_evaluation()` calls `_assert_predictions_in_step_window(list_df_predictions)`
at line ~2058, before the `prediction_format` dispatch block at line ~2065.
When a model's `_evaluate_model_artifact()` returns `List[PredictionFrame]`,
the method tries to call `df.index.get_level_values(0)` on a PredictionFrame,
which has no `.index` attribute.  The result is a runtime `AttributeError` that
prevents any evaluation from completing.

The method body iterates over `predictions`, gets the MultiIndex level 0 for
each item (the time/month axis), checks for "rogue months" outside the declared
step mapping window, and raises a descriptive `ValueError` with origin-shift
diagnostics if violations are found.  This is the pre-flight guard (ADR-031
Layer 3 logging) that runs before inference starts saving results.

### Implementation

**File**: `views_pipeline_core/managers/model/model.py`

**Method**: `_assert_predictions_in_step_window()`

Current signature (line ~3043):
```python
def _assert_predictions_in_step_window(self, predictions: List[pd.DataFrame]) -> None:
```

Changes needed:

1. Widen the type annotation:
   ```python
   def _assert_predictions_in_step_window(
       self, predictions: Union[List[pd.DataFrame], List[PredictionFrame]]
   ) -> None:
   ```

2. Detect the type of the first item and extract months accordingly.  Both
   PredictionFrame and DataFrame have a time axis; they just expose it differently:
   - DataFrame: `df.index.get_level_values(0).unique()`
   - PredictionFrame: `set(pf.identifiers["time"].tolist())` (already a set, no
     need for `.unique()`)

3. Minimal dispatch — replace the single line that extracts `pred_months`:
   ```python
   # Before:
   pred_months = set(df.index.get_level_values(0).unique())

   # After:
   if isinstance(df, PredictionFrame):
       pred_months = set(df.identifiers["time"].tolist())
   else:
       pred_months = set(df.index.get_level_values(0).unique())
   ```
   Note: the loop variable is named `df` in the current code.  Rename the loop
   variable to `item` to avoid confusion, OR leave the name as-is and add the
   dispatch — keeping the rename minimal reduces diff noise.

4. No other logic changes.  The rogue-month check, the `pred_min`/`pred_max`
   computation, the origin-shift diagnostics, and the raise block all use
   `pred_months` and `sorted(mapping)` — none of these need changing.

5. `pred_min` and `pred_max` are computed from `pred_months` — a set of ints in
   both the DF and PF cases — so those lines are already format-agnostic.

### TDD steps

**Test file**: `tests/test_managers/test_model_manager_prediction_format.py`

Write these tests before implementing (they must be RED first):

```python
class TestAssertPredictionsInStepWindow:
    """Verify that _assert_predictions_in_step_window() handles both types."""

    def _stub(self):
        return object.__new__(_ForecastStub)

    def _mapping(self, base=444, steps=range(1, 37)):
        return {base + s: s for s in steps}

    def test_pf_within_window_passes(self):
        """PF whose time values are all inside the declared window must not raise."""
        pf = PredictionFrame(
            y_pred=np.ones((2, 2)),
            identifiers={"time": np.array([445, 446]), "unit": np.array([1, 2])},
        )
        # Patch _get_evaluation_step_mappings to return one valid mapping
        m = self._stub()
        m._partition_dict = {"calibration": {"train": (121, 444), "test": (445, 492)}}
        m._args = MagicMock(); m._args.run_type = "calibration"
        m._config_manager = MagicMock()
        m._config_manager.get_combined_config.return_value = {
            "steps": list(range(1, 37)), "prediction_format": "prediction_frame"
        }
        m._assert_predictions_in_step_window([pf])

    def test_pf_rogue_month_raises(self):
        """PF with a month outside the declared window must raise ValueError."""
        pf = PredictionFrame(
            y_pred=np.ones((2, 2)),
            identifiers={"time": np.array([445, 999]), "unit": np.array([1, 2])},
        )
        m = self._stub()
        m._partition_dict = {"calibration": {"train": (121, 444), "test": (445, 492)}}
        m._args = MagicMock(); m._args.run_type = "calibration"
        m._config_manager = MagicMock()
        m._config_manager.get_combined_config.return_value = {
            "steps": list(range(1, 37)), "prediction_format": "prediction_frame"
        }
        with pytest.raises(ValueError):
            m._assert_predictions_in_step_window([pf])
```

### Verification

```bash
conda run -n views_pipeline pytest tests/test_managers/test_model_manager_prediction_format.py::TestAssertPredictionsInStepWindow -v
conda run -n views_pipeline pytest --tb=short -q   # full suite green
conda run -n views_pipeline ruff check views_pipeline_core/managers/model/model.py
```

---

## Issue 2 — HIGH: Sweep path has no PF dispatch

### Context

`_execute_model_sweeping()` calls `self._evaluate_sweep(self._eval_type, model)`
and then iterates over the result with:

```python
for i, df in enumerate(df_predictions):
    CorePredictionSniffer(level=self.configs["level"]).sniff_predictions(df, ...)
    self._save_predictions(df, ...)
```

`_evaluate_sweep()` now has a return type of `Union[List[pd.DataFrame], List[PredictionFrame]]`
(Phase 5), but the calling code is entirely DataFrame-centric.  A sweep model
returning PFs will immediately crash when `sniff_predictions` tries to access
`.columns` on a PredictionFrame.

### Implementation

**File**: `views_pipeline_core/managers/model/model.py`

**Method**: `_execute_model_sweeping()` — locate the sniff-and-save loop.

Add dispatch immediately after `df_predictions = self._evaluate_sweep(...)`:

```python
prediction_format = self.configs.get("prediction_format", "dataframe")

if prediction_format == "prediction_frame":
    from views_pipeline_core.modules.validation.adapter import _pf_to_legacy_dfs
    _all_targets = (
        self.configs.get("regression_targets", []) +
        self.configs.get("classification_targets", [])
    ) or self.configs.get("targets", ["unknown"])
    _primary_target = _all_targets[0]
    for i, pf in enumerate(df_predictions):
        df_for_save = _pf_to_legacy_dfs([pf], _primary_target)[0]
        self._save_predictions(df_for_save, self._model_path.data_generated, i, send_alert=False)
else:
    for i, df in enumerate(df_predictions):
        CorePredictionSniffer(level=self.configs["level"]).sniff_predictions(
            df, targets=self.configs["targets"]
        )
        self._save_predictions(df, self._model_path.data_generated, i, send_alert=False)
```

Then, for the `_evaluate_prediction_dataframe()` call within the sweep path (which
already dispatches correctly), no changes are needed — it reads `prediction_format`
from `self.configs.get()` and handles both paths.

Also check for any `_assert_predictions_in_step_window` call in the sweep path
and ensure it is either patched out in tests or uses the updated PF-aware method
from Issue 1.

### TDD steps

Add `TestSweepDispatch` to `test_model_manager_prediction_format.py`:

```python
class TestSweepDispatch:
    """Verify that _execute_model_sweeping() routes by prediction_format."""

    def test_sweep_pf_path_skips_sniffer(self):
        """PF path: CorePredictionSniffer.sniff_predictions must NOT be called."""
        ...
    def test_sweep_df_path_calls_sniffer(self):
        """DF path (regression): sniffer IS called."""
        ...
```

Pattern mirrors `TestEvalDispatch` exactly.  Use `_make_eval_stub()` (or a sweep
variant) with `_test_eval_return = [pf]` and `_evaluate_sweep` returning
`self._test_eval_return`.

### Verification

Same as Issue 1 — full suite, ruff.

---

## Issue 3 — HIGH: Multi-target PF saves only the first target column

### Context

In both `_execute_model_forecasting()` and `_execute_model_evaluation()` (PF
save loop), the current code picks `_primary_target = _all_targets[0]` and calls
`_pf_to_legacy_dfs([pf], _primary_target)[0]`.  This produces a single-column
DF with `pred_{primary_target}` only.

For multi-target models (e.g. a joint regression + classification model) returning
a single `PredictionFrame`, only the first target's predictions are saved.  The
other targets are silently absent from the output files.

This is not immediately fixable without a decision on the multi-target PF
contract.  Two options exist:

**Option A**: Require one PF per target.  `_evaluate_model_artifact()` for a two-
target model would return `List[PredictionFrame]` where each PF corresponds to
one target.  This breaks the 1-PF-per-sequence assumption; the list would have
length `n_sequences × n_targets`.  Not compatible with the rolling-origin
architecture where the list length equals sequence count.

**Option B**: Require one PF per sequence with `y_pred` shape `(N, S * n_targets)`,
with a secondary identifier `"target_idx"` that maps columns to target names.
Complex and non-standard.

**Option C (recommended for this transition)**: Maintain the single-target-per-PF
contract and document it explicitly.  Multi-target models must return a
`Dict[str, PredictionFrame]` (one PF per target name) from their artifact methods.
The dispatch code iterates over the dict.  This is a breaking change to the
abstract method signature but makes the contract explicit.

**Short-term fix (block current silent data loss)**: Add an explicit warning when
`len(_all_targets) > 1` and a PF path is being used, so model authors know their
other targets are being dropped:

```python
if len(_all_targets) > 1:
    logger.warning(
        f"PF path: only '{_primary_target}' will be saved. "
        f"Multi-target PF output is not yet supported. "
        f"Targets {_all_targets[1:]} are dropped. See ADR-042 Issue 3."
    )
```

### Implementation (short-term)

Add the `logger.warning` block immediately before the `_pf_to_legacy_dfs` call
in both `_execute_model_forecasting()` and `_execute_model_evaluation()` PF paths.

### TDD steps

Write a test that asserts the warning is emitted when `len(_all_targets) > 1`.
Use `pytest.warns(None)` or `caplog` / `capfd` as appropriate.  Do not attempt
to fix the multi-target save until the Option C contract is agreed.

---

## Issue 4 — HIGH: No parity audit in forecast PF path

### Context

The `_execute_model_forecasting()` PF path converts the PredictionFrame to a
legacy list-in-cell DF via `_pf_to_legacy_dfs()` but never compares the result
to any reference.  If the conversion is incorrect (e.g. wrong target name,
wrong index levels), the error is silent and the saved forecast will be wrong.

In the evaluation path, `_audit_parity_ef()` catches this class of divergence.
The forecast path should have an equivalent check.

### Proposed audit structure

```python
if prediction_format == "prediction_frame":
    from views_pipeline_core.modules.validation.adapter import (
        _pf_to_legacy_dfs,
        PandasAdapter,
    )
    _pf = self._forecast_model_artifact(self.args.artifact_name)
    _target = (
        self.configs["targets"][0]
        if isinstance(self.configs["targets"], list)
        else self.configs["targets"]
    )
    df_predictions = _pf_to_legacy_dfs([_pf], _target)[0]

    # Parity audit (ADR-042): PF-derived EF vs DF-derived EF.
    # Requires actuals DF for alignment — load lazily.
    # NOTE: Forecast does not have actuals for future months; skip the
    # full EF comparison.  Instead, audit the converted DF structurally:
    # verify column name, MultiIndex names, and shape match the PF arrays.
    _pf_rows, _pf_samples = _pf.y_pred.shape
    _df_rows = len(df_predictions)
    if _pf_rows != _df_rows:
        raise ValueError(
            f"Forecast parity failure: PF has {_pf_rows} rows but "
            f"converted DF has {_df_rows} rows."
        )
    if f"pred_{_target}" not in df_predictions.columns:
        raise ValueError(
            f"Forecast parity failure: expected column 'pred_{_target}' "
            f"not found in converted DF (columns: {list(df_predictions.columns)})."
        )
    logger.info(f"\033[92mFORECAST STRUCTURAL PARITY OK for {_target.upper()}\033[0m")
```

This is a structural audit (row count + column name), not a full EF-level
numerical audit (which requires actuals).  It catches the most common failure
modes (wrong target, wrong conversion shape) without needing ground truth data.

### TDD steps

Add `test_forecast_pf_structural_parity_passes` and
`test_forecast_pf_structural_parity_row_mismatch_raises` to `TestForecastDispatch`.
The mismatch test can inject a PF whose `_pf_to_legacy_dfs` produces a DF with
the wrong number of rows by mocking `_pf_to_legacy_dfs` to return a truncated DF.

---

## Issue 5 — MEDIUM: No validation that PF's `y_pred` corresponds to requested target

### Context

In `_evaluate_prediction_dataframe()`, the PF path builds an EvaluationFrame
by calling:

```python
ef = PandasAdapter.from_prediction_frames(actual, raw_preds, target, step_mappings)
```

`from_prediction_frames()` uses `pf.y_pred` directly — a `(N, S)` numpy array.
There is no named-column check (the DF path checks `f"pred_{target}" not in first_df.columns`).

If a model returns a PF whose `y_pred` corresponds to target B but the config
declares target A, no exception is raised.  The metrics will be computed for
the wrong target.

### Proposed fix

This requires a convention decision.  Two options:

**Option A**: Add an optional `target` attribute to `PredictionFrame`.  The
manager checks `pf.target == target` before calling the adapter.

**Option B**: Document the convention that `_evaluate_model_artifact()` is called
once per target, and the model must return PFs in the correct order (same order
as `regression_targets + classification_targets`).  The manager logs a warning
that target correspondence is the model author's responsibility (same as the
identifier contract).

**Recommended short-term**: Option B — add a clear logger.warning inside the PF
dispatch block stating that `y_pred` must correspond to `target` by convention,
and that this is the model author's responsibility per ADR-042.

### Implementation

Add to `_evaluate_prediction_dataframe()` PF dispatch block:

```python
logger.debug(
    f"PF path: assuming raw_preds[i].y_pred corresponds to target '{target}'. "
    f"Target-to-PF alignment is the model author's responsibility (ADR-042)."
)
```

---

## Issue 6 — MEDIUM: `from_prediction_frame()` (singular) lacks I3 window integrity

### Context

`PandasAdapter.from_prediction_frame()` (the single-PF, origin=0 path) does not
enforce the I3 window integrity invariant that `from_prediction_frames()` and
`from_dataframes()` both enforce.

If a caller passes a `step_mapping` along with a PF that contains months outside
that mapping, the method silently accepts them.  The months are dropped by the
actuals intersection but the violation is never flagged.  This is the
pre-intersection blindspot that the I3 invariant was designed to close.

### Implementation

**File**: `views_pipeline_core/modules/validation/adapter.py`

**Method**: `PandasAdapter.from_prediction_frame()`

After the target column check and before building `pf_index`, add the I3 check:

```python
# INVARIANT I3 — Window integrity (pre-intersection blindspot).
if step_mapping is not None:
    pred_months = set(prediction_frame.identifiers['time'].tolist())
    rogue_months = pred_months - set(step_mapping.keys())
    if rogue_months:
        raise ValueError(
            f"Prediction contains month(s) {sorted(rogue_months)} that are not "
            f"in the declared step_mapping window "
            f"(expected months: {sorted(step_mapping.keys())[:5]}"
            f"{'...' if len(step_mapping) > 5 else ''}). "
            f"This indicates that the declared origin does not match the model's "
            f"actual forecast origin."
        )
```

### TDD steps

Add to `TestPandasAdapter` or create `TestFromPredictionFrameSingular`:

```python
def test_singular_window_integrity_rogue_month_raises(self):
    """I3: singular path raises on months outside declared mapping."""
    pf = PredictionFrame(
        y_pred=np.ones((2, 2)),
        identifiers={"time": np.array([100, 999]), "unit": np.array([1, 2])},
    )
    actual = pd.DataFrame(
        {"lr_sb": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples([(100, 1), (999, 2)], names=["month_id", "pgm_id"]),
    )
    mapping = {100: 1, 101: 2}  # month 999 not in mapping
    with pytest.raises(ValueError, match="base_origin|step_mapping"):
        PandasAdapter.from_prediction_frame(actual, pf, "lr_sb", mapping)
```

---

## Issue 7 — MEDIUM: Asymmetric `prediction_format` access

### Context

- `_execute_model_forecasting()`: `prediction_format = self.configs["prediction_format"]`
  — direct key access, raises `KeyError` if absent.
- `_execute_model_evaluation()`: `prediction_format = self.configs.get("prediction_format", "dataframe")`
  — silent fallback to DF path.
- `_evaluate_prediction_dataframe()`: same `.get()` fallback.

The inconsistency means that a model config missing `prediction_format` will:
- Crash forecasting with a KeyError.
- Silently use the DF path for evaluation.

The `CoreConfigSniffer` is the authoritative enforcement point; by the time any
execution method is reached, the key should be present.  The right behaviour for
execution methods during the transition is `.get("prediction_format", "dataframe")`
everywhere, with a comment explaining that the fallback is for the transition
period only.

### Implementation

**File**: `views_pipeline_core/managers/model/model.py`

In `_execute_model_forecasting()`, change:
```python
prediction_format = self.configs["prediction_format"]
```
to:
```python
# Legacy fallback: "dataframe" preserves existing behaviour for models that
# pre-date the mandatory prediction_format key. CoreConfigSniffer enforces
# the key is present at run time.
prediction_format = self.configs.get("prediction_format", "dataframe")
```

### TDD steps

No new test required.  Verify that the existing `TestForecastDispatch` tests still
pass (they set `"prediction_format"` in the stub config, so this change is transparent).
Also run the full suite to confirm no regressions.

---

## Issue 8 — LOW: `_audit_parity_ef()` invocation never tested end-to-end

### Context

`TestAuditParityEf` tests the method in isolation with SimpleNamespace stubs.
`TestEvalMetricsDispatch.test_eval_pf_path_calls_from_prediction_frames` patches
`_audit_parity_ef` out.  No test ever:
1. Creates a real PredictionFrame with specific predictions.
2. Builds a list-in-cell DF from the same data via `_pf_to_legacy_dfs`.
3. Runs both adapters.
4. Calls `_audit_parity_ef` with the results.
5. Verifies the audit passes (i.e. the two EFs are identical).

This test is essentially the parity closure test for the manager level, analogous
to `TestPfToLegacyDfs.test_parity_closure` at the adapter level.

### Implementation

Add `TestEvalParityAuditEndToEnd` to `test_model_manager_prediction_format.py`:

```python
class TestEvalParityAuditEndToEnd:
    """Verify that _audit_parity_ef fires correctly in _evaluate_prediction_dataframe."""

    def test_parity_audit_passes_for_consistent_data(self):
        """
        When PF data matches DF data exactly, _audit_parity_ef must not raise.
        Uses a real (non-mocked) _audit_parity_ef to verify the full chain:
          PF → from_prediction_frames → ef_pf
          PF → _pf_to_legacy_dfs → from_dataframes → ef_leg
          _audit_parity_ef(ef_pf, ef_leg, target)  ← must pass
        """
        # Build a real PF and run _evaluate_prediction_dataframe with
        # _audit_parity_ef NOT patched. Assert no ValueError is raised.
        ...
```

This requires a real `EvaluationFrame` (not a MagicMock) to be produced by the
adapters.  The test must use a DummyEvaluationFrame or the real class from
`views_evaluation` with the `test_explicit_tasks.py` pollution handled.

**Strategy**: Isolate this test class using `pytest.importorskip` or mark it with
`@pytest.mark.integration` so it can be skipped if `views_evaluation` is unavailable.

---

## Execution Order and Dependencies

```
Issue 7 (asymmetric access)  — standalone, no deps, do first, trivial
     ↓
Issue 1 (step window crash)  — blocks all real PF evaluation runs
     ↓
Issue 6 (singular I3)        — adapter-level, safe to do in parallel with Issue 2
Issue 2 (sweep dispatch)     — mirrors eval dispatch pattern exactly
     ↓
Issue 3 (multi-target warn)  — short-term: just add the warning
Issue 4 (forecast parity)    — standalone, do after Issue 7
     ↓
Issue 5 (target correspondence doc)  — doc + log only
Issue 8 (end-to-end parity test)     — last, after issues 1–4 resolved
```

---

## Verification (full suite after all issues resolved)

```bash
# All tests green
conda run -n views_pipeline pytest --tb=short -q

# Target file lint
conda run -n views_pipeline ruff check \
  views_pipeline_core/managers/model/model.py \
  views_pipeline_core/modules/validation/adapter.py \
  tests/test_managers/test_model_manager_prediction_format.py \
  tests/test_modules/test_evaluation_adapter.py

# Confirm no new test count regression
conda run -n views_pipeline pytest --co -q | tail -3
```

After all issues resolved, `_assert_predictions_in_step_window` should be
callable with a real `List[PredictionFrame]` without crashing.  A smoke test
verifying this end-to-end (without mocking the step-window check) should be
added as part of Issue 1's test suite.
