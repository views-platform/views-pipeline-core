# Sprint Plan: C-97 + C-99 — Timestamp/Artifact Path Single Source of Truth

**Risk register entries:** C-97 (Tier 3), C-99 (Tier 3)
**Target branch:** `fix/artifact-timestamp-path-agreement`
**Base branch:** `development`
**Estimated effort:** 3–4 hours
**Priority score:** 4.0 (cluster — scope 1 single refactor)

---

## 1. Problem Statement

### The Timestamp Derivation Chain

When a PredictionFrame model runs evaluation or forecasting, the output numpy files
are saved to a directory whose name includes a timestamp:

```
data_generated/predictions_{run_type}_{YYYYMMDD_HHMMSS}/origin_{i}/{target}/y_pred.npy
```

This timestamp is derived at two sites in `model.py`:

- **Eval Track A+** (`model.py:1296-1300`):
  ```python
  _ts = self._model_path.get_latest_model_artifact_path(
      self.args.run_type
  ).stem[-15:]
  ```

- **Forecast Track A+** (`model.py:1486-1488`): identical pattern.

The timestamp comes from `get_latest_model_artifact_path()`, which returns the
**newest** artifact file in the `artifacts/` directory sorted by filename. The last 15
characters of the stem encode the artifact creation timestamp (`YYYYMMDD_HHMMSS`).

### C-99: `--artifact_name` Is Ignored for Save Paths

`ForecastingModelArgs` accepts `--artifact_name` (CLI flag `-a`), which lets a user
specify a particular artifact to evaluate or forecast with. The abstract methods
`_evaluate_model_artifact()` and `_forecast_model_artifact()` receive this argument so
subclasses can load the named artifact. However, the **save path timestamp** always
comes from `get_latest_model_artifact_path()`, ignoring `args.artifact_name`.

**Consequence:** When a user runs `--artifact_name calibration_model_20260101_120000.pt`
(an older artifact), the model loads and evaluates that old artifact but saves the PF
output at the **latest** artifact's timestamp path. The predictions are misattributed.

The ensemble consumer also calls `get_latest_model_artifact_path()`, so producer and
consumer agree on the path in the default case — but both point to the wrong artifact
when `--artifact_name` is used.

An existing falsification test (`test_falsification_pr84_merge_readiness.py:
TestF3ArtifactTimestampAgreement::test_forecast_save_uses_named_artifact_timestamp`)
is `@pytest.mark.xfail(strict=True)` — it documents the bug without passing.

### C-97: DF Ensemble Fuzzy Matching Masks the Same Bug

`DataFrameEnsembleManager._evaluate_model_artifact()` and `_forecast_model_artifact()`
(`dataframe_ensemble.py:604-608`, `644-648`) derive timestamps from
`path_artifact.stem[-15:]` using `get_latest_model_artifact_path()` — the exact same
pattern as the PF path.

The DF path's `_get_generated_predictions_data_file_paths()` (`model_path.py:597-619`)
uses **prefix-only matching**: `f.stem.startswith(f"predictions_{run_type}")`. It
returns the newest file regardless of exact timestamp. This fuzzy fallback masks the
divergence — if timestamps don't match, the newest prediction file is returned anyway.

**Consequence:** If the DF path is ever tightened to exact-match timestamps (for
reproducibility or multi-run disambiguation), it will silently load wrong predictions
or fail to find any.

### The Common (Default) Path Is Correct

When `artifact_name=None` (the default — no `--artifact_name` flag), subclasses also
fall back to `get_latest_model_artifact_path()` to decide which artifact to load. So
the producer (save) and consumer (load) agree. The bug only manifests when a user
explicitly names a non-latest artifact.

Ensembles never pass `artifact_name` — `_create_model_args()` in both DF and PF
ensemble managers constructs `ForecastingModelArgs` without setting it, so it defaults
to `None`. Ensemble-triggered sub-model runs always use the latest artifact.

---

## 2. Root Cause

`get_latest_model_artifact_path()` (`model_path.py:648-686`) has **no `artifact_name`
parameter**. It always returns the latest artifact. The method is called at the save
site (model.py) and at the load site (ensemble managers) independently — they happen
to agree because both call the same method, but neither respects the user's
`--artifact_name` choice.

The fix is a single method: add an `artifact_name` parameter to the timestamp
derivation, falling back to `get_latest_model_artifact_path()` when `None`.

---

## 3. Design

### Option A: Add `artifact_name` to `get_latest_model_artifact_path()` (Rejected)

Overloading `get_latest_model_artifact_path()` to accept an explicit name changes its
semantics from "find the latest" to "find this specific one or the latest." The method
name becomes misleading.

### Option B: New Helper Method (Chosen)

Add a method `resolve_artifact_path(run_type, artifact_name=None)` to
`ModelPathManager`:

```python
def resolve_artifact_path(self, run_type: str, artifact_name: str = None) -> Path:
    """Return the artifact path for the given run_type.

    If artifact_name is provided, resolve it in the artifacts directory.
    Otherwise, fall back to get_latest_model_artifact_path(run_type).
    """
    if artifact_name is not None:
        path = self.artifacts / artifact_name
        if not path.exists():
            raise FileNotFoundError(
                f"ModelPathManager: named artifact '{artifact_name}' "
                f"not found in {self.artifacts}"
            )
        return path
    return self.get_latest_model_artifact_path(run_type)
```

Then change the two save-path derivation sites in `model.py` from:

```python
_ts = self._model_path.get_latest_model_artifact_path(
    self.args.run_type
).stem[-15:]
```

to:

```python
_ts = self._model_path.resolve_artifact_path(
    self.args.run_type, self.args.artifact_name
).stem[-15:]
```

### DF Ensemble: Address C-97

The DF ensemble similarly derives timestamps from `get_latest_model_artifact_path()`:

- `dataframe_ensemble.py:604-608` (eval)
- `dataframe_ensemble.py:644-648` (forecast)

These call sites should be updated to use `resolve_artifact_path(run_type)` (no
`artifact_name` — ensembles don't pass it). This is a no-behavior-change refactor
that makes the ensemble path consistent with the new method, preparing it for any
future support of named artifacts in ensemble sub-model runs.

### PF Ensemble: Already Correct

`PredictionFrameEnsembleManager._evaluate_model_artifact()` and
`_forecast_model_artifact()` (`prediction_frame_ensemble.py:576-580`, `617-621`)
use the same `get_latest_model_artifact_path()` pattern. Update to
`resolve_artifact_path()` for consistency.

---

## 4. Implementation Steps

### Step 1: Add `resolve_artifact_path()` to ModelPathManager

**File:** `views_pipeline_core/data/model_path.py`

Add the method after `get_latest_model_artifact_path()` (around line 686). Include:
- `artifact_name: Optional[str] = None` parameter
- FileNotFoundError if the named artifact doesn't exist (Fail Loud)
- Fallback to `get_latest_model_artifact_path(run_type)` when `None`

### Step 2: Update Save-Path Derivation in model.py

**File:** `views_pipeline_core/managers/model/model.py`

Two sites:
- **Eval** (~line 1296): Replace `get_latest_model_artifact_path(self.args.run_type)`
  with `resolve_artifact_path(self.args.run_type, self.args.artifact_name)`
- **Forecast** (~line 1486): Same replacement

### Step 3: Update Ensemble Timestamp Derivation

**Files:**
- `views_pipeline_core/managers/ensemble/dataframe_ensemble.py` (~lines 604, 644)
- `views_pipeline_core/managers/ensemble/prediction_frame_ensemble.py` (~lines 576, 617)

Replace `get_latest_model_artifact_path(run_type=run_type)` with
`resolve_artifact_path(run_type=run_type)` (no `artifact_name` — ensembles use
`None`). This is a no-behavior-change refactor.

### Step 4: Write Tests

**File:** `tests/test_data/test_model_path.py` (or new file if `test_model_path.py`
doesn't have a natural home for this)

Tests for `resolve_artifact_path()`:
1. `test_resolve_artifact_path_none_returns_latest` — `artifact_name=None` delegates
   to `get_latest_model_artifact_path()`
2. `test_resolve_artifact_path_named_returns_exact` — named artifact returns that
   exact path
3. `test_resolve_artifact_path_missing_raises` — named artifact that doesn't exist
   raises `FileNotFoundError`
4. `test_resolve_artifact_path_stem_matches_artifact` — verify `.stem[-15:]` of the
   returned path contains the artifact's timestamp

**File:** `tests/test_managers/test_model_manager_prediction_format.py`

Tests for the save-path fix:
5. `test_pf_eval_save_uses_named_artifact_timestamp` — when `args.artifact_name` is
   set, the save path uses that artifact's timestamp, not the latest
6. `test_pf_forecast_save_uses_named_artifact_timestamp` — same for forecast path

### Step 5: Unfail the Existing xfail Test

**File:** `tests/test_falsification_pr84_merge_readiness.py`

Remove the `@pytest.mark.xfail(reason="C-99: ...")` decorator from
`test_forecast_save_uses_named_artifact_timestamp`. The test should now pass.

### Step 6: Update CICs

**File:** `documentation/CICs/ModelPathManager.md`

Add `resolve_artifact_path()` to §3 (Responsibilities and Guarantees) and §10 (Test
Alignment).

---

## 5. Files Modified

| File | Change |
|------|--------|
| `views_pipeline_core/data/model_path.py` | Add `resolve_artifact_path()` method |
| `views_pipeline_core/managers/model/model.py` | 2 call sites: eval + forecast timestamp derivation |
| `views_pipeline_core/managers/ensemble/dataframe_ensemble.py` | 2 call sites: eval + forecast timestamp derivation |
| `views_pipeline_core/managers/ensemble/prediction_frame_ensemble.py` | 2 call sites: eval + forecast timestamp derivation |
| `tests/test_data/test_model_path.py` | 4 new tests for `resolve_artifact_path()` |
| `tests/test_managers/test_model_manager_prediction_format.py` | 2 new tests for save-path fix |
| `tests/test_falsification_pr84_merge_readiness.py` | Remove xfail decorator |
| `documentation/CICs/ModelPathManager.md` | §3 + §10 update |
| `reports/technical_risk_register.md` | Resolve C-97, C-99 |

---

## 6. Acceptance Criteria

- [ ] `resolve_artifact_path(run_type, artifact_name)` exists on ModelPathManager
- [ ] `resolve_artifact_path(run_type, None)` delegates to `get_latest_model_artifact_path`
- [ ] `resolve_artifact_path(run_type, "missing.pt")` raises `FileNotFoundError`
- [ ] PF eval save path uses named artifact's timestamp when `--artifact_name` is set
- [ ] PF forecast save path uses named artifact's timestamp when `--artifact_name` is set
- [ ] Default path (`artifact_name=None`) behavior unchanged for all three manager types
- [ ] Existing xfail test (`TestF3ArtifactTimestampAgreement`) passes without xfail marker
- [ ] All ensemble manager timestamp derivation uses `resolve_artifact_path()`
- [ ] `ruff check .` clean
- [ ] Full test suite passes
- [ ] ModelPathManager CIC updated
- [ ] C-97 and C-99 marked Resolved in risk register

---

## 7. Risk Assessment

**Blast radius:** Low-Medium. The fix changes 6 call sites across 3 files, but the
default behavior (`artifact_name=None`) is identical to the current code. The only
behavioral change is when a user explicitly passes `--artifact_name`, which is an
uncommon operation.

**Regression risk:** The existing test suite covers the default path extensively
(TestP2TimestampFromArtifact, TestP4ProducerConsumerPathAgreement). The xfail test
covers the fix path. New tests cover the method itself.

**Ensemble impact:** Ensembles never pass `artifact_name`, so the refactor from
`get_latest_model_artifact_path()` to `resolve_artifact_path(run_type)` is a
no-behavior-change substitution.

**Cross-repo impact:** Zero. `resolve_artifact_path()` is internal to
views-pipeline-core. Downstream model repos call abstract methods that receive
`artifact_name` as a parameter — they don't call `resolve_artifact_path()` directly.

---

## 8. Post-Merge

- Verify the existing xfail test passes in CI (the `strict=True` marker means it
  would fail if the fix works but the xfail is left in place — but we're removing
  the marker, so it should just pass)
- Consider whether `_get_generated_predictions_data_file_paths()` should be tightened
  to exact-match timestamps (separate future work — the fuzzy matching is a
  convenience, not a bug, but it weakens reproducibility)
