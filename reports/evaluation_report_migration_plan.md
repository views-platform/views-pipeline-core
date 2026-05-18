# Foundational EvaluationReport Migration Plan

**Date:** 2026-04-04
**Author:** Simon / Claude (views-evaluation session)
**Status:** Approved, ready for execution
**Repo:** views-pipeline-core
**Triggered by:** Integration test failure 2026-04-03 (6/6 models crashed)

---

## 1. Incident Summary

On 2026-04-03, all model evaluation runs failed with:

```
ModuleNotFoundError: No module named 'views_evaluation.evaluation.evaluation_manager'
```

**Root cause:** views-evaluation PR #16 (`feat!: threshold metrics, Phase 3 purge, and governance adoption`) was merged into `development` and installed into the `views_pipeline` conda environment as v0.5.0. That release deleted `EvaluationManager` and `PandasAdapter` as part of the Phase 3 legacy code purge. However, views-pipeline-core's `model.py` still contains a lazy import of the deleted class at line 2684:

```python
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
```

**Affected models:** average_cmbaseline, average_pgmbaseline, black_ranger (both calibration and validation partitions). Training completes successfully; only the evaluation step crashes because the import is deferred inside `_evaluate_prediction_dataframe()`.

**Log location:** `views-models/logs/integration_test_2026-04-03_234109/`

---

## 2. Background: The EvaluationManager Lifecycle

### Phase 1-2: Original EvaluationManager (pre-v0.4.0)
- Stateful class that accepted DataFrames, computed metrics via legacy dispatch dicts (`REGRESSION_POINT_NATIVE`, etc.)
- Returned a dict of 2-tuples: `{schema: (dict-of-dataclass-instances, pd.DataFrame)}`
- Constructor: `EvaluationManager(metrics_list)` then later refactored to `EvaluationManager()` (config-driven)

### Phase 3: Wrapper Around NativeEvaluator (v0.4.0, commit ed7cea7)
- `EvaluationManager.evaluate()` became a thin wrapper:
  1. Adapted DataFrames to `EvaluationFrame` via `PandasAdapter`
  2. Delegated to `NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)`
  3. Reformatted the `EvaluationReport` back to the old 2-tuple format:
     ```python
     return {
         schema: (report.get_schema_results(schema), report.to_dataframe(schema))
         for schema in ["month", "time_series", "step"]
     }
     ```
- This was the version views-pipeline-core was coded against

### Phase 4: Deletion (v0.5.0, PR #16)
- `EvaluationManager`, `PandasAdapter`, pandas runtime dependency, legacy dispatch dicts all removed
- Public API: `NativeEvaluator`, `EvaluationFrame`, `EvaluationReport`, `MetricCatalog`
- `NativeEvaluator.evaluate()` returns `EvaluationReport` directly

---

## 3. Architecture Context

### views-evaluation v0.5.0 (3-Layer Architecture)

```
Level 0: Pure Core (NumPy only, no pandas)
  - EvaluationFrame        (validated N-array container)
  - NativeEvaluator        (stateless math engine)
  - MetricCatalog          (genome registry + resolver)
  - native_metric_calculators.py (metric functions)
  - Named Profiles         (BASE_PROFILE, HYDRANET_UCDP_PROFILE)

Level 1: Bridge/Adapter (reserved, empty adapters/)
  - EvaluationReport       (structured results with to_dict(), to_dataframe())

Level 2: Orchestration (external — lives in views-pipeline-core)
  - EvaluationAdapter      (DataFrame/PredictionFrame → EvaluationFrame)
```

### views-pipeline-core Evaluation Data Flow

```
_evaluate_prediction_dataframe() in model.py
  │
  ├─ Load actuals from viewser parquet
  ├─ For each target:
  │   ├─ Build EvaluationFrame via EvaluationAdapter (already done — lines 2760, 2788)
  │   ├─ Call evaluation_manager.evaluate(...)  ← BROKEN IMPORT
  │   ├─ Unpack result as {schema: (metrics_dict, DataFrame)}
  │   ├─ Log metrics_dict to WandB  (wandb/utils.py)
  │   └─ Save DataFrames to disk    (prediction/io.py)
  └─ Send WandB summary alert
```

---

## 4. Return Format Mismatch

### Old: EvaluationManager.evaluate() returned:

```python
{
    "month": (
        {"month445": RegressionPointEvaluationMetrics(MSE=0.1, MSLE=0.05, ...), ...},
        pd.DataFrame(index=["month445",...], columns=["MSE","MSLE",...])
    ),
    "time_series": (
        {"ts00": RegressionPointEvaluationMetrics(...), ...},
        pd.DataFrame(...)
    ),
    "step": (
        {"step01": RegressionPointEvaluationMetrics(...), ...},
        pd.DataFrame(...)
    ),
}
```

Each value is a **2-tuple of (dict mapping group-key to dataclass instance, DataFrame)**.

### New: NativeEvaluator.evaluate() returns:

```python
EvaluationReport(
    target="ged_sb_best",
    task="regression",
    pred_type="point",
    results={
        "month":       {"month445": {"MSE": 0.1, "MSLE": 0.05, ...}, ...},
        "time_series": {"ts00":     {"MSE": 0.1, ...}, ...},
        "step":        {"step01":   {"MSE": 0.1, ...}, ...},
    }
)
```

Internally: `{schema: {group: {metric: float}}}` — plain nested dicts, no dataclass instances, no DataFrames.

### EvaluationReport API:

| Method | Returns | Notes |
|--------|---------|-------|
| `to_dict()` | `{"target":..., "task":..., "pred_type":..., "schemas": {schema: {group: {metric: float}}}}` | Raw dict, no dataclass involvement |
| `to_dict()["schemas"]` | `{schema: {group: {metric: float}}}` | The native format — same structure WandB needs |
| `get_schema_results(schema)` | `{group: dataclass_instance}` | Instantiates 2x2 dataclasses for backward compat |
| `to_dataframe(schema)` | `pd.DataFrame` | Goes through `get_schema_results()` internally |

---

## 5. Downstream Consumer Analysis

### Consumer 1: WandB Logging (`modules/wandb/utils.py`)

**Functions affected:**
- `generate_wandb_step_wise_log_dict` (line 48-71)
- `generate_wandb_month_wise_log_dict` (line 74-97)
- `generate_wandb_time_series_wise_log_dict` (line 100-123)
- `calculate_mean_evaluation_metrics` (line 126-153)

**Current contract:** All three `generate_*` functions call `asdict(dict_of_eval_dicts[key])` which requires dataclass instances. `calculate_mean_evaluation_metrics` uses `vars(item).keys()` and `vars(item).get(key)` — also requires dataclass-like objects.

**What plain dicts provide:** `.items()` iterates key-value pairs (same as `asdict()` output). `.keys()` and `.get(key)` work identically to `vars()` on dataclasses.

**Conclusion:** These functions can accept plain dicts with a simple `isinstance` check, remaining backward-compatible with dataclass inputs.

### Consumer 2: File Saving (`managers/prediction/io.py:146-222`)

**Current contract:** `save_evaluations()` receives three `pd.DataFrame` objects. It calls `save_dataframe()` to write parquets and `wandb.Table(dataframe=df)` to log WandB tables.

**Conclusion:** No change needed. `EvaluationReport.to_dataframe(schema)` returns a proper `pd.DataFrame`. The method already receives unpacked DataFrames, not the 2-tuple.

### Consumer 3: 2-Tuple Unpacking (`model.py:2799-2823`)

**Current contract:** A 24-line defensive block that unpacks `eval_result_dict[key]` as `(metrics_dict, DataFrame)` into 6 local variables. Includes type checks, length checks, and a try/except safety net.

**Conclusion:** Replace entirely with direct `EvaluationReport` access. The report is already validated by `NativeEvaluator` — no defensive unpacking needed.

### Consumer 4: WandB Summary Alert (`model.py:2841-2847`)

**Current contract:** Calls `_generate_evaluation_table(wandb.summary._as_dict())`. Input is `wandb.summary`, not evaluation results directly.

**Conclusion:** No change needed. This consumes WandB's own summary dict, not evaluation output.

### Consumer 5: EnsembleManager (`managers/ensemble/ensemble.py:232`)

**Current contract:** Calls `self._evaluate_prediction_dataframe(df_predictions, self._eval_type, ensemble=True)`. Inherited from `ForecastingModelManager`.

**Conclusion:** Automatically fixed when the parent method is updated.

---

## 6. Detailed Changes

### File 1: `views_pipeline_core/modules/wandb/utils.py`

#### Change 1a: `generate_wandb_step_wise_log_dict` (lines 67-69)

```python
# CURRENT (line 67-69)
    for key, value in asdict(dict_of_eval_dicts[step]).items():
        if value is not None:
            log_dict[f"step-wise/{target_identifier}/{key}"] = value

# REPLACEMENT
    entry = dict_of_eval_dicts[step]
    items = entry.items() if isinstance(entry, dict) else asdict(entry).items()
    for key, value in items:
        if value is not None:
            log_dict[f"step-wise/{target_identifier}/{key}"] = value
```

#### Change 1b: `generate_wandb_month_wise_log_dict` (lines 93-95)

Same pattern — replace `asdict(dict_of_eval_dicts[month]).items()` with the `isinstance` check, using `f"month-wise/{target_identifier}/{key}"` prefix.

#### Change 1c: `generate_wandb_time_series_wise_log_dict` (lines 119-121)

Same pattern — replace `asdict(dict_of_eval_dicts[time_series]).items()` with the `isinstance` check, using `f"time-series-wise/{target_identifier}/{key}"` prefix.

#### Change 1d: `calculate_mean_evaluation_metrics` (lines 140-148)

```python
# CURRENT (lines 140-148)
    first_item = next(iter(evaluation_dict.values()))
    metric_names = vars(first_item).keys()

    for key in metric_names:
        valid_values = [
            value
            for value in (vars(item).get(key) for item in evaluation_dict.values())
            if value is not None
        ]

# REPLACEMENT
    first_item = next(iter(evaluation_dict.values()))
    metric_names = first_item.keys() if isinstance(first_item, dict) else vars(first_item).keys()

    for key in metric_names:
        valid_values = [
            value
            for value in (
                (item.get(key) if isinstance(item, dict) else vars(item).get(key))
                for item in evaluation_dict.values()
            )
            if value is not None
        ]
```

**Backward compatibility:** All changes use `isinstance(entry, dict)` to detect the input type. Dataclass inputs continue to work via the `else` branch. This means any other code that still passes dataclass instances will not break.

---

### File 2: `views_pipeline_core/managers/model/model.py`

#### Change 2a: Import (line 2684)

```python
# CURRENT
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager

# REPLACEMENT
        from views_evaluation import NativeEvaluator
```

#### Change 2b: Constructor (line 2721)

```python
# CURRENT
        evaluation_manager = EvaluationManager()

# REPLACEMENT
        evaluator = NativeEvaluator(self.configs)
```

**Context on `self.configs`:** This is the merged config dict containing keys from both `config_meta.py` and `config_hyperparameters.py`. NativeEvaluator reads:
- `evaluation_profile` (defaults to `"base"` if absent — safe for all existing models)
- `metric_hyperparameters` (defaults to `{}` if absent — safe)
- `regression_targets`, `classification_targets` (already present)
- `regression_point_metrics`, `regression_sample_metrics`, etc. (already present)
- `steps` (already present, from hyperparameters)

Verified model configs:
- `average_cmbaseline`: `regression_point_metrics: ["MSE", "MSLE"]`, no `evaluation_profile` key (uses default `"base"`)
- `black_ranger`: `regression_sample_metrics: ["twCRPS", "QIS", "MIS", "MCR_sample"]`, no `evaluation_profile` key
- `purple_alien`: `evaluation_profile: "hydranet_ucdp"`, `classification_point_metrics: ["AP"]`

#### Change 2c: PF-path evaluate call (lines 2770-2771)

```python
# CURRENT
                    eval_result_dict = evaluation_manager.evaluate(
                        actual_slice, None, target, self.configs, ef=ef
                    )

# REPLACEMENT
                    report = evaluator.evaluate(ef=ef, legacy_compatibility=True)
```

#### Change 2d: DF-path evaluate call (lines 2795-2796)

```python
# CURRENT
                    eval_result_dict = evaluation_manager.evaluate(
                        actual_slice, None, target, self.configs, ef=ef
                    )

# REPLACEMENT
                    report = evaluator.evaluate(ef=ef, legacy_compatibility=True)
```

**Why `legacy_compatibility=True`:** The deleted Phase 4 wrapper (commit `ed7cea7`, line 599) explicitly passed `legacy_compatibility=True` to NativeEvaluator. This flag controls step-wise truncation — when `True`, steps are truncated to the shortest sequence in a rolling-origin evaluation. Flipping to `False` changes numeric results (more steps appear). To isolate the format migration from behavioral changes, we preserve `True` here. Flipping to `False` is a separate, deliberate change (see Deferred Items).

#### Change 2e: Replace 2-tuple unpacking block (lines 2799-2823)

```python
# CURRENT (24 lines)
                # Initialize local variables to avoid UnboundLocalError
                step_wise_evaluation, df_step_wise_evaluation = ({}, pd.DataFrame())
                time_series_wise_evaluation, df_time_series_wise_evaluation = ({}, pd.DataFrame())
                month_wise_evaluation, df_month_wise_evaluation = ({}, pd.DataFrame())

                # Safety check: Ensure all expected keys are present and have enough values to unpack
                for eval_key in ["step", "time_series", "month"]:
                    try:
                        # Attempt to retrieve the 2-tuple (metrics_dict, dataframe)
                        res = eval_result_dict[eval_key]
                        
                        # Type and length check
                        if not isinstance(res, (list, tuple)) or len(res) < 2:
                            raise ValueError(f"Expected 2-tuple, got {type(res)} with length {len(res) if hasattr(res, '__len__') else 'N/A'}")

                        # Unpack into local variables
                        if eval_key == "step":
                            step_wise_evaluation, df_step_wise_evaluation = res
                        elif eval_key == "time_series":
                            time_series_wise_evaluation, df_time_series_wise_evaluation = res
                        elif eval_key == "month":
                            month_wise_evaluation, df_month_wise_evaluation = res
                            
                    except (KeyError, TypeError, ValueError, IndexError) as e:
                        logger.warning(f"Evaluation for {target} returned invalid data for '{eval_key}': {e}. Skipping WandB/File logging for this component.")

# REPLACEMENT (6 lines)
                # Extract native dict format for WandB logging and DataFrames for file saving
                schemas = report.to_dict()["schemas"]
                step_wise_evaluation = schemas.get("step", {})
                time_series_wise_evaluation = schemas.get("time_series", {})
                month_wise_evaluation = schemas.get("month", {})

                df_step_wise_evaluation = report.to_dataframe("step")
                df_time_series_wise_evaluation = report.to_dataframe("time_series")
                df_month_wise_evaluation = report.to_dataframe("month")
```

**Why no try/except:** `NativeEvaluator.evaluate()` always returns an `EvaluationReport` with all three schemas populated (even if empty). The `to_dict()` and `to_dataframe()` methods are well-tested (21 tests in views-evaluation's `test_evaluation_report.py`). Defensive unpacking was needed for the old format because `EvaluationManager` had inconsistent error handling; `NativeEvaluator` follows fail-loud (ADR-013).

#### Change 2f: Stale comments and docstring

```python
# Line 2672 — docstring
# CURRENT:  "- Calculates metrics using EvaluationManager"
# REPLACE:  "- Calculates metrics using NativeEvaluator"

# Lines 2712-2715 — comment block
# CURRENT:
        # Task definitions: target lists only.
        # EvaluationManager reads the metric keys (regression_point_metrics,
        # regression_sample_metrics, etc.) directly from the config passed to
        # evaluate(). Point vs uncertainty dispatch is also handled internally.
# REPLACE:
        # Task definitions: target lists only.
        # NativeEvaluator resolves task/pred_type from config and EvaluationFrame.
        # Metric dispatch is handled by MetricCatalog (ADR-042).

# Line 2734 — comment
# CURRENT:  "# EvaluationManager.evaluate() operates on one target at a time and"
# REPLACE:  "# NativeEvaluator.evaluate() operates on one target at a time via"
```

---

### File 3: New test (suggested location: `tests/test_evaluation_integration.py`)

```python
"""
Regression guard: prevent cross-repo import breakage.

This test was added after the 2026-04-03 incident where views-evaluation
PR #16 deleted EvaluationManager but views-pipeline-core still imported it,
crashing all 6 integration test runs.
"""
import pytest
import numpy as np


def test_native_evaluator_import_and_basic_call():
    """NativeEvaluator can be imported and produces a valid EvaluationReport."""
    from views_evaluation import NativeEvaluator, EvaluationFrame

    config = {
        "regression_targets": ["test_target"],
        "regression_point_metrics": ["MSE"],
        "steps": [1],
    }
    ef = EvaluationFrame(
        y_true=np.array([1.0, 2.0]),
        y_pred=np.array([[1.1], [1.9]]),
        identifiers={
            "time": np.array([100, 100]),
            "unit": np.array([1, 2]),
            "origin": np.array([99, 99]),
            "step": np.array([1, 1]),
        },
        metadata={"target": "test_target"},
    )
    report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)

    # Verify report structure
    result = report.to_dict()
    assert result["target"] == "test_target"
    assert result["task"] == "regression"
    assert result["pred_type"] == "point"
    for schema in ("step", "month", "time_series"):
        assert schema in result["schemas"]

    # Verify to_dataframe() produces a DataFrame with expected metric columns
    df = report.to_dataframe("step")
    assert "MSE" in df.columns


def test_evaluation_report_native_dict_format():
    """EvaluationReport.to_dict()['schemas'] returns plain dicts, not dataclasses.

    This is the format WandB logging now expects after the migration
    from EvaluationManager's 2-tuple format.
    """
    from views_evaluation import NativeEvaluator, EvaluationFrame

    config = {
        "regression_targets": ["t"],
        "regression_point_metrics": ["MSE"],
        "steps": [1],
    }
    ef = EvaluationFrame(
        y_true=np.array([1.0]),
        y_pred=np.array([[1.1]]),
        identifiers={
            "time": np.array([100]),
            "unit": np.array([1]),
            "origin": np.array([99]),
            "step": np.array([1]),
        },
        metadata={"target": "t"},
    )
    report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)
    schemas = report.to_dict()["schemas"]

    # Each schema value should be a dict of dicts, not dataclass instances
    for schema_name, groups in schemas.items():
        assert isinstance(groups, dict), f"{schema_name} is not a dict"
        for group_key, metrics in groups.items():
            assert isinstance(metrics, dict), f"{schema_name}/{group_key} is not a dict"
            for metric_name, value in metrics.items():
                assert isinstance(value, (int, float)), (
                    f"{schema_name}/{group_key}/{metric_name} is {type(value)}, not numeric"
                )
```

---

## 7. What This Achieves

| Before | After |
|--------|-------|
| `model.py` imports deleted `EvaluationManager` | Imports `NativeEvaluator` from public API |
| WandB utils require dataclass instances (`asdict()`) | WandB utils accept plain dicts or dataclasses |
| 24-line defensive 2-tuple unpacking block | 6-line direct `EvaluationReport` access |
| Dataclass round-trip: `dict → dataclass → asdict() → dict` | Direct: `dict → dict` (dataclass only used inside `to_dataframe()`) |
| No cross-repo regression test | Regression guard test prevents future import breakage |
| Stale comments reference deleted class | Comments reference current architecture |

---

## 8. Verification Strategy

### Level 1: Unit-level (views-pipeline-core tests)

```bash
cd /home/simon/Documents/scripts/views_platform/views-pipeline-core
conda run --name views_pipeline pytest tests/ -v --tb=short -x
```

**Success:** All existing tests pass. New regression guard test passes.

### Level 2: Smoke test (single model, ~5 min)

```bash
cd /home/simon/Documents/scripts/views_platform/views-models
conda run --name views_pipeline python models/average_cmbaseline/main.py --run_type calibration
```

**Check:**
- No `ModuleNotFoundError` or any other import error
- Evaluation step completes (log shows `INFO - Evaluating model average_cmbaseline...` followed by WandB sync, no `ERROR` lines)
- Output files written to `models/average_cmbaseline/data/generated/` (3 evaluation parquets: step, ts, month)
- WandB run dashboard shows step-wise, month-wise, time-series-wise metric charts

### Level 3: Format parity (verify output structure)

```bash
# Check generated evaluation files exist and are non-empty
ls -la models/average_cmbaseline/data/generated/*eval*

# Verify DataFrame structure
python -c "
import pandas as pd
from pathlib import Path

gen = Path('models/average_cmbaseline/data/generated')
for f in sorted(gen.glob('*eval*.parquet')):
    df = pd.read_parquet(f)
    print(f'{f.name}: shape={df.shape}, columns={df.columns.tolist()}')
"
```

**Expected:** Column names match metric names from the model config (MSE, MSLE for average_cmbaseline). Row indices match group keys (step01-step36, month445-month480, ts00-ts12).

### Level 4: Full integration (6 runs)

Re-run the integration test that produced the 2026-04-03 failure:
- `average_cmbaseline` x calibration, validation
- `average_pgmbaseline` x calibration, validation
- `black_ranger` x calibration, validation

**Success:** 0 tracebacks across all 6 log files. All evaluation parquets generated. WandB dashboards populated for all models.

### Level 5: Sample-prediction model (black_ranger)

`black_ranger` uses `prediction_format: "prediction_frame"` and `regression_sample_metrics: ["twCRPS", "QIS", "MIS", "MCR_sample"]`. This exercises:
- The PF path (line 2760, `EvaluationAdapter.from_prediction_frames`)
- Sample-based evaluation (`is_sample=True`, S=256 ensemble members)
- Metrics requiring profile hyperparameters (twCRPS needs `threshold`, QIS needs `lower_quantile`/`upper_quantile`, MIS needs `alpha`)

If black_ranger evaluates successfully, the full metric resolution chain (MetricCatalog + BASE_PROFILE) is working end-to-end through pipeline-core.

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| WandB metric key names change (e.g., `Brier` -> `Brier_sample`) | Low | Low | Only affects models that configure new metrics; existing models unchanged |
| `to_dataframe()` raises `ImportError` (pandas not installed) | Very Low | High | views-pipeline-core is pandas-heavy; pandas is always in the conda env |
| `legacy_compatibility=True` doesn't match old behavior exactly | Very Low | Medium | Same flag value used by the deleted Phase 4 wrapper (commit `ed7cea7` line 599) |
| `self.configs` missing `evaluation_profile` key | Low | None | `NativeEvaluator.__init__` defaults to `config.get("evaluation_profile", "base")` |
| Empty schema results (no groups for a schema) | Low | None | `schemas.get("step", {})` returns empty dict; `to_dataframe()` returns empty DataFrame; WandB iterates empty dict harmlessly |
| Model config has metric not in dataclass fields | Low | High | `get_schema_results()` raises `ValueError` — but this only affects `to_dataframe()` path. PR #16 already added all new fields (QS_point, Brier_sample, etc.) |
| Other code in pipeline-core imports EvaluationManager | None | N/A | Grep confirms line 2684 is the only import site |

---

## 10. Deferred Items (not in this change)

1. **Flip `legacy_compatibility` to `False`** — separate change after confirming numeric equivalence between old and new outputs. The old truncation behavior (truncate step-wise to shortest sequence) is a known legacy bug; disabling it is correct but changes metric values.

2. **Remove `get_schema_results()` dataclass path** — once no consumer in the ecosystem calls this method, it can be simplified or removed. Currently still used internally by `to_dataframe()`.

3. **Collapse the three identical WandB log functions** — `generate_wandb_step_wise_log_dict`, `generate_wandb_month_wise_log_dict`, and `generate_wandb_time_series_wise_log_dict` are identical except for the string prefix. They should be a single parameterized function.

4. **Bump views-evaluation version constraint** — pipeline-core's `pyproject.toml` should declare `views-evaluation = "^0.5.0"` to prevent accidental installation of pre-Phase-3 versions.

5. **Add contract test** — a test in views-pipeline-core that asserts the public API surface of views-evaluation exists (`NativeEvaluator`, `EvaluationFrame`, `EvaluationReport`, `METRIC_CATALOG`, etc.) to catch future breaking changes before integration testing.

---

## 11. Expert Review Findings (from 8-perspective code review)

Key findings from the multi-expert review that informed this plan:

- **Feathers (Legacy Code):** The `legacy_compatibility` flag change is the hidden risk. The old wrapper passed `True`; the new NativeEvaluator default is `False`. This plan explicitly preserves `True` to isolate format migration from behavioral change.

- **Hickey (Simplicity):** The dataclass round-trip (`dict → dataclass → asdict() → dict`) is incidental complexity. This plan eliminates it from pipeline-core by passing plain dicts to WandB logging.

- **Beck (Testing):** The absence of any cross-repo integration test is how this incident happened. This plan includes a mandatory regression guard test (Level 5 promoted from optional to required).

- **Nygard (Reliability):** Rollback strategy if Level 3 verification fails: `pip install views_evaluation==0.4.0` in the conda env restores the old code path immediately (but prevents using new metrics).

- **Kleppmann (Data):** Schema evolution risk — the 2x2 dataclasses now have new fields (QS_point, Brier_sample). WandB dashboards may show new metric columns. This is expected and correct, not a bug.
