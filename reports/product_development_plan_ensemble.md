# Product Development Plan: Ensemble Architecture Refactoring

**Date:** 2026-03-17
**Branch:** `feature/samples_for_fao`
**Prerequisite:** Read `reports/rd_roadmap_ensemble.md` for architectural context.

---

## Task List (ordered by dependency and safety)

### Task 1: Extract WandB execution context manager

**Priority:** P0 (unblocks nothing, but highest safety / highest DRY impact)
**SOLID principle:** DRY
**Effort:** Small

**Current state (ensemble.py:186-276):**
Three identical blocks:
```python
try:
    with self._wandb_module.initialize_run(...) as run:
        # ... stage-specific work ...
        self._wandb_module.send_alert(title=..., text=...)
except Exception:
    raise PipelineException(f"... failed: {traceback.format_exc()}", ...)
```

**Target:**
```python
# New: views_pipeline_core/managers/ensemble/execution_context.py
class EnsembleExecutionContext:
    def __init__(self, wandb_module, job_type, configs, model_path):
        ...
    def __enter__(self): ...  # init WandB run
    def __exit__(self, exc_type, exc_val, exc_tb): ...  # alert + PipelineException wrap
```

**Files to modify:**
- Create: `views_pipeline_core/managers/ensemble/execution_context.py`
- Modify: `views_pipeline_core/managers/ensemble/ensemble.py` (lines 186-276)

**Test strategy:**
- Before: `conda run -n views_pipeline pytest tests/test_managers/test_ensemble_manager.py -q`
- After: same tests pass; add 1 test for ExecutionContext error wrapping

**Verification:** `grep -c "PipelineException" ensemble.py` should decrease from 3 to 0 (moved to context manager)

---

### Task 2: Create `MemberPredictionLoader` protocol

**Priority:** P1 (unblocks Task 5)
**SOLID principle:** SRP, DIP, OCP
**Effort:** Medium

**Current state (ensemble.py:514-592):**
`_load_or_generate_prediction()` contains 78 lines mixing:
- Prediction store loading (`pd.DataFrame.forecasts.read_store()`)
- Local file discovery and loading (`read_dataframe()`)
- Subprocess fallback (`_execute_shell_script()`)

**Target:**
```python
# New: views_pipeline_core/managers/ensemble/prediction_loader.py

class MemberPredictionLoader(Protocol):
    def load(self, model_name: str, run_type: str, timestamp: str,
             sequence_number: Optional[int] = None) -> pd.DataFrame: ...

class LocalFilePredictionLoader:
    """Loads from models/{name}/data/generated/*.parquet"""
    def __init__(self, fallback_generator: Optional[Callable] = None): ...
    def load(self, ...) -> pd.DataFrame: ...

class PredictionStorePredictionLoader:
    """Loads from views-forecasts prediction store"""
    def __init__(self, pred_store_name: str): ...
    def load(self, ...) -> pd.DataFrame: ...
```

**Files to modify:**
- Create: `views_pipeline_core/managers/ensemble/prediction_loader.py`
- Modify: `views_pipeline_core/managers/ensemble/ensemble.py`
  - Constructor: accept `prediction_loader: MemberPredictionLoader`
  - Remove `_load_or_generate_prediction()` method
  - `_evaluate_model_artifact()` and `_forecast_model_artifact()` call `self._prediction_loader.load()`

**Test strategy:**
- Add unit tests for `LocalFilePredictionLoader` with tmp_path fixtures
- Add unit tests for `PredictionStorePredictionLoader` with mocked store
- Existing ensemble manager tests updated to inject mock loader

**Scope boundary:** Do NOT move subprocess generation logic into the loader. Keep `_execute_shell_script()` in EnsembleManager; the loader can accept a `fallback_generator` callable for the generate-then-load path.

---

### Task 3: Create `ReconciliationStrategy` protocol

**Priority:** P1 (independent of Task 2)
**SOLID principle:** OCP, DIP, SRP
**Effort:** Medium

**Current state (ensemble.py:594-741):**
148 lines of reconciliation logic embedded in EnsembleManager:
- `_apply_reconciliation()` — string-matching dispatch
- `__reconcile_pg_with_c()` — PG↔C reconciliation orchestration
- `_load_c_dataset()` — dataset loading for reconciliation

**Target:**
```python
# New: views_pipeline_core/modules/reconciliation/strategy.py

class ReconciliationStrategy(Protocol):
    def reconcile(self, df_prediction: pd.DataFrame,
                  configs: dict) -> pd.DataFrame: ...

class PgmCmPointReconciliation:
    """Reconciles PGM predictions against CM predictions."""
    def __init__(self, prediction_loader: MemberPredictionLoader,
                 wandb_notifications: bool = True): ...
    def reconcile(self, df_prediction, configs) -> pd.DataFrame: ...

class NullReconciliation:
    """No-op: returns predictions unchanged."""
    def reconcile(self, df_prediction, configs) -> pd.DataFrame:
        return df_prediction

def create_reconciliation_strategy(
    reconciliation_type: Optional[str], **kwargs
) -> ReconciliationStrategy:
    if reconciliation_type == "pgm_cm_point":
        return PgmCmPointReconciliation(**kwargs)
    return NullReconciliation()
```

**Files to modify:**
- Create: `views_pipeline_core/modules/reconciliation/strategy.py`
- Modify: `views_pipeline_core/managers/ensemble/ensemble.py`
  - Constructor: accept `reconciliation_strategy: ReconciliationStrategy`
  - Remove `_apply_reconciliation()`, `__reconcile_pg_with_c()`, `_load_c_dataset()` (148 lines)
  - `_forecast_ensemble()` becomes: `df = self._reconciliation_strategy.reconcile(df, self.configs)`

**Test strategy:**
- Unit tests for `PgmCmPointReconciliation` with synthetic data
- Unit test for `NullReconciliation` (identity)
- Unit test for factory function
- Existing ensemble reconciliation tests adapted to use strategy directly

**Scope boundary:** Do NOT modify `ReconciliationModule` internals. The strategy wraps it; the core torch-based algorithm stays unchanged.

---

### Task 4: Inject AggregationManager via factory

**Priority:** P2 (independent)
**SOLID principle:** DIP
**Effort:** Small

**Current state (ensemble.py:743-812):**
`_get_aggregated_df()` directly instantiates `AggregationManager`:
```python
manager = AggregationManager(index_cols=index_cols, target_cols=target_cols)
```

**Target:**
- EnsembleManager constructor accepts optional `aggregation_factory: Callable`
- Default factory creates `AggregationManager` with current behavior
- Future factory could create PredictionFrame-native aggregator

**Files to modify:**
- Modify: `views_pipeline_core/managers/ensemble/ensemble.py`
  - Constructor: add `aggregation_factory` parameter with default
  - `_get_aggregated_df()`: use `self._aggregation_factory(...)` instead of direct instantiation

**Test strategy:**
- Existing tests pass (default factory preserves behavior)
- Add 1 test verifying custom factory is called

**Scope boundary:** Do NOT create the `AggregationStrategy` protocol yet. This task only inverts the dependency. The protocol is Phase 4 in the roadmap.

---

### Task 5: Extend AggregationManager to accept callable methods

**Priority:** P3 (nice-to-have, OCP improvement)
**SOLID principle:** OCP
**Effort:** Small

**Current state (aggregator.py:199):**
```python
valid_methods = {"mean", "median", "min", "max"}
```

**Target:**
```python
def aggregate(self, *, method: Union[str, Callable] = None, ...) -> pl.DataFrame:
    if callable(method):
        return self._apply_custom_aggregation(method)
    # ... existing string-based dispatch
```

**Files to modify:**
- Modify: `views_pipeline_core/modules/ensemble_aggregator/aggregator.py` (lines 138-219)

**Test strategy:**
- Existing tests pass (string methods unchanged)
- Add test: custom lambda aggregation function

---

## Dependency Graph

```
Task 1 (WandB context)     — independent
Task 2 (PredictionLoader)  — independent
Task 3 (ReconciliationStrategy) — depends on Task 2 (uses loader for C-dataset loading)
Task 4 (Aggregation factory) — independent
Task 5 (Callable methods)   — independent
```

**Recommended execution order:** 1 → 2 → 3 → 4 → 5

---

## Verification (after all tasks)

```bash
# Full test suite
conda run -n views_pipeline pytest tests/ --tb=short -q

# Lint
conda run -n views_pipeline ruff check views_pipeline_core/managers/ensemble/ views_pipeline_core/modules/reconciliation/ views_pipeline_core/modules/ensemble_aggregator/

# Line count reduction target
wc -l views_pipeline_core/managers/ensemble/ensemble.py
# Target: < 500 lines (from 827)
```

---

## Scope Boundaries

**DO:**
- Extract collaborators behind protocols
- Inject dependencies via constructors
- Preserve exact runtime behavior

**DO NOT:**
- Change the parquet on-disk format
- Modify AggregationManager's internal Polars logic
- Modify ReconciliationModule's torch-based algorithm
- Change the subprocess delegation pattern
- Fix the LSP violation (EnsembleManager ↔ ForecastingModelManager inheritance) — that's a separate, higher-risk effort
- Add PredictionFrame-native aggregation — that's Phase 5 in the roadmap, deferred until demand exists
