# ADR-045: Pipeline Stage Architecture

**Status:** Accepted
**Date:** 2026-04-06
**Deciders:** Project maintainers

---

## Context

`ForecastingModelManager` is 3,176 LOC and concentrates orchestration, evaluation, persistence, format dispatch, WandB lifecycle, and report generation. Risk register entry C-01 (Tier 2) tracks this as a "god class."

Investigation (2026-04-06) determined that the god class is a **symptom** of five structural defects, not the root problem. Fixing the class without addressing the root causes would produce another god class elsewhere.

### Root Causes

1. **Inverted dependencies.** `ModelPathManager` lives in `managers/` but is a data structure for path resolution. Lower layers (`data/handlers.py`, `modules/dataloaders/`) import upward from `managers/`, creating circular coupling. This forces validators, data handlers, and dataloaders to depend on the manager layer.

2. **Missing pipeline abstraction.** There is no `Stage` interface or `Pipeline` container. The seven `_execute_*` methods are private methods on the god class — not independently callable, testable, or composable units. The class IS the pipeline.

3. **Missing stage context.** Each pipeline stage pulls what it needs from `self` rather than receiving an explicit context object. Subclasses implementing abstract methods also reach into `self._model_path`, `self._data_loader`, `self.configs` directly. There is no declared contract for what a stage needs.

4. **Lifecycle entanglement.** Every `_execute_*` method interleaves WandB run lifecycle management (`initialize_run`/`finish_run`/`send_alert`) with business logic. Stages cannot be tested, reused, or composed without WandB infrastructure.

5. **Push-based abstract contracts.** Abstract methods (`_train_model_artifact`, `_evaluate_model_artifact`, etc.) receive no context parameters. Subclasses must reach into the god class's internals to obtain data, paths, and configuration. This tightly couples every downstream model repo to the internal structure of ForecastingModelManager.

## Decision

Decompose ForecastingModelManager using a **Stage + Context** pattern:

1. **Each pipeline stage becomes a class** that receives an explicit, immutable context object.
2. **Stages do not inherit from ForecastingModelManager** and do not access `self` on the manager.
3. **ForecastingModelManager becomes a thin façade** that constructs contexts and delegates to stages.
4. **WandB lifecycle is a cross-cutting concern** handled by the façade or a decorator, not by each stage.
5. **Abstract methods evolve** to accept context objects as an optional parameter (backward-compatible default).

### Target Architecture

```
ForecastingModelManager (thin façade, ~500 LOC)
├── Constructs ExecutionContext from init state
├── Routes to stages based on args flags
└── Manages WandB run lifecycle

Stages (independently testable, ~150-250 LOC each)
├── EvaluationStage(wandb_module, io_manager)
│   └── evaluate(predictions, EvaluationContext) → metrics published
├── TrainingStage(wandb_module)           [future]
├── ForecastingStage(wandb_module, io_manager) [future]
├── ReportingStage(wandb_module)          [future]
└── DataFetchingStage(data_loader)        [future]

Context Objects (frozen dataclasses, passed explicitly)
├── EvaluationContext(configs, model_path, prediction_format, ...)
├── TrainingContext(configs, model_path, ...)  [future]
└── ForecastingContext(configs, model_path, ...) [future]
```

### Migration Strategy

Staged Strangler Fig extraction, one stage per PR:

1. **E2 (this PR):** Extract `EvaluationStage` with `EvaluationContext`. Delegate from `_evaluate_prediction_dataframe()`.
2. **E3 (future):** Extract `ReportingStage` from `_execute_forecast_reporting()` and `_execute_evaluation_reporting()`.
3. **E4 (future):** Extract `ForecastingStage` from `_execute_model_forecasting()`.
4. **E5 (future):** Extract `TrainingStage` from `_execute_model_training()`.
5. **E6 (future):** Relocate `ModelPathManager` from `managers/` to `data/` (cross-repo impact — separate ADR).

Each extraction follows the same pattern: create frozen context → extract stage class → delegate from façade → verify existing tests pass.

### Deferred Decisions

- **ModelPathManager relocation** — impacts all downstream model repos. Requires separate ADR and coordinated migration.
- **Abstract method context parameters** — evolving `_train_model_artifact(self)` to `_train_model_artifact(self, context=None)` requires downstream model repo updates. Deferred until at least 2 stages are extracted.
- **Pipeline composition** — a `Pipeline` container that composes stages is valuable but premature before 3+ stages exist.

## Implementation Notes

### Context objects are frozen dataclasses

```python
@dataclass(frozen=True)
class EvaluationContext:
    configs: Dict
    model_path: Any  # ModelPathManager
    prediction_format: str
    partition_dict: Dict
    run_type: str
    data_loader: Any
    prepare_actuals_df: Callable
```

Frozen ensures immutability — stages cannot mutate the context. `Any` typing for `model_path` and `data_loader` avoids importing from the manager layer (would recreate root cause #1).

### Stages are injected eagerly (following E1 pattern)

```python
# In ForecastingModelManager.__init__()
self._evaluation_stage = EvaluationStage(
    wandb_module=self._wandb_module,
    io_manager=self._io,
)
```

This follows the `PredictionIOManager` (E1) extraction pattern established in commit `017c85a`.

### Thin delegate preserves test surface

```python
def _evaluate_prediction_dataframe(self, df_predictions, eval_type, ensemble=False):
    context = EvaluationContext(...)
    self._evaluation_stage.evaluate(df_predictions, context, ensemble=ensemble)
```

The existing 53 evaluation-path tests continue to mock at `_evaluate_prediction_dataframe` and never know the stage exists.

## Consequences

- ForecastingModelManager shrinks by ~225 LOC per extraction (evaluation is first)
- Each stage is independently unit-testable with a mock context
- Future extractions follow the documented pattern without re-investigation
- Downstream model repos require no changes (delegate preserves signatures)
- Root causes #2 (missing pipeline abstraction) and #3 (missing stage context) are directly addressed
- Root causes #1, #4, and #5 are documented for future work
