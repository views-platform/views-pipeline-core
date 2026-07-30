# ADR-051: Composition-Based Ensemble Architecture

| ADR Info   | Details                                      |
|------------|----------------------------------------------|
| Subject    | Composition-Based Ensemble Architecture      |
| ADR Number | 051                                          |
| Status     | Accepted                                     |
| Author     | Simon                                        |
| Date       | 2026-05-18                                   |

## Context

`EnsembleManager` inherits from `ForecastingModelManager`, but an ensemble is not a
forecasting model. This creates several structural problems tracked in the risk
register:

- **C-65 (Tier 2): LSP violation.** `EnsembleManager` overrides abstract methods
  with incompatible signatures and different semantics. It never calls
  `_train_model_artifact()`, `_evaluate_model_artifact()`, or
  `_forecast_model_artifact()` as defined by `ForecastingModelManager`; instead it
  replaces the higher-level `_execute_model_*` methods entirely.
- **C-55 (Tier 2): Dropped CoreConfigSniffer.** `EnsembleManager.execute_single_run()`
  replaces the parent's implementation without calling `CoreConfigSniffer.sniff_all()`,
  removing a pre-flight safety check that `ForecastingModelManager` guarantees.
- **DIP violation.** Infrastructure access (WandB, IO, evaluation, reporting) is
  coupled to the inheritance chain rather than injected.

ADR-045 extracted five independently composable stages (`EvaluationStage`,
`PredictionIOManager`, `ReportingStage`, `ForecastingStage`, `TrainingStage`) with
frozen context dataclasses. These provide the escape hatch for a composition-based
ensemble that does not inherit from the model manager hierarchy.

Additionally, three new HydraNet ensembles are planned that require a
PredictionFrame code path (64 posterior samples per prediction). The current
`EnsembleManager` only supports the DataFrame path. Building composition-based
ensembles from scratch avoids forcing PredictionFrame support through the
inheritance chain.

## Decision

Build `DataFrameEnsembleManager` using **composition** (no inheritance), proving
the pattern works before extending to PredictionFrame support.

### Design Principles

1. **Composition over inheritance.** `DataFrameEnsembleManager` does not extend
   `ForecastingModelManager` or `ModelManager`. It constructs and composes ADR-045
   stages directly.

2. **Immutable context threading.** A frozen `EnsembleContext` dataclass (extending
   `BaseStageContext`) is built once in `execute_single_run()` and passed to every
   method. No method reads mutable `self` state for config, args, or run-type
   during execution.

3. **WET-before-DRY.** Business logic (aggregation, reconciliation, subprocess
   delegation) is copied from `EnsembleManager`, not shared via abstraction. The
   two implementations coexist. Shared abstractions are deferred until both
   composition-based managers are proven.

4. **Strangler Fig coexistence.** The existing `EnsembleManager` is not modified.
   Both classes are exported from `managers/ensemble/__init__.py`. Existing
   ensembles continue to use `EnsembleManager`.

5. **CoreConfigSniffer integration (C-55 fix).** `execute_single_run()` calls
   `CoreConfigSniffer(configs, partition_dict).sniff_all(run_type)` before WandB
   login, matching `ForecastingModelManager` behavior.

### Architecture

```
DataFrameEnsembleManager (composition, no inheritance)
    |
    |-- EnsembleContext (frozen dataclass, extends BaseStageContext)
    |   |-- configs, model_path, run_type (from BaseStageContext)
    |   |-- project, eval_type, args, models, aggregation, targets
    |   |-- reconciliation, reconcile_with, use_weights, weights
    |   |-- timestamp, deployment_status, prediction_format, partition_dict
    |
    |-- Composed ADR-045 stages:
    |   |-- EvaluationStage  (metric computation)
    |   |-- PredictionIOManager (persistence)
    |   |-- ReportingStage (HTML reports)
    |
    |-- Infrastructure (constructed in __init__, not inherited):
    |   |-- WandBModule, LoggingModule, ConfigurationManager
    |
    |-- Business logic (WET copies from EnsembleManager):
        |-- _train_ensemble, _evaluate_ensemble, _forecast_ensemble
        |-- _get_aggregated_df (delegates to AggregationManager)
        |-- _apply_reconciliation (delegates to ReconciliationModule)
        |-- _execute_shell_script (subprocess, timeout=7200)
```

### Migration Trajectory

```
Phase 1 (this ADR): DataFrameEnsembleManager
   - Proves composition works for DataFrame path
   - 36 characterization tests verify behavior
   - No existing ensembles migrated

Phase 2 (future): PredictionFrameEnsembleManager
   - Composition-based manager for sample-based predictions
   - Addresses C-66 (OOM aggregation) with PredictionFrame-native aggregation
   - Goes into production with HydraNet ensembles

Phase 3 (optional): Legacy migration
   - Existing ensembles migrate from EnsembleManager to DataFrameEnsembleManager
   - Decision deferred until Phase 2 is proven
```

## Consequences

### Positive

- **C-55 fixed.** `CoreConfigSniffer` runs before execution, matching the
  guarantee `ForecastingModelManager` provides for single models.
- **C-65 resolved (for new code).** No LSP violation because there is no
  inheritance. The class stands on its own.
- **Testable in isolation.** All collaborators are injected at construction.
  Tests mock 7 dependencies with no inheritance chain to navigate.
- **Clear composition surface.** ADR-045 stages are used via their public
  interfaces only. The manager does not reach into stage internals.
- **PredictionFrame path unblocked.** The composition pattern can be extended
  to PredictionFrame without threading it through the inheritance chain.
- **Immutable execution context.** `EnsembleContext` is frozen; no method can
  mutate shared state during a run. Eliminates a class of temporal coupling bugs.

### Negative

- **Code duplication.** Business logic is duplicated between `EnsembleManager` and
  `DataFrameEnsembleManager`. This is intentional (WET-before-DRY) but increases
  maintenance cost until one is retired.
- **Two ensemble managers to maintain.** Until legacy migration (Phase 3) is
  decided, both classes exist. Changes to ensemble behavior must be considered
  in both.
- **Static method dependency.** `_evaluate_model_artifact` calls
  `ForecastingModelManager._resolve_evaluation_sequence_number()`. This is a
  static method (no instance coupling) but creates a symbolic dependency on the
  class being replaced.

### Neutral

- **No downstream changes required.** Existing ensemble configs, sub-model scripts,
  and pipeline invocations are unaffected.
- **Export surface expanded.** `DataFrameEnsembleManager` is exported alongside
  `EnsembleManager` from `managers/ensemble/__init__.py`.

## Risk Register Cross-References

| Entry | Relationship |
|-------|-------------|
| C-55 | Fixed: CoreConfigSniffer now runs in execute_single_run() |
| C-65 | Resolved for new code: no inheritance, no LSP violation |
| C-66 | Unblocked: composition pattern extends to PredictionFrame (Phase 2) |
| C-68 | Preserved: reconciliation None-return behavior copied as-is |
| D-03 | Resolved: Strangler Fig chosen over Clean Break |
| D-13 | Resolved: DataFrame-first proven safer than PredictionFrame-first |
| D-14 | Resolved: minimal composition (no shared facade) for Phase 1 |

## Implementation Notes

- **Config loading:** Uses the same `importlib` pattern as `ModelManager.__init__()`.
  Config scripts are loaded by name from `ensemble_path.get_scripts()`.
- **Evaluation delegation:** Builds `EvaluationContext` with
  `prediction_format="dataframe"`, `data_loader=None`, and
  `prepare_actuals_df=lambda df: df`, then calls
  `EvaluationStage.evaluate(preds, ctx, ensemble=True)`.
- **wandb.AlertLevel.WARN:** The original `EnsembleManager` used
  `wandb.AlertLevel.WARNING`, which does not exist in the wandb enum.
  `DataFrameEnsembleManager` uses the correct `wandb.AlertLevel.WARN`.
