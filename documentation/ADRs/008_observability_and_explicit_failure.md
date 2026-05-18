# ADR-008: Observability and Explicit Failure

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

This project has a strong "Fail Loud and Proud" culture. Sniffers raise exceptions on contract violations. `PredictionFrame` validates at construction. Custom exceptions (`PipelineException` and subclasses) auto-send WandB alerts on creation.

However, repo-assimilation identified inconsistencies:
- **WandB operations** catch all exceptions and log — a broken WandB connection means no alerts fire, but the pipeline continues. This is silent observability loss.
- **Appwrite uploads** return `OperationResult` objects instead of raising — failures are data, not errors.
- **Reconciliation** logs failed parallel tasks and continues with partial results.

The project has logging infrastructure (`LoggingModule` with YAML config) but no explicit standard for when to log vs raise vs both.

## Decision

Structural failures must be both logged persistently and raised explicitly. The system must never silently degrade.

### Failure Classification

| Category | Action | Example |
|----------|--------|---------|
| **Contract violation** | Raise exception immediately | Sniffer detecting invalid MultiIndex, PredictionFrame with 0 rows |
| **Configuration error** | Raise exception before state mutation | Missing mandatory config key, invalid deployment status |
| **Structural data error** | Raise exception | Empty DataFrame passed to evaluation, type mismatch |
| **Infrastructure failure** | Log at ERROR + raise or propagate | Appwrite upload failure, WandB connection loss |
| **Degraded operation** | Log at WARNING + document scope of degradation | Partial reconciliation, missing shapefile for one region |

### Logging Levels

| Level | When to use |
|-------|-------------|
| `DEBUG` | Internal state tracing (disabled in production) |
| `INFO` | Successful completion of pipeline stages, sniffer passes |
| `WARNING` | Degraded but continuing operation (must document what is degraded) |
| `ERROR` | Failure that affects output correctness (must raise exception) |
| `CRITICAL` | Pipeline cannot continue (must raise exception + alert) |

### Existing Enforcement Points

- `PipelineException.__init__()` — auto-sends WandB alert at exception creation
- `CoreConfigSniffer` — raises on all contract violations, `logger.info` on success
- `CoreDataSniffer` — raises `ValueError`/`NotImplementedError` on violations
- `CorePredictionSniffer` — raises `ValueError` on violations
- `PredictionFrame._validate_input()` — raises on invariant violation

### Known Deviations from This Standard

- **WandB silent failure:** All WandB operations in `WandBModule` catch exceptions and log but do not raise. This means loss of experiment tracking is invisible. Accepted as a pragmatic trade-off (pipeline should complete even if WandB is down) but the loss of observability should itself be observable — a meta-alert or log line at pipeline completion.
- **Appwrite OperationResult pattern:** Returns success/failure as data rather than raising. Acceptable at the integration boundary but callers must check results.
- **Ensemble subprocess errors:** Shell script failures depend on child process error handling, which this library cannot control.

## Rationale

This library produces outputs consumed by policy decision-makers. Silent data corruption is worse than a loud crash. The fail-loud principle ensures that problems are caught at the point of origin, not downstream.

## Consequences

### Positive
- Errors are caught where they originate
- WandB alerts fire on exception creation (automatic)
- Sniffer pattern enforces structural auditing at every pipeline stage

### Negative
- Pipeline stops on first error (no batch error collection)
- WandB dependency for alerting creates a single point of failure for observability

## References

- [ADR-003: Authority of Declarations Over Inference](003_authority_of_declarations_over_inference.md)
- [ADR-009: Boundary Contracts](009_boundary_contracts_and_configuration_validation.md)
- [Logging and Observability Standard](../standards/logging_and_observability_standard.md)
- Custom exceptions: `views_pipeline_core/exceptions/exceptions.py`

---
*End of ADR-008.*
