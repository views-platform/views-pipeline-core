# ADR-050: WandB as Cross-Cutting Observability Infrastructure

**Status:** Accepted
**Date:** 2026-04-08
**Deciders:** Project maintainers

---

## Context

Weights & Biases (WandB) is the pipeline's experiment tracking and alerting
platform. Every model training run, evaluation, and forecast is tracked as a WandB
"run" with associated metrics, artifacts, and alerts. WandB integration touches
every stage of the pipeline: the facade manages lifecycle, stages log metrics and
send alerts, and exception classes auto-alert on failure.

This cross-cutting nature was never formally documented. The integration grew
organically: lifecycle management in the facade, metric logging in evaluation,
artifact upload in reporting, alert routing in exceptions. Without an ADR,
contributors cannot distinguish between intentional architectural decisions (e.g.,
"alerts are best-effort") and accidental patterns (e.g., "duplicate alert sending").

Six components depend on `WandBModule`: `ForecastingModelManager` (lifecycle),
`EvaluationStage` (metrics), `ForecastingStage` (alerts), `TrainingStage` (alerts),
`ReportingStage` (artifacts + alerts), and `PredictionIOManager` (alerts). The
`PipelineException` hierarchy also auto-alerts via injected WandBModule references.
This is the most widely-depended-on module in the pipeline after `ModelPathManager`.

---

## Decision

### 1. Architectural role: cross-cutting utility

WandB is classified as a **cross-cutting concern** (in the Aspect-Oriented
Programming sense) rather than a bounded context. It is not encapsulated in a single
stage because its responsibilities span multiple pipeline layers:

| Layer | WandB responsibility | Owner |
|-------|---------------------|-------|
| Facade (ModelManager) | Run lifecycle: `initialize_run()` / `finish_run()` | Facade exclusively |
| Stages | Metric logging, alert sending | Each stage |
| Exceptions | Auto-alerting on pipeline failures | Exception `__init__` |
| Configuration | Hyperparameter capture at run init | Facade passes config dict |

No single component can encapsulate all four responsibilities without violating
separation of concerns. The documented architecture is: **facade owns lifecycle,
stages own observability, exceptions own error alerting**.

### 2. Lifecycle management: facade-only

Only `ForecastingModelManager` (the facade) calls `initialize_run()` and
`finish_run()`. Stages never manage lifecycle. This is enforced by convention, not by
type system.

The lifecycle follows this pattern in all 7 execution methods:

```python
with self._wandb_module.initialize_run(
    project=project, config=self.configs, job_type=job_type
):
    try:
        # ... stage work ...
    except Exception as e:
        raise PipelineException(..., wandb_module=self._wandb_module)
    finally:
        self._wandb_module.finish_run()
```

The `with` statement (context manager) and explicit `finally` block provide
redundant cleanup. This is intentional: `wandb.Run.__exit__()` handles normal exit,
`finally` handles exception paths where the context manager might not reach cleanup.

### 3. Failure propagation policy

WandB operations follow a two-tier failure policy:

| Operation | Failure behaviour | Rationale |
|-----------|-------------------|-----------|
| `initialize_run()` | **Raises** (fatal) | Cannot track experiment without a run |
| `finish_run()` | **Raises** (fatal) | Orphaned runs leak resources and corrupt dashboards |
| `log_metrics()` | **Swallows** (logs error) | Missing metrics don't invalidate results |
| `log()` | **Swallows** (logs error) | Best-effort observability |
| `send_alert()` | **Swallows** (logs error) | Missing notification doesn't affect correctness |
| `log_artifact()` | **Raises** (re-raises after logging) | Artifact upload failure is significant |

The principle: **lifecycle failures are fatal; observability failures are
best-effort**. A WandB outage during `initialize_run()` halts the pipeline (by
design — we cannot run experiments we cannot track). A WandB outage during metric
logging lets the experiment continue (metrics can be reconstructed from local data).

This is the sole cross-cutting exception to ADR-008's "Fail Loud and Proud" policy.
Observability operations silently degrade because the alternative (crashing a 6-hour
training run because a metric log failed) is worse.

### 4. Alert routing and path redaction

`send_alert()` redacts filesystem paths before sending to WandB:

```python
text = str(text).replace(str(models_path), "[REDACTED]")
```

This prevents leaking internal directory structures to WandB dashboards and Slack
integrations. The `models_path` typically contains user home directories and
institutional paths that should not appear in external-facing alerts.

The `wandb_notifications` boolean flag controls whether alerts are actually
dispatched. When `False`, `send_alert()` is a no-op. This flag is propagated from
the facade to all stages at construction time. Its value is determined by
command-line arguments (via `ForecastingModelArgs`).

### 5. Auto-alerting in exceptions

All `PipelineException` subclasses accept an optional `wandb_module` parameter in
their constructor. When provided, the exception sends a WandB alert at
`ERROR` level before propagating:

```python
class PipelineException(Exception):
    def __init__(self, message, wandb_module=None, alert_level=ERROR):
        super().__init__(message)
        if wandb_module:
            wandb_module.send_alert(title=self.__class__.__name__, text=message, ...)
```

This couples exception semantics to observability but ensures that every pipeline
failure is visible in WandB regardless of whether the calling code remembers to
send an alert. The coupling is intentional and load-bearing.

### 6. Metric axis registration

`initialize_run()` registers custom WandB metric axes immediately after
`wandb.init()`:

- `step-wise/step` — axis for lead-time step metrics
- `month-wise/month` — axis for calendar month metrics
- `time-series-wise/time-series` — axis for rolling-origin sequence metrics

These axes must be defined before any `log_metrics()` call that references them.
Since only the facade calls `initialize_run()`, axis registration happens exactly
once per run, before any stage begins work.

### 7. Dependency injection pattern

`WandBModule` is injected into stages as a constructor parameter alongside the
`wandb_notifications` flag:

```python
self._evaluation_stage = EvaluationStage(
    wandb_module=self._wandb_module,
    io_manager=self._io,
    wandb_notifications=self._wandb_notifications,
)
```

Stages receive a concrete `WandBModule` reference, not a protocol or interface
(C-48). This is a known DIP violation tracked in the risk register. Introducing a
`WandBProtocol` would allow testing stages without mocking WandB internals, but the
current concrete injection is sufficient for the existing test suite (which mocks at
`sys.modules` level).

---

## Consequences

### Positive

- The cross-cutting nature of WandB is explicitly acknowledged rather than treated as
  accidental complexity.
- The failure propagation policy is documented: lifecycle failures halt, observability
  failures degrade gracefully.
- Path redaction prevents security-sensitive directory structures from appearing in
  dashboards.
- Auto-alerting in exceptions ensures no pipeline failure goes unnoticed in WandB,
  regardless of calling code quality.
- The `wandb_notifications` flag provides a clean opt-out for local development and
  testing.

### Negative

- Cross-cutting integration means `WandBModule` changes affect 6+ components.
  Interface changes have high blast radius.
- The 7-method lifecycle duplication in the facade (D-06) is boilerplate that cannot
  be eliminated without a Template Method or similar abstraction.
- Concrete injection (not protocol-based) limits testability and violates DIP (C-48).
- The `send_alert()` silent failure policy means persistent WandB outages are
  detectable only through log monitoring, not through pipeline failures.

---

## Rationale

WandB is cross-cutting by nature, not by accident. Experiment tracking requires
awareness of lifecycle (when a run starts and ends), metrics (what a stage
produces), artifacts (what files are generated), and errors (what went wrong). No
single stage has all this information. The facade has lifecycle context; stages have
metric context; exceptions have error context. Centralizing all WandB interaction in
one component would require that component to know about training, evaluation,
forecasting, and reporting — recreating the god-class problem that ADR-045 solved.

The failure policy follows a general principle: **infrastructure that supports the
pipeline should not be able to crash the pipeline**. WandB is observability
infrastructure. If WandB is down, the pipeline still produces correct predictions.
The one exception (lifecycle failures) exists because tracking experiments we cannot
observe is operationally meaningless — better to fail fast and retry when WandB
recovers.

The auto-alerting pattern in exceptions is a pragmatic choice. The alternative
(requiring every `except` block to manually send alerts) was tried and resulted in
inconsistent alerting. By embedding alert logic in the exception constructor, we
guarantee coverage without relying on developer discipline at every call site.

---

## References

- **ADR-008:** Observability and Explicit Failure
- **ADR-045:** Pipeline Stage Architecture (stages as WandB consumers)
- **C-48:** Concrete dependencies where abstractions needed
- **D-06:** WandB lifecycle template extraction timing
- **CIC:** WandBModule (`documentation/CICs/WandBModule.md`)
