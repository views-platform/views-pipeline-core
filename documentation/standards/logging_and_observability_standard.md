# Logging & Observability Standard

**Status:** Active
**Governing ADRs:** ADR-003 (Authority of Declarations), ADR-005 (Testing), ADR-008 (Observability and Explicit Failure)

---

## 1. Purpose

This document defines operational standards for:

- Logging behavior
- Log levels
- Error propagation patterns
- Alerting and observability expectations

This standard operationalizes:

> Structural failures must be raised explicitly and logged persistently. (ADR-008)

It does not redefine architectural principles.

---

## 2. Core Principles

### 2.1 Fail Loud and Persist

- Structural failures must:
  - be logged at `ERROR` or higher
  - be raised as exceptions
- Logging is not a substitute for raising.
- Raising is not a substitute for logging.

Silent degradation is prohibited.

In this project, `PipelineException` and its subclasses (`ModelTrainingException`, `ModelEvaluationException`, `ModelForecastingException`, `ConfigurationException`, `DataFetchException`, `ValidationException`) auto-send WandB alerts on creation via `PipelineException.__init__()`.

---

### 2.2 Logs Must Support Understanding

Logs must:
- provide sufficient context to reconstruct state
- include relevant identifiers (model_name, run_type, timestamp, level)
- avoid ambiguity

Logs must not:
- rely on implicit assumptions
- require tribal knowledge to interpret

---

### 2.3 Logs Must Not Leak Sensitive Data

- WandB API keys must never be logged.
- Appwrite credentials must never be logged.
- `PipelineException.send_alert()` redacts path information from WandB alerts.

---

## 3. Log Levels (Normative Definitions)

### DEBUG
- Development diagnostics.
- Detailed internal state (tensor shapes, intermediate values).
- Must not be required to understand production failures.

### INFO
- High-level lifecycle events.
- Sniffer pass messages: `CoreConfigSniffer`, `CoreDataSniffer`, `CorePredictionSniffer` all `logger.info` on successful validation.
- Pipeline stage transitions (data fetch, training start/finish, evaluation, forecasting).
- Model identifiers and configuration summaries.

### WARNING
- Unexpected but recoverable conditions.
- Degraded behavior that does not violate invariants.
- Example: partial reconciliation (some countries failed but pipeline continues).
- Must not mask structural errors.

Warnings must not be used to hide invariant violations.

### ERROR
- Structural failure within a component.
- Operation failed and cannot proceed correctly.
- Must be raised and logged.
- All sniffer failures raise at this level.

### CRITICAL
- System-wide failure.
- Corruption, irrecoverable state, or orchestration breakdown.
- `PipelineException` subclasses trigger WandB alerts at CRITICAL level.
- Immediate attention required.

---

## 4. Error Propagation Pattern

Structural errors must follow this minimal pattern:

1. Construct a clear, descriptive error message.
2. Log the error (`ERROR` or `CRITICAL`).
3. Raise the appropriate exception with the same message.

Example:

```python
err_msg = f"Run type '{run_type}' not recognized; expected one of {SUPPORTED_RUN_TYPES}."
logger.error(err_msg)
raise PipelineException(err_msg, wandb_module=self._wandb_module)
```

For sniffer pattern classes, the raise itself is sufficient — sniffers are not responsible for logging (the caller logs).

---

## 5. Logging Scope Expectations

### 5.1 Required Logging

The following must be logged:

* Pipeline stage transitions (`ForecastingModelManager._execute_model_tasks()`)
* Model training start/finish
* Data loading and sniffer validation outcomes
* Configuration summaries (logged to WandB)
* All structural failures (via exception + WandB alert)
* Prediction persistence (disk path, store upload status)

### 5.2 Optional Logging

* Intermediate tensor shapes (DEBUG)
* Per-step evaluation metrics during rolling-origin (DEBUG)
* Appwrite upload/download progress (DEBUG)
* Detailed internal diagnostics

---

## 6. Log Structure and Context

Log entries should include:

* Timestamp
* Level
* Module or component name
* Relevant identifiers (model_name, run_type, etc.)

This project uses `LoggingModule` with YAML-configured logging (loaded from model's logging directory). Structured logging (JSON format) is recommended where possible.

---

## 7. Alerting

Alerting is built on `PipelineException.__init__()` which auto-sends WandB alerts.

At minimum:

* `PipelineException` and subclasses send WandB alerts on creation.
* `ERROR` and `CRITICAL` logs must be alertable.
* Alert routing is via WandB (configured per WandB project).

**Known deviation:** WandB operations themselves catch all exceptions and log — if WandB is down, alerting silently fails. This is accepted pragmatically but the loss of observability should itself be logged locally.

---

## 8. Testing Requirements

Logging behavior must be testable where meaningful.

Tests should verify:

* Errors are both logged and raised (sniffer tests verify this).
* Log level separation works as expected.
* WandB alerts trigger on configured severity thresholds.

Logging tests: `tests/test_modules/test_logging.py` (6 tests — coverage gap noted).

---

## 9. Anti-Patterns (Prohibited)

* Swallowing exceptions without logging
* Logging and continuing after invariant violation
* Downgrading errors to warnings to "keep things running"
* Using `print()` for structural diagnostics (known deviation in `ReconciliationModule`)
* Logging entire objects without context

---

## 10. Evolution

This document may evolve independently of ADRs.

If logging semantics change in a way that affects system meaning,
ADR-008 must be revisited.
