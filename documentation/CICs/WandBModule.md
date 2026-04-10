
# Class Intent Contract: WandBModule

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-04-08
**Related ADRs:** ADR-008 (Observability)

---

## 1. Purpose

Centralizes all Weights & Biases operations for the pipeline: run initialization,
metrics logging, structured evaluation result publishing, alerts with path redaction,
artifact versioning, and run lifecycle management. It is the single point of contact
between the pipeline and the WandB observability backend.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** compute metrics. It receives pre-computed metric dicts and
  forwards them to WandB.
- This class does **not** decide when to start or stop a run. The caller (typically
  the model manager facade) owns the run lifecycle.
- This class does **not** validate prediction DataFrames, configs, or data.
- This class does **not** persist predictions to disk. That is `PredictionIOManager`'s
  responsibility.
- This class does **not** handle Appwrite or any non-WandB external service.

---

## 3. Responsibilities and Guarantees

- Guarantees that `initialize_run()` creates a WandB run with the specified project,
  entity, config, and job type, and registers custom step-wise, month-wise, and
  time-series-wise metric definitions before returning.
- Guarantees that `finish_run()` is idempotent: safe to call multiple times, no-op
  if no active run.
- Guarantees that `send_alert()` redacts `models_path` from alert text before
  sending, preventing filesystem path leakage. No-op if `notifications_enabled=False`
  or no active WandB run.
- Guarantees that `log_artifact()` raises on upload failure (does not swallow the
  exception).
- Guarantees that `log_metrics()` and `log()` never raise; failures are caught and
  logged as errors.
- Guarantees that `login()` is a static method that can be called before any instance
  exists.

---

## 4. Inputs and Assumptions

- `entity: str` -- WandB entity (team or user) for run organization.
- `notifications_enabled: bool` -- gate for `send_alert()`. Default `False`.
- `models_path: Optional[Path]` -- filesystem path to redact from alert text.
- `initialize_run()` requires:
  - `project: str` -- WandB project name.
  - `config: Dict` -- hyperparameters and pipeline settings.
  - `job_type: str` -- one of `"train"`, `"evaluate"`, `"forecast"`, `"sweep"`.
  - `name: Optional[str]` -- human-readable run name.
- `log_evaluation_results()` requires three dicts keyed by step/month/time-series
  and a `target_identifier: str`.
- Only one run can be active at a time per instance.

---

## 5. Outputs and Side Effects

- **Primary outputs:**
  - `initialize_run()` returns the active `wandb.sdk.wandb_run.Run` object.
  - All other methods return `None`.
- **Side effects:**
  - Creates and finalizes WandB runs on the remote server.
  - Logs metrics, artifacts, and alerts to the WandB dashboard.
  - Uploads artifact files to WandB storage.
  - Copies files to WandB's run directory via `save()`.
  - Defines custom metric axes (`step-wise/step`, `month-wise/month`,
    `time-series-wise/time-series`) on the run.

---

## 6. Failure Modes and Loudness

- `log_metrics()` and `log()` catch all exceptions and log them as errors. They
  never raise. This is intentional: observability failures must not abort a
  training or forecasting run.
- `log_artifact()` catches exceptions, logs them as errors, and **re-raises**.
  Artifact upload failures are considered significant enough to propagate.
- `send_alert()` catches exceptions and logs them as errors. It never raises.
- `initialize_run()` delegates to `wandb.init()` and propagates any exception
  (e.g., invalid credentials, network failure).
- `finish_run()` delegates to `wandb.finish()` and propagates any exception.

---

## 7. Boundaries and Interactions

- **Depends on:**
  - `wandb` SDK (`wandb.init`, `wandb.log`, `wandb.alert`, `wandb.Artifact`,
    `wandb.save`, `wandb.finish`, `wandb.login`, `wandb.define_metric`).
  - `views_pipeline_core.modules.wandb.log_wandb_log_dict` -- helper for
    structured evaluation result logging (called by `log_evaluation_results()`).
- **Does not depend on:**
  - Any model manager, data loader, sniffer, or Appwrite module.
- **Injected into:**
  - `EvaluationStage` and `ForecastingStage` as the `wandb_module` collaborator.
  - Model manager facades for run lifecycle.
- **Trust boundary:** WandB API. Network failures are handled gracefully for
  metrics/alerts but may propagate for artifacts and run lifecycle.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.modules.wandb.wandb import WandBModule

wandb_mod = WandBModule(entity="views-team", notifications_enabled=True)

try:
    run = wandb_mod.initialize_run(
        project="views-forecasting",
        config={"algorithm": "rf", "steps": 36},
        job_type="train",
        name="experiment_001",
    )
    wandb_mod.log_metrics({"train/loss": 0.234, "epoch": 5})
    wandb_mod.log_artifact(
        artifact_path=Path("model.pt"),
        artifact_name="conflict_model_v2",
        artifact_type="model",
    )
finally:
    wandb_mod.finish_run()
```

```python
# Static alert (no instance needed for login)
WandBModule.login()
WandBModule.send_alert(
    title="Forecasting complete",
    text=f"Model saved to {models_path}/output.pt",
    models_path=models_path,
    notifications_enabled=True,
)
```

---

## 9. Examples of Incorrect Usage

- **Calling `log_metrics()` without `initialize_run()`:** The method silently
  no-ops if `_active_run` is `None`. Metrics are lost without warning. Always
  initialize a run first.
- **Calling `log_artifact()` without an active `wandb.run`:** `wandb.run` is
  `None` before `initialize_run()`, causing `AttributeError` on
  `wandb.run.log_artifact()`.
- **Relying on `send_alert()` without checking `notifications_enabled`:** The
  method is a no-op when disabled. Do not assume alerts were sent without
  verifying the flag.

---

## 10. Test Alignment

- **Green tests:** Unit tests with mocked `wandb` module can verify that
  `initialize_run()` calls `wandb.init()` with correct args, that
  `send_alert()` redacts paths, and that `finish_run()` is idempotent.
- **Beige tests:** Integration tests require a valid WandB API key and network
  access.
- **Red tests:** Tests should verify that `log_artifact()` propagates exceptions,
  that `log_metrics()` swallows them, and that `send_alert()` is a no-op when
  `notifications_enabled=False`.

---

## 11. Evolution Notes (Optional)

- `log_evaluation_results()` delegates to a module-level function
  `log_wandb_log_dict()`. This indirection exists for historical reasons and
  may be inlined in a future cleanup.
- Context manager support (`__enter__`/`__exit__`) is referenced in docstrings
  but not implemented in the current code. Adding it would improve run lifecycle
  safety.

---

## End of Contract

This document defines the **intended meaning** of `WandBModule`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
