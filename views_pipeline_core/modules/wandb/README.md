# WandB Module for VIEWS Pipeline Core

## Overview

The `WandBModule` is a centralized utility for managing all **Weights & Biases (WandB)** operations within VIEWS Pipeline Core. It provides a streamlined interface for initializing runs, logging metrics, managing artifacts, sending alerts, and tracking experiments. This module is designed to integrate seamlessly with the VIEWS pipeline, enabling structured and reproducible experiment tracking.

---

## Features

- **Run Management**: Initialize, finalize, and manage WandB runs.
- **Custom Metrics**: Define and log structured metrics for step-wise, month-wise, and time-series evaluations.
- **Artifact Management**: Upload and version models, datasets, and results.
- **Alerts**: Send real-time notifications for important events or errors.
- **Integration**: Works seamlessly with the VIEWS pipeline's configuration and evaluation systems.

---

## Usage

### 1. Initialize a WandB Run

```python
from views_pipeline_core.modules.wandb import WandBModule

# Initialize the module
wandb_module = WandBModule(
    entity="views_pipeline",
    notifications_enabled=True,
    models_path="/path/to/models"
)

# Start a new run
config = {"algorithm": "random_forest", "features": ["f1", "f2"]}
run = wandb_module.initialize_run(
    project="views-forecasting",
    config=config,
    job_type="train",
    name="experiment_001"
)

print(f"Run initialized: {run.name}")
```

### 2. Log Metrics

```python
# Log training metrics
wandb_module.log_metrics({
    "train/loss": 0.234,
    "train/mse": 0.045,
    "epoch": 5
})
```

### 3. Log Evaluation Results

```python
# Log evaluation metrics at multiple aggregation levels
wandb_module.log_evaluation_results(
    step_wise={1: {"mse": 0.01}, 2: {"mse": 0.02}},
    month_wise={"2024-01": {"mae": 0.05}},
    time_series_wise={"ts_001": {"r2": 0.85}},
    conflict_type="sb"
)
```

### 4. Log Artifacts

```python
# Log a model artifact
wandb_module.log_artifact(
    artifact_path="/path/to/model.pt",
    artifact_name="conflict_model_v2.3",
    artifact_type="model",
    description="Random Forest trained on 2020-2024 data",
    metadata={"accuracy": 0.87, "features": 42}
)
```

### 5. Send Alerts

```python
# Send a real-time alert
wandb_module.send_alert(
    title="Training Completed",
    text="Model achieved 0.87 accuracy",
    level=wandb.AlertLevel.INFO,
    models_path="/path/to",
    notifications_enabled=True
)
```

### 6. Finalize the Run

```python
# Finish the run
wandb_module.finish_run()
```

---

## Best Practices

- **Use Namespaces for Metrics**: Organize metrics using namespaces (e.g., `train/loss`, `eval/mse`).
- **Log Artifacts for Versioning**: Use `log_artifact` to track models, datasets, and results.
- **Send Alerts for Critical Events**: Use `send_alert` to notify about important events or errors.
- **Finalize Runs**: Always call `finish_run` to ensure all data is uploaded.
- **Redact Sensitive Paths**: Use the `models_path` argument in `send_alert` to redact sensitive file paths.

---

## Example

```python
from views_pipeline_core.modules.wandb.wandb import WandBModule

# Initialize WandB module
wandb_module = WandBModule(entity="views_pipeline", notifications_enabled=True)

# Start a new run
config = {"algorithm": "random_forest", "features": ["f1", "f2"]}
wandb_module.initialize_run(
    project="views-forecasting",
    config=config,
    job_type="train",
    name="experiment_001"
)

# Log metrics
wandb_module.log_metrics({"train/loss": 0.234, "epoch": 1})

# Log evaluation results
wandb_module.log_evaluation_results(
    step_wise={1: {"mse": 0.01}, 2: {"mse": 0.02}},
    month_wise={"2024-01": {"mae": 0.05}},
    time_series_wise={"ts_001": {"r2": 0.85}},
    conflict_type="sb"
)

# Log an artifact
wandb_module.log_artifact(
    artifact_path="/path/to/model.pt",
    artifact_name="conflict_model_v2.3",
    artifact_type="model",
    description="Random Forest trained on 2020-2024 data"
)

# Send an alert
wandb_module.send_alert(
    title="Training Completed",
    text="Model achieved 0.87 accuracy",
    level=wandb.AlertLevel.INFO
)

# Finalize the run
wandb_module.finish_run()
```