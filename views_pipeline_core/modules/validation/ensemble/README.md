# VIEWS Pipeline Core: Ensemble Model Validation Module

This module provides comprehensive validation utilities for ensemble models within the VIEWS Pipeline Core. It ensures that all constituent models in an ensemble meet strict temporal, deployment, and partition compatibility requirements, safeguarding the integrity and reproducibility of ensemble forecasts and evaluations.

---

## Contents

- [Overview](#overview)
- [Functions](#functions)
    - [`validate_model_conditions`](#validate_model_conditions)
    - [`validate_ensemble_model_deployment_status`](#validate_ensemble_model_deployment_status)
    - [`validate_partition_config`](#validate_partition_config)
    - [`validate_ensemble_model`](#validate_ensemble_model)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [References](#references)

---

## Overview

The ensemble validation module is designed to:

- **Enforce temporal freshness**: Ensures models and data are up-to-date for the current training cycle and month.
- **Check deployment status**: Prevents use of deprecated models or ensembles, and ensures deployment status compatibility.
- **Validate partition alignment**: Confirms that train/test splits are consistent between ensembles and their constituent models, preventing data leakage.
- **Automate ensemble validation**: Provides a single entry point to validate all constituent models before ensemble operations.
- **Log detailed errors**: Uses clear, actionable logging for rapid debugging and operational transparency.

---

## Functions

### `validate_model_conditions`

```python
def validate_model_conditions(path_generated, run_type):
        """
        Validates that a model was trained in the current training cycle (after July) 
        and that both generated features and raw data were fetched in the current month. 
        This ensures data freshness and compliance with VIEWS operational standards.

        Args:
                path_generated (Path): Path to the model's data_generated directory containing log files.
                run_type (str): Type of run to validate: 'calibration', 'forecasting', or 'validation'.

        Returns:
                bool: True if all temporal conditions are met, False otherwise.

        Raises:
                Does not raise, but logs errors and returns False on failure.
        """
```

**Example:**

```python
from pathlib import Path

is_valid = validate_model_conditions(Path('models/conflict_model/data/generated'), 'forecasting')
if not is_valid:
        print("Model does not meet temporal requirements")
```

**Notes:**

- Training cycle is July–June; models must be trained after July of the previous year.
- Data and raw data must be generated/fetched in the current month.
- Logs detailed error messages before returning `False`.

---

### `validate_ensemble_model_deployment_status`

```python
def validate_ensemble_model_deployment_status(path_generated, run_type, ensemble_deployment_status):
        """
        Validates deployment status compatibility between an ensemble and its constituent models. 
        Prevents use of deprecated models or ensembles and ensures that production models 
        are only included in production ensembles.

        Args:
                path_generated (Path): Path to the model's data_generated directory containing log files.
                run_type (str): Type of run to validate: 'calibration', 'forecasting', or 'validation'.
                ensemble_deployment_status (str): Deployment status of the ensemble: 'production', 'shadow', or 'deprecated'.

        Returns:
                bool: True if deployment status conditions are met, False otherwise.

        Raises:
                Does not raise, but logs errors and returns False on failure.
        """
```

**Example:**

```python
is_valid = validate_ensemble_model_deployment_status(
        Path('models/rf_model/data/generated'), 'forecasting', 'production'
)
if not is_valid:
        print("Deployment status mismatch")
```

**Notes:**

- Deprecated ensembles or constituent models cannot be used.
- Production models must only be included in production ensembles.
- Logs errors for any status mismatches.

---

### `validate_partition_config`

```python
def validate_partition_config(ensemble_manager, model_manager, run_type):
        """
        Validates that the partition configuration (train/test split) for the ensemble matches 
        that of the constituent model for a given run type. Prevents data leakage and ensures fair evaluation.

        Args:
                ensemble_manager: EnsembleManager instance containing partition configuration.
                model_manager: ModelManager instance containing partition configuration.
                run_type (str): Type of run to validate: 'calibration', 'forecasting', or 'validation'.

        Returns:
                bool: True if partition configurations match, False otherwise.

        Raises:
                Does not raise, but logs errors and returns False on mismatch.
        """
```

**Example:**

```python
is_valid = validate_partition_config(ensemble_manager, model_manager, 'calibration')
if not is_valid:
        print("Partition mismatch detected")
```

**Notes:**

- Critical for fair ensemble evaluation.
- Logs both partition configs on mismatch.

---

### `validate_ensemble_model`

```python
def validate_ensemble_model(config):
        """
        Automates validation of all constituent models in an ensemble. 
        Checks temporal freshness, deployment status, and partition alignment for each model. 
        Exits the process if any check fails.

        Args:
                config (dict): Ensemble configuration dictionary. Must include:
                        "name": Ensemble name.
                        "models": List of constituent model names.
                        "run_type": Run type ('calibration', 'forecasting', 'validation').
                        "deployment_status": Deployment status of the ensemble.

        Returns:
                None. Exits the process (exit(1)) if any validation fails.
        """
```

**Example:**

```python
config = {
        "name": "ensemble_v1",
        "models": ["model_a", "model_b"],
        "run_type": "calibration",
        "deployment_status": "production"
}
validate_ensemble_model(config)
```

**Notes:**

- Imports managers internally to avoid circular dependencies.
- Logs a success message if all checks pass.

---

## Usage Examples

### Validating an Ensemble Before Forecasting

```python
from views_pipeline_core.modules.validation.ensemble import validate_ensemble_model

config = {
        "name": "ensemble_v1",
        "models": ["rf_model", "xgb_model"],
        "run_type": "forecasting",
        "deployment_status": "production"
}
validate_ensemble_model(config)
# If any model fails validation, the process will exit with an error.
```

---

## Best Practices

- Always validate all constituent models before running ensemble forecasts or evaluations.
- Ensure models and data are up-to-date for the current training cycle and month.
- Check deployment status to prevent accidental use of deprecated or shadow models in production.
- Align partition configurations to avoid data leakage and ensure fair ensemble evaluation.
- Integrate validation into your pipeline as an automated pre-check for robust, reproducible workflows.
- Review logs for detailed error messages when validation fails.
