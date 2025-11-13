
# VIEWS Pipeline Core: Model Validation Module

This module provides robust validation utilities for model predictions and configuration files within the VIEWS Pipeline Core. It ensures that prediction outputs and model configurations adhere to the expected structure and standards, supporting both grid-level (PGM) and country-level (CM) models. These checks are essential for maintaining data integrity, reproducibility, and operational reliability in conflict forecasting pipelines.

---

## Contents

- [Overview](#overview)
- [Functions](#functions)
    - [validate_prediction_dataframe](#validate_prediction_dataframe)
    - [validate_config](#validate_config)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [References](#references)

---

## Overview

The validation module is designed to:

- **Enforce structural standards** for prediction DataFrames, ensuring compatibility with downstream evaluation and reporting tools.
- **Normalize and check model configuration files**, including deployment status and target variable formatting.
- **Support both PGM (priogrid-month) and CM (country-month) models**, automatically detecting the structure and validating accordingly.
- **Provide clear, color-coded console feedback** for validation status, aiding rapid debugging and transparency.
- **Raise informative errors** when validation fails, preventing silent failures and ensuring pipeline robustness.

---

## Functions

### `validate_prediction_dataframe`

```python
def validate_prediction_dataframe(dataframe: pd.DataFrame, target: Union[list, str]) -> None
```

**Description**:  
Validates the structure and required components of a prediction DataFrame. Ensures that the DataFrame contains the necessary target columns, proper index or column structure (for PGM or CM models), and valid data for evaluation and reporting.

**Args**:
- `dataframe` (`pd.DataFrame`): The prediction DataFrame to validate. Must contain:
    - For PGM: MultiIndex with `priogrid_id` or `priogrid_gid` and `month_id`
    - For CM: Columns `country_id`, `month_id`
    - Prediction columns named as `pred_{target}`
- `target` (`Union[list, str]`): Target variable name(s). Can be a single string (e.g., `'ged_sb'`) or a list of strings (e.g., `['ged_sb', 'ged_ns']`). Prediction columns must match these names.

**Raises**:
- `ValueError`: If validation fails due to:
    - Empty DataFrame
    - Invalid target type (not `str` or `list`)
    - Missing prediction columns
    - Missing `month_id` in index or columns
    - Unrecognized model structure (not PGM or CM)

**Example**:

```python
# Valid PGM prediction DataFrame
df = pd.DataFrame({
        'pred_ged_sb': [0.1, 0.2, 0.3]
}, index=pd.MultiIndex.from_tuples([
        (100, 480), (100, 481), (101, 480)
], names=['priogrid_gid', 'month_id']))
validate_prediction_dataframe(df, 'ged_sb')

# Valid CM prediction DataFrame
df = pd.DataFrame({
        'country_id': [1, 1, 2],
        'month_id': [480, 481, 480],
        'pred_ged_sb': [0.1, 0.2, 0.3]
})
validate_prediction_dataframe(df, 'ged_sb')
```

**Notes**:
- Supports both PGM (priogrid) and CM (country-month) models.
- For MultiIndex, checks index names for model type.
- For regular index, checks column names for model type.
- Prints colored validation status to the console for quick feedback.

---

### `validate_config`

```python
def validate_config(config: dict) -> None
```

**Description**:  
Validates and normalizes a model configuration dictionary. Checks the deployment status and ensures that the `targets` field is always a list, converting it if necessary. Prevents deprecated models from being used in the pipeline.

**Args**:
- `config` (`dict`): Model configuration dictionary with keys:
    - `'deployment_status'` (`str`): Model status (`'production'`, `'deprecated'`, `'shadow'`)
    - `'targets'` (`str` or `list`): Target variable name(s). Will be converted to a list if a string.
    - `'name'` (`str`): Model name (for error messages)

**Raises**:
- `ValueError`: If validation fails due to:
    - Model being deprecated
    - `targets` not being a string or list

**Example**:

```python
config = {
        'name': 'conflict_model_v1',
        'deployment_status': 'production',
        'targets': 'ged_sb'
}
validate_config(config)
print(config['targets'])  # ['ged_sb']

config = {
        'deployment_status': 'deprecated',
        'targets': ['ged_sb']
}
validate_config(config)  # Raises ValueError: Model is deprecated and cannot be used.
```

**Notes**:
- Modifies the `config` dictionary in-place.
- Converts string `targets` to a single-element list.
- Logs an error before raising an exception for deprecated models.

---

## Usage Examples

### Validating a Prediction DataFrame

```python
import pandas as pd
from views_pipeline_core.modules.validation.model.check import validate_prediction_dataframe

# Example for PGM
df = pd.DataFrame({
        'pred_ged_sb': [0.1, 0.2, 0.3]
}, index=pd.MultiIndex.from_tuples([
        (100, 480), (100, 481), (101, 480)
], names=['priogrid_gid', 'month_id']))
validate_prediction_dataframe(df, 'ged_sb')

# Example for CM
df = pd.DataFrame({
        'country_id': [1, 1, 2],
        'month_id': [480, 481, 480],
        'pred_ged_sb': [0.1, 0.2, 0.3]
})
validate_prediction_dataframe(df, ['ged_sb'])
```

### Validating and Normalizing a Model Config

```python
from views_pipeline_core.modules.validation.model.check import validate_config

config = {
        'name': 'my_model',
        'deployment_status': 'production',
        'targets': 'ged_sb'
}
validate_config(config)
print(config['targets'])  # ['ged_sb']
```

---

## Best Practices

- Always validate prediction outputs before evaluation or submission to ensure compatibility and prevent downstream errors.
- Normalize configuration files at load time to guarantee consistent handling of target variables.
- Check deployment status to avoid using deprecated models in production or reporting.