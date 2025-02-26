# Data Pipeline File Utilities Documentation

## Overview
This module provides file handling utilities for VIEWS data pipelines, including:
- Log file management for data provenance
- DataFrame serialization/deserialization
- Model deployment tracking

## Core Functions

### 1. Log File Management

#### `read_log_file(log_file_path: Union[str, Path]) -> dict`
Reads log files into a dictionary structure.

**Parameters:**
- `log_file_path`: Path to log file

**Returns:**  
Dictionary with log entries

**Example:**
```python
log_data = read_log_file("/data/raw/calibration_log.txt")
print(log_data["Data Fetch Timestamp"])
```

### `create_data_fetch_log_file(path_raw, run_type, model_name, data_fetch_timestamp)`

Creates standardized data fetch logs.

**Parameters:**

- `path_raw`: Raw data directory path
- `run_type`: Pipeline phase (calibration/validation/forecasting)
- `model_name`: Name of current model
- `data_fetch_timestamp`: Timestamp of data fetch operation

**Example:**

```python
create_data_fetch_log_file(
    Path("/data/raw"),
    "calibration",
    "orange_pasta",
    "20240315_142300"
)
```

### `create_specific_log_file(...)`

Creates detailed model-specific logs.

**Parameters:**

| Parameter                 | Description                          |
|---------------------------|--------------------------------------|
| `path_generated`          | Output directory                     |
| `run_type`                | Pipeline phase                       |
| `model_name`              | Model identifier                     |
| `deployment_status`       | Production status                    |
| `model_timestamp`         | Model training timestamp             |
| `data_generation_timestamp` | Feature processing timestamp        |
| `data_fetch_timestamp`    | Raw data acquisition timestamp       |
| `model_type`              | "single" or "ensemble"               |
| `mode`                    | File write mode ("w" or "a")         |

**Example:**
```python
create_specific_log_file(
    Path("/models/orange_pasta"),
    "calibration",
    "orange_pasta",
    "shadow",
    "20240315_093000",
    "20240315_142300",
    "20240315_140000",
    model_type="ensemble"
)
```

### `create_log_file(...)`

High-level log creation with ensemble support.

**Parameters:**

- `models`: List of component models for ensembles

Inherits parameters from `create_specific_log_file`.

**Example:**
```python
model_config = {
    "run_type": "calibration",
    "name": "example_ensemble",
    "deployment_status": "production"
}

create_log_file(
    Path("/ensembles/example_ensemble"),
    model_config,
    "20240315_100000",
    "20240315_143000", 
    "20240315_140000",
    model_type="ensemble",
    models=["purple_alien", "orange_pasta"]
)
```

### DataFrame Handling

#### `save_dataframe(dataframe: pd.DataFrame, save_path: Union[str, Path])`

Serializes DataFrames with format detection.

**Supported Formats:**

- Parquet (`.parquet`)
- Pickle (`.pkl`)

**Example:**
```python
df = pd.read_csv("data.csv")
save_dataframe(df, "processed_data.parquet")
```

#### `read_dataframe(file_path: Union[str, Path]) -> pd.DataFrame`

Loads serialized DataFrames with format detection.

```python
df = read_dataframe("forecasts_202403.pkl")
```

### Design Notes

**Log File Structure**

[Model Type] Model Name: orange_pasta    
[Model Type] Model Timestamp: 20240315_093000    
Data Generation Timestamp: 20240315_142300    
Data Fetch Timestamp: 20240315_140000    
Deployment Status: shadow    


