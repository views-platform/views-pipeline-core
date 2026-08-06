# DatastoreModule

A high-level interface for managing prediction files and metadata in Appwrite cloud storage.

## Overview

The `DatastoreModule` provides a simplified API for the ViEWS pipeline to upload, download, search, and manage prediction files. It abstracts away the complexities of the underlying Appwrite storage system while providing essential features like metadata management, file versioning, and intelligent caching.

## Key Features

- **File Upload with Metadata**: Upload prediction files with structured metadata (loa, model name, targets, etc.)
- **Hash-based Deduplication**: Automatic detection and handling of duplicate files via SHA-256 hashing
- **Intelligent Caching**: Local file cache with TTL-based validation to reduce network requests
- **Metadata Search**: Query predictions by metadata attributes (loa, type, category, targets)
- **Automatic Bucket Creation**: Creates storage buckets on-demand if they don't exist
- **Version Tracking**: Files sorted by creation timestamp for easy access to latest versions

## Installation

The module is part of `views-pipeline-core`. Ensure you have the package installed:

```bash
pip install 'views-pipeline-core[appwrite]'
```

`DatastoreModule` reaches Appwrite, so it requires the optional `appwrite` extra
(#345, register C-253). Without it, importing this module fails with the install command.


## Quick Start

```python
from views_pipeline_core.modules.datastore import DatastoreModule
from views_pipeline_core.modules.appwrite import AppwriteConfig

# Configure connection to Appwrite
config = AppwriteConfig(
    endpoint="https://cloud.appwrite.io/v1",
    project_id="your_project_id",
    credentials="your_api_key",
    bucket_id="forecasts",
    collection_name="Predictions",
    database_id="file_metadata"
)

# Initialize the datastore
datastore = DatastoreModule(config)

# Upload a prediction file
result = datastore.upload_data(
    file="/path/to/predictions.parquet",
    filename="pgm_forecast_202401.parquet",
    loa="pgm",
    name="fatalities_ensemble",
    type="model",
    targets=["pred_ged_sb", "pred_ged_ns"],
    category="forecast",
    description="January 2024 fatality predictions"
)

if result.success:
    print(f"Uploaded file ID: {result.data['file_id']}")
```

## Environment Variables

For convenience, you can configure Appwrite credentials via environment variables:

```bash
APPWRITE_ENDPOINT=https://cloud.appwrite.io/v1
APPWRITE_DATASTORE_PROJECT_ID=your_project_id
APPWRITE_DATASTORE_API_KEY=your_api_key
```

## Classes

### FileMetadata

A validation container for prediction file metadata.

```python
from views_pipeline_core.modules.datastore import FileMetadata

metadata = FileMetadata(
    loa="pgm",                              # Level of analysis
    name="fatalities_model_v2",             # Model name
    type="model",                           # Type: model, postprocessor, ensemble
    targets=["pred_ged_sb", "pred_ged_ns"], # Target variables
    category="forecast",                    # forecast or historical
    description="Optional description"      # Optional
)

# Convert to dictionary for storage
metadata_dict = metadata.to_dict()
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `loa` | str | Yes | Level of analysis (e.g., 'pgm', 'cm') |
| `name` | str | Yes | Model name or identifier |
| `type` | str | Yes | Type of model ('model', 'postprocessor', 'ensemble') |
| `targets` | List[str] | Yes | List of target variable names |
| `category` | str | Yes | Must be 'forecast' or 'historical' |
| `description` | str | No | Human-readable description |

### DatastoreModule

The main interface for file operations.

```python
from views_pipeline_core.modules.datastore import DatastoreModule
from views_pipeline_core.modules.appwrite import AppwriteConfig

config = AppwriteConfig(...)
datastore = DatastoreModule(config)
```

## Methods

### upload_data()

Upload a file with associated metadata to Appwrite storage.

```python
result = datastore.upload_data(
    file="/path/to/predictions.parquet",
    filename="predictions_202401.parquet",
    loa="pgm",
    name="my_model",
    type="model",
    targets=["pred_ged_sb"],
    category="forecast",
    description="January 2024 predictions"
)
```

**Returns**: `OperationResult` with:
- `success`: Boolean indicating upload success
- `data`: Dict containing `file_id`, `document_id`, and metadata
- `code`: Status code ('UPLOAD_SUCCESS', 'METADATA_UPDATED', etc.)

### get_predictions_by_metadata()

Search for predictions matching metadata filters.

```python
# Find all PGM forecasts
predictions = datastore.get_predictions_by_metadata(
    filters={
        "loa": "pgm",
        "category": "forecast",
        "type": "model"
    }
)

for pred in predictions:
    print(f"{pred['filename']}: {pred['fileId']}")
```

**Returns**: List of metadata documents sorted by creation date (newest first)

### download_prediction()

Download a specific file by ID.

```python
# Download to memory
result = datastore.download_prediction(file_id="abc123")
file_bytes = result.data['file_bytes']

# Download to disk
result = datastore.download_prediction(
    file_id="abc123",
    save_path="/tmp/prediction.parquet",
    use_cache=True,
    validate_cache=True
)
```

**Parameters**:
- `file_id`: Appwrite file ID
- `save_path`: Optional path to save file (if None, returns bytes)
- `use_cache`: Whether to use local cache (default: True)
- `validate_cache`: Whether to validate cache freshness (default: True)

### download_latest_file()

Download the most recent file matching filters.

```python
result = datastore.download_latest_file(
    filters={"loa": "pgm", "category": "forecast"},
    save_path="/tmp/latest_forecast.parquet"
)

if result.success:
    print(f"Downloaded: {result.data['save_path']}")
    print(f"From cache: {result.data['from_cache']}")
```

**Raises**: `FileNotFoundError` if no files match the filters

### get_latest_file_id()

Get the file ID of the most recent matching prediction.

```python
file_id = datastore.get_latest_file_id(
    filters={"loa": "pgm", "type": "ensemble"}
)

if file_id:
    print(f"Latest file: {file_id}")
```

**Returns**: File ID string or None if no match

### get_file_metadata()

Retrieve metadata for a specific file.

```python
result = datastore.get_file_metadata(file_id="abc123")

if result.success:
    metadata = result.data
    print(f"Model: {metadata['name']}")
    print(f"Targets: {metadata['targets']}")
```

### update_prediction_metadata()

Update metadata for an existing file.

```python
result = datastore.update_prediction_metadata(
    file_id="abc123",
    metadata_updates={
        "description": "Updated description",
        "validated": True
    }
)
```

### delete_prediction()

Delete a file from storage.

```python
result = datastore.delete_prediction(file_id="abc123")

if result.success:
    print("File deleted successfully")
```

### list_all_predictions()

List all predictions for the current model.

```python
predictions = datastore.list_all_predictions()
print(f"Found {len(predictions)} predictions for this model")
```

### list_all_predictions_unfiltered()

List all predictions without filters (useful for debugging).

```python
all_predictions = datastore.list_all_predictions_unfiltered()
for pred in all_predictions:
    print(f"{pred['name']}: {pred['filename']}")
```

## Metadata Schema

Files uploaded through DatastoreModule have the following metadata structure:

| Field | Type | Description |
|-------|------|-------------|
| `loa` | string | Level of analysis (pgm, cm, etc.) |
| `name` | string | Model name |
| `type` | string | Model type (model, postprocessor, ensemble) |
| `targets` | array | List of target variable names |
| `category` | string | Either 'forecast' or 'historical' |
| `description` | string | Optional description |
| `fileId` | string | Appwrite storage file ID |
| `filename` | string | Original filename |
| `file_hash` | string | SHA-256 hash for deduplication |
| `uploaded_at` | datetime | Upload timestamp |

## Error Handling

All methods return an `OperationResult` object for consistent error handling:

```python
result = datastore.upload_data(...)

if result.success:
    print(f"Success! File ID: {result.data['file_id']}")
else:
    print(f"Error ({result.code}): {result.error}")
```

Common error codes:
- `storage_bucket_not_found`: Bucket doesn't exist
- `NOT_FOUND`: File or metadata not found
- `UPLOAD_FAILED`: Upload operation failed
- `METADATA_ERROR`: Metadata storage failed

## Caching Behavior

The DatastoreModule uses intelligent caching to minimize network requests:

1. **Cache Location**: Files cached in `{cache_dir}/appwrite_cache/{bucket_id}/`
2. **TTL Validation**: Cached files expire after configurable TTL (default: 24 hours)
3. **Timestamp Validation**: Cache invalidated if remote file is newer
4. **Automatic Updates**: Cache updated after successful downloads

Control caching behavior:
```python
# Bypass cache entirely
result = datastore.download_prediction(
    file_id="abc123",
    use_cache=False
)

# Use cache but don't validate timestamps
result = datastore.download_prediction(
    file_id="abc123",
    use_cache=True,
    validate_cache=False
)
```

## Example: Complete Workflow

```python
from views_pipeline_core.modules.datastore import DatastoreModule
from views_pipeline_core.modules.appwrite import AppwriteConfig
import pandas as pd

# Setup
config = AppwriteConfig(
    endpoint="https://cloud.appwrite.io/v1",
    project_id="views_project",
    credentials="api_key",
    bucket_id="production_forecasts"
)
datastore = DatastoreModule(config)

# Upload predictions
predictions_df = pd.read_parquet("/output/predictions.parquet")
predictions_df.to_parquet("/tmp/upload.parquet")

upload_result = datastore.upload_data(
    file="/tmp/upload.parquet",
    filename="pgm_forecast_202501.parquet",
    loa="pgm",
    name="ensemble_model",
    type="ensemble",
    targets=["pred_ged_sb", "pred_ged_ns", "pred_ged_os"],
    category="forecast"
)

print(f"Uploaded: {upload_result.data['file_id']}")

# Later: Download latest forecast
download_result = datastore.download_latest_file(
    filters={"loa": "pgm", "category": "forecast"},
    save_path="/tmp/latest.parquet"
)

df = pd.read_parquet(download_result.data['save_path'])
print(f"Loaded {len(df)} rows")
```

## See Also

- [AppWriteFileModule](../appwrite/README.md) - Lower-level Appwrite file operations
- [AppwriteConfig](../appwrite/README.md#appwriteconfig) - Configuration options
- [Appwrite Documentation](https://appwrite.io/docs) - Appwrite platform docs
