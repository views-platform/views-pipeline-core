# AppWriteFileModule

A comprehensive Python interface for Appwrite cloud storage with metadata management, caching, and flexible authentication.

## Overview

The `AppWriteFileModule` provides a full-featured interface for interacting with Appwrite cloud storage services. It handles file uploads/downloads, metadata tracking in Appwrite databases, local caching with TTL validation, and supports both API key and session-based authentication.

## Key Features

- **File Operations**: Upload, download, list, and delete files in Appwrite storage
- **Metadata Management**: Automatic database/collection creation and metadata tracking
- **Hash-based Deduplication**: SHA-256 hashing to detect and handle duplicate files
- **Intelligent Caching**: Local file cache with TTL and timestamp validation
- **Flexible Authentication**: Support for API key and user session authentication
- **Dynamic Schema**: Automatic database attribute creation based on metadata structure
- **Bucket Management**: Create and manage storage buckets programmatically

## Installation

```bash
pip install views-pipeline-core
```

## Quick Start

```python
from views_pipeline_core.modules.appwrite import (
    AppWriteFileModule,
    AppwriteConfig,
    AuthMethod
)

# Configure with API key authentication
config = AppwriteConfig(
    endpoint="https://cloud.appwrite.io/v1",
    project_id="my_project",
    credentials="my_api_key",
    auth_method=AuthMethod.API_KEY,
    bucket_id="my_bucket"
)

# Initialize the file manager
file_manager = AppWriteFileModule(config)

# Upload a file with metadata
result = file_manager.upload_file_with_metadata(
    bucket_id="my_bucket",
    file_path="/data/predictions.parquet",
    filename="predictions.parquet",
    metadata={"model": "ensemble_v2", "loa": "pgm"}
)

if result.success:
    print(f"File ID: {result.data['file_id']}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AppWriteFileModule                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │   AuthManager   │  │  CacheManager    │  │ MetadataHandler│ │
│  │  (API/Session)  │  │  (TTL Cache)     │  │  (Database)   │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Appwrite SDK Services                        │
│  ┌─────────┐  ┌───────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Storage │  │ Databases │  │ Account │  │  Users  │         │
│  └─────────┘  └───────────┘  └─────────┘  └─────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### AppwriteConfig

The central configuration dataclass for all Appwrite settings.

```python
from views_pipeline_core.modules.appwrite import AppwriteConfig, AuthMethod

config = AppwriteConfig(
    # Required
    endpoint="https://cloud.appwrite.io/v1",
    project_id="your_project_id",
    credentials="your_api_key",  # or {"email": "...", "password": "..."}
    
    # Authentication (default: API_KEY)
    auth_method=AuthMethod.API_KEY,
    
    # Storage settings
    bucket_id="production_forecasts",
    bucket_name="Production Forecasts",  # Optional, derived from bucket_id
    
    # Metadata settings
    collection_name="Metadata",
    collection_id="metadata",
    database_name="File Metadata",
    database_id="file_metadata",
    
    # Cache settings
    cache_dir="/path/to/cache",  # Optional
    cache_ttl_hours=24,
    allow_metadata_only_updates=True
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `endpoint` | str | Required | Appwrite server URL |
| `project_id` | str | Required | Appwrite project ID |
| `credentials` | str/dict | Required | API key or email/password dict |
| `auth_method` | AuthMethod | API_KEY | Authentication method |
| `bucket_id` | str | "production_forecasts" | Default storage bucket |
| `bucket_name` | str | Auto | Human-readable bucket name |
| `collection_name` | str | "Metadata" | Metadata collection name |
| `collection_id` | str | "metadata" | Metadata collection ID |
| `database_name` | str | "File Metadata" | Metadata database name |
| `database_id` | str | "file_metadata" | Metadata database ID |
| `cache_dir` | str | Auto | Local cache directory |
| `cache_ttl_hours` | int | 24 | Cache time-to-live in hours |
| `allow_metadata_only_updates` | bool | True | Update metadata only for existing files |

## Authentication

### API Key Authentication

For server-side applications and automated pipelines:

```python
config = AppwriteConfig(
    endpoint="https://cloud.appwrite.io/v1",
    project_id="my_project",
    credentials="your_server_api_key",
    auth_method=AuthMethod.API_KEY
)
```

### Session Authentication

### Session authentication was removed (#344)

`AuthMethod.SESSION`, `SessionAuth` and `get_current_user()` no longer exist. This
section used to show how to authenticate with an email and password; that mode was
never constructed anywhere on the platform, and it carried a credential shape the seam
contract has no slot for (þing-01 open item **O3**, register **C-255**).

`AuthMethod` now has exactly one member, `API_KEY`. A config still asking for
`auth_method="session"` fails at construction with
`ValueError: 'session' is not a valid AuthMethod` — deliberately, rather than falling
back to something that appears to work.

If you need per-user access, that is a new design question, not a revival of this one.

## Core Classes

### OperationResult

Standard result container for all operations:

```python
@dataclass
class OperationResult:
    success: bool           # Whether operation succeeded
    data: Any = None        # Result data (varies by operation)
    error: str = None       # Error message on failure
    code: str = None        # Status/error code

# Usage
result = file_manager.upload_file(...)
if result.success:
    file_id = result.data['$id']
else:
    print(f"Error ({result.code}): {result.error}")
```

### Common Result Codes

| Code | Description |
|------|-------------|
| `CREATED` | New resource created |
| `EXISTS` | Resource already exists |
| `UPLOAD_SUCCESS` | File uploaded with metadata |
| `METADATA_UPDATED` | Only metadata was updated (file unchanged) |
| `SAVED_FROM_CACHE` | File served from local cache |
| `SAVED_FROM_REMOTE` | File downloaded from server |
| `DELETED` | Resource deleted |
| `NOT_FOUND` | Resource not found |
| `FOUND_BY_HASH` | Duplicate found by file hash |
| `FOUND_BY_NAME` | File found by filename |

## File Operations

### Upload File

Upload a file without metadata tracking:

```python
result = file_manager.upload_file(
    bucket_id="my_bucket",
    file_path="/data/file.parquet",
    file_id=None,           # Auto-generated if None
    permissions=[],         # Optional permission strings
    check_duplicates=True,  # Check for existing files
    overwrite=False         # If duplicate, don't overwrite
)
```

### Upload File with Metadata

Upload a file and store metadata in the database:

```python
result = file_manager.upload_file_with_metadata(
    bucket_id="my_bucket",
    file_path="/data/predictions.parquet",
    filename="predictions.parquet",
    metadata={
        "model": "ensemble_v2",
        "loa": "pgm",
        "targets": ["pred_ged_sb", "pred_ged_ns"],
        "run_date": "2024-01-15"
    },
    collection_name="Predictions",  # Optional, uses config default
    collection_id="predictions"     # Optional, uses config default
)

if result.success:
    print(f"File ID: {result.data['file_id']}")
    print(f"Document ID: {result.data['document_id']}")
```

### Upload from Bytes

Upload in-memory data:

```python
import pandas as pd

df = pd.DataFrame({"col": [1, 2, 3]})
parquet_bytes = df.to_parquet()

result = file_manager.upload_file_from_bytes_with_metadata(
    bucket_id="my_bucket",
    file_bytes=parquet_bytes,
    filename="data.parquet",
    metadata={"type": "test"}
)
```

### Download File

Download with caching support:

```python
# Download to file
result = file_manager.download_file(
    bucket_id="my_bucket",
    file_id="abc123",
    save_path="/tmp/output.parquet",
    use_cache=True,
    validate_cache=True
)

print(f"From cache: {result.data['from_cache']}")

# Download to memory
result = file_manager.download_file(
    bucket_id="my_bucket",
    file_id="abc123"
)
file_bytes = result.data['file_bytes']
```

### List Files

List files in a bucket:

```python
result = file_manager.list_files(
    bucket_id="my_bucket",
    queries=None,           # Optional Appwrite Query objects
    limit=100,              # Max results
    offset=0,               # Pagination offset
    order_field="$createdAt",
    order_type="DESC"
)

for file in result.data['files']:
    print(f"{file['name']}: {file['$id']}")
```

### Delete File

```python
result = file_manager.delete_file(
    bucket_id="my_bucket",
    file_id="abc123"
)
```

### Get File Info

```python
result = file_manager.get_file(
    bucket_id="my_bucket",
    file_id="abc123"
)

if result.success:
    print(f"Name: {result.data['name']}")
    print(f"Size: {result.data['sizeOriginal']} bytes")
```

## Metadata Operations

### Search by Metadata

```python
# Equality filters
result = file_manager.metadata_manager.search_files_by_metadata(
    filters={"model": "ensemble", "loa": "pgm"},
    collection_name="Predictions"
)

# Array containment (for array fields like targets)
result = file_manager.metadata_manager.search_files_by_metadata(
    array_filters={"targets": "pred_ged_sb"},
    collection_name="Predictions"
)

for doc in result.data['documents']:
    print(f"{doc['filename']}: {doc['fileId']}")
```

### Update Metadata

```python
result = file_manager.metadata_manager.update_file_metadata(
    file_id="abc123",
    metadata_updates={
        "status": "validated",
        "score": 0.95
    }
)
```

### Check File Exists by Hash

```python
import hashlib

with open("/data/file.parquet", "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()

result = file_manager.metadata_manager.check_file_exists_by_hash(file_hash)

if result.success and result.code == "FOUND_BY_HASH":
    print(f"File exists: {result.data['fileId']}")
```

## Bucket Operations

### Create Bucket

```python
result = file_manager.create_bucket(
    bucket_id="new_bucket",
    name="New Bucket",
    permissions=[],
    file_security=True,
    maximum_file_size=100 * 1024 * 1024,  # 100MB
    allowed_file_extensions=["parquet", "csv"],
    encryption=False,
    compression="none",
    antivirus=True,
    create_metadata_db=True  # Auto-create metadata database
)
```

### List Buckets

```python
result = file_manager.list_buckets(
    search="forecast",  # Optional name search
    limit=100,
    offset=0
)

for bucket in result.data['buckets']:
    print(f"{bucket['name']}: {bucket['$id']}")
```

### Get Bucket Info

```python
result = file_manager.get_bucket(bucket_id="my_bucket")
```

## Caching

### How Caching Works

1. **Cache Key**: Generated as `{bucket_id}_{file_id}`
2. **Storage**: Files stored in `{cache_dir}/{bucket_id}/{filename}`
3. **Validation**: Checks TTL expiration and remote file timestamps
4. **Metadata**: Stored in `cache_metadata.json` for persistence

### Cache Validation Results

| Result | Description |
|--------|-------------|
| `VALID` | Cache is fresh and usable |
| `INVALID_TTL` | Cache has expired (exceeded TTL) |
| `INVALID_TIMESTAMP` | Remote file is newer than cache |
| `NOT_FOUND` | File not in cache |

### Cache Management

```python
# Clear all cache
result = file_manager.clear_cache()

# Clear cache for specific bucket
result = file_manager.clear_cache(bucket_id="my_bucket")

# Clear old cache entries
result = file_manager.clear_cache(older_than_hours=48)

print(f"Freed {result.data['deleted_bytes']} bytes")

# Get cache statistics
stats = file_manager.get_cache_stats()
print(f"Total: {stats['total_files']} files, {stats['total_size_mb']}MB")

for bucket, info in stats['by_bucket'].items():
    print(f"  {bucket}: {info['files']} files")
```

## Dynamic Schema

The module automatically creates database attributes based on metadata structure:

```python
# First upload with new metadata field
result = file_manager.upload_file_with_metadata(
    bucket_id="my_bucket",
    file_path="/data/file.parquet",
    filename="file.parquet",
    metadata={
        "model": "test",           # Creates string attribute
        "score": 0.95,             # Creates double attribute
        "targets": ["a", "b"],     # Creates string array attribute
        "validated": True          # Creates boolean attribute
    }
)
```

### Supported Attribute Types

| Python Type | Appwrite Type |
|-------------|---------------|
| str | string |
| int | integer |
| float | double |
| bool | boolean |
| datetime/ISO string | datetime |
| List[str] | string array |
| List[int] | integer array |

## Error Handling

```python
from appwrite.exception import AppwriteException

try:
    result = file_manager.upload_file_with_metadata(...)
    
    if result.success:
        print("Success!")
    else:
        if result.code == "storage_bucket_not_found":
            print("Bucket doesn't exist")
        elif result.code == "METADATA_ERROR":
            print("File uploaded but metadata failed")
        else:
            print(f"Error: {result.error}")
            
except AppwriteException as e:
    print(f"Appwrite error: {e.message}")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Example: Complete Workflow

```python
from views_pipeline_core.modules.appwrite import (
    AppWriteFileModule,
    AppwriteConfig,
    AuthMethod
)
import pandas as pd

# Initialize
config = AppwriteConfig(
    endpoint="https://cloud.appwrite.io/v1",
    project_id="views_project",
    credentials="api_key",
    bucket_id="forecasts"
)
file_manager = AppWriteFileModule(config)

# Create bucket if needed
bucket = file_manager.get_bucket("forecasts")
if not bucket.success:
    file_manager.create_bucket(
        bucket_id="forecasts",
        name="Forecasts",
        maximum_file_size=500 * 1024 * 1024
    )

# Upload prediction file
df = pd.read_parquet("/output/predictions.parquet")
df.to_parquet("/tmp/upload.parquet")

upload = file_manager.upload_file_with_metadata(
    bucket_id="forecasts",
    file_path="/tmp/upload.parquet",
    filename="pgm_forecast_202501.parquet",
    metadata={
        "model": "ensemble_v2",
        "loa": "pgm",
        "targets": ["pred_ged_sb", "pred_ged_ns"],
        "category": "forecast"
    }
)
print(f"Uploaded: {upload.data['file_id']}")

# Search for files
search = file_manager.metadata_manager.search_files_by_metadata(
    filters={"loa": "pgm", "category": "forecast"}
)
print(f"Found {search.data['total']} matching files")

# Download with caching
download = file_manager.download_file(
    bucket_id="forecasts",
    file_id=upload.data['file_id'],
    save_path="/tmp/downloaded.parquet"
)
print(f"Downloaded (from cache: {download.data['from_cache']})")

# Check cache stats
stats = file_manager.get_cache_stats()
print(f"Cache: {stats['total_size_mb']}MB")
```

## Debugging

### List Collection Attributes

```python
result = file_manager.debug_collection_attributes(
    collection_id="metadata",
    database_id="file_metadata"
)
# Attributes are logged at INFO level
```

### Check File Hash

```python
# Calculate hash for a local file
file_hash = file_manager._calculate_file_hash(file_path="/data/file.parquet")
print(f"Hash: {file_hash}")

# Check if file exists by hash
result = file_manager._file_exists_by_hash("bucket_id", file_hash, "filename.parquet")
print(f"Exists: {result.success}, Code: {result.code}")
```

## See Also

- [DatastoreModule](../datastore/README.md) - High-level prediction file management
- [Appwrite Documentation](https://appwrite.io/docs) - Appwrite platform documentation
- [Appwrite Python SDK](https://github.com/appwrite/sdk-for-python) - Python SDK reference
