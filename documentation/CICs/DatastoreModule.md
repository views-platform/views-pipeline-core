# Class Intent Contract: DatastoreModule

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology of the Repository), ADR-008 (Observability)

---

## 1. Purpose

Provides a high-level interface for uploading, downloading, searching, and managing prediction files stored in Appwrite cloud storage. Wraps the lower-level `AppWriteFileModule` and handles metadata management, file versioning and caching. It does **not** create buckets: provisioning moved to `views_pipeline_core.modules.appwrite.provisioning` in þing-02 #331. Paired with `FileMetadata`, which validates and encapsulates the metadata required for each stored file.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** implement Appwrite SDK calls directly; all storage/database operations are delegated to `AppWriteFileModule` and its `metadata_manager`.
- Does **not** support uploading `pd.DataFrame` objects directly (raises `NotImplementedError`).
- Does **not** validate the semantic correctness of prediction data (e.g., column names, index structure, value ranges). It stores and retrieves opaque files.
- Does **not** manage Appwrite authentication or connection lifecycle; that is the responsibility of `AppwriteConfig`.
- Does **not** enforce uniqueness of uploaded files; duplicate detection is handled at the `AppWriteFileModule` layer via file hashing.

---

## 3. Responsibilities and Guarantees

- **`FileMetadata`**: Validates all metadata fields at construction time. Enforces type safety (`TypeError`) and valid `category` values (`"forecast"` or `"historical"`, `ValueError`). The `to_dict()` method omits `description` if it is `None` or empty.
- **`upload_data()`**: Uploads a file (`Path` or `str`) with validated metadata to the configured Appwrite bucket. On `storage_bucket_not_found` it **fails and logs the remediation command** -- it no longer creates the bucket and retries. Auto-creation meant a mistyped or renamed coordinate silently provisioned a new production bucket and published the forecast where nobody reads (register C-228). Returns `OperationResult`.
- **`get_predictions_by_metadata()`**: Searches metadata documents with caller-supplied filters, automatically merging `model_path.model_name` as the `"name"` filter. Returns results sorted by `$createdAt` descending. Returns empty list on search failure.
- **`download_prediction()`**: Downloads a file by ID with optional caching (`use_cache`, `validate_cache`, `save_path`). The `validate_cache` parameter (default `True`) controls whether cached files are checked against their TTL before reuse. Supports saving to disk (`save_path`) or returning bytes in-memory.
- **`download_latest_file()`**: Convenience method combining `get_latest_file_id()` and `download_prediction()`. Raises `FileNotFoundError` if no files match.
- **`get_file_metadata()`**: Retrieves metadata for a single file ID. Returns `OperationResult` with code `"FOUND"` or `"NOT_FOUND"`.
- **`update_prediction_metadata()`**: Delegates metadata updates to `metadata_manager.update_file_metadata()`.
- **`delete_prediction()`**: Deletes a file from storage by ID.
- **`list_all_predictions()`**: Lists all predictions filtered by `model_path.model_name`.
- **`list_all_predictions_unfiltered()`**: Debug method that lists all predictions without filters.

---

## 4. Inputs and Assumptions

- Constructor requires an `AppwriteConfig` instance with valid `endpoint`, `project_id`, `credentials`, `bucket_id`, and optionally `collection_name`, `collection_id`, `database_id`, and `path_manager`.
- `upload_data()` expects `file` as `Path`, `str`, or `pd.DataFrame` (DataFrame not yet implemented). `filename`, `loa`, `type`, `targets`, and `category` are required. `name` defaults to `model_path.model_name` if `None`.
- `category` must be `"forecast"` or `"historical"`.
- `targets` must be a `List[str]`.
- Assumes `AppWriteFileModule` and `metadata_manager` are correctly configured and accessible.

---

## 5. Outputs and Side Effects

- All mutating operations (`upload_data`, `delete_prediction`, `update_prediction_metadata`) return `OperationResult` with `.success`, `.data`, `.code`, and `.error` fields.
- `get_predictions_by_metadata()` and `list_all_*` methods return `List[Dict[str, Any]]`.
- `download_prediction()` and `download_latest_file()` return `OperationResult` with either `file_bytes` or `save_path` in `.data`.
- Side effects: **creates no Appwrite infrastructure.** Writes files to local disk when `save_path` is provided. Logs extensively via `logging.getLogger(__name__)`.

---

## 6. Failure Modes and Loudness

- **`FileMetadata` construction**: Raises `TypeError` for wrong types, `ValueError` for invalid `category`. Fail loud and proud.
- **`upload_data()` with DataFrame**: Raises `NotImplementedError`. With unsupported type: raises `TypeError`.
- **`upload_data()` missing bucket**: Returns `OperationResult(success=False, code="storage_bucket_not_found")` and logs at ERROR naming the bucket and the `provisioning ensure-bucket` command. Does **not** raise, and does **not** create the bucket.
- **Callers must inspect the returned `OperationResult`.** Failure is reported in-band, not by exception: the SDK's `AppwriteException` is converted to `success=False` inside the storage module, so an `except` clause around `upload_data()` will not fire (register C-227, þing-02 #330).
- **`download_latest_file()` with no matches**: Raises `FileNotFoundError`.
- **`get_predictions_by_metadata()` search failure**: Logs warning/error and returns empty list. Does **not** raise.
- **`get_file_metadata()` errors**: Catches all exceptions and returns `OperationResult(success=False)`.
- The `OperationResult` pattern means most failures are data, not exceptions. Callers must check `.success` on returned results.

---

## 7. Boundaries and Interactions

- **Delegates to**: `AppWriteFileModule` (file operations), `AppWriteFileModule.metadata_manager` (metadata CRUD), `AppwriteConfig` (connection settings).
- **Used by**: Model managers and pipeline orchestration code that need to persist and retrieve prediction files.
- **`FileMetadata`** is a standalone validation container; it is instantiated inside `upload_data()` and `upload_predictions()` and converted to dict via `to_dict()`.
- `upload_predictions()` is deprecated and delegates entirely to `upload_data()`.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.modules.datastore import DatastoreModule
from views_pipeline_core.modules.appwrite import AppwriteConfig

config = AppwriteConfig(
    endpoint="https://cloud.appwrite.io/v1",
    project_id="views_project",
    credentials="api_key",
    bucket_id="forecasts",
    collection_name="Predictions",
    path_manager=my_path_manager,
)
datastore = DatastoreModule(config)

# Upload
result = datastore.upload_data(
    file="/data/predictions.parquet",
    filename="pgm_forecast_202401.parquet",
    loa="pgm",
    name="fatalities_model",
    type="model",
    targets=["pred_ged_sb"],
    category="forecast",
)
if result.success:
    print(result.data["file_id"])

# Download latest
result = datastore.download_latest_file(
    filters={"loa": "pgm", "category": "forecast"},
    save_path="/tmp/latest.parquet",
)

# Search
predictions = datastore.get_predictions_by_metadata(
    filters={"loa": "pgm"}
)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: Uploading a DataFrame directly (not implemented)
datastore.upload_data(file=my_dataframe, ...)  # raises NotImplementedError

# WRONG: Invalid category
FileMetadata(loa="pgm", name="m", type="model", targets=["t"], category="test")
# raises ValueError

# WRONG: Ignoring OperationResult.success
result = datastore.upload_data(...)
file_id = result.data["file_id"]  # may KeyError if result.success is False

# WRONG: Passing non-string targets
FileMetadata(loa="pgm", name="m", type="model", targets=[1, 2], category="forecast")
# raises TypeError
```

---

## 10. Test Alignment

Tests live in `tests/test_modules/test_datastore.py`. Coverage includes:

- **`TestPredictionMetadata`**: Valid initialization, missing description, all type validation errors (`TypeError` for loa, name, type, targets, description), invalid category (`ValueError`), `to_dict()` with and without description.
- **`TestPredictionStoreManager`**: Initialization, upload from `Path` and `str`, DataFrame `NotImplementedError`, invalid file type `TypeError`, missing-bucket reported not created, remediation command logged.

All tests use mocked `AppWriteFileModule` via `unittest.mock.patch` to avoid real Appwrite calls.

---

## 11. Evolution Notes

- `upload_predictions()` is deprecated; all new code should use `upload_data()`.
- DataFrame upload support (`NotImplementedError`) is a known gap that may be implemented in a future version.
- The `download_latest_file()` default `filters={}` uses a mutable default argument, which is a Python anti-pattern but is safe here since it is never mutated.

---

## 12. Known Deviations

- **`OperationResult` pattern**: Most methods return `OperationResult` instead of raising exceptions on failure. This means callers must explicitly check `.success` -- silent failure is possible if callers ignore the return value.
- **`upload_predictions()` deprecation**: Still exists and is tested. Delegates entirely to `upload_data()` with a `logger.warning()`.
- **Mutable default argument**: `download_latest_file(filters={})` uses a mutable default dict. Safe in practice but flagged by linters.

---

## End of Contract

This document defines the **intended meaning** of `DatastoreModule`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
