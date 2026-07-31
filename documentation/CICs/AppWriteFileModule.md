
# Class Intent Contract: AppWriteFileModule

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-04-08
**Related ADRs:** ADR-008 (Observability)

---

## 1. Purpose

Provides a unified interface for uploading, downloading, listing, and deleting files
in Appwrite cloud storage with metadata tracking, SHA-256 hash-based deduplication,
TTL-validated local caching, and pluggable authentication (API key or session). It is
the pipeline's sole gateway to the Appwrite storage backend.

**Note (C-35):** This class is a known god class. It aggregates storage, metadata,
caching, and authentication concerns into a single facade. Decomposition is deferred
until usage patterns stabilize.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** interpret file contents. It is format-agnostic; parquet,
  CSV, and binary files are all treated as opaque byte streams.
- This class does **not** decide what to upload. Callers (e.g., `PredictionIOManager`)
  determine which files need to be stored.
- This class does **not** manage model paths or pipeline configuration.
- This class does **not** validate prediction DataFrames or model outputs.
- This class does **not** handle WandB logging or any observability beyond its own
  `logger` calls.

---

## 3. Responsibilities and Guarantees

- Guarantees that authentication is verified at construction; if credentials are
  invalid, `__init__` raises `ValueError`.
- Guarantees that `upload_file_with_metadata()` computes a SHA-256 hash before
  upload and checks for duplicates in both the metadata database and storage.
  When `allow_metadata_only_updates=True` and the hash matches, only metadata is
  updated (no redundant upload).
- Guarantees that `_calculate_file_hash()` uses SHA-256 with 4 KB chunked reads
  for file-path inputs.
- Guarantees that `CacheManager` validates cached files against both TTL expiry and
  remote `$updatedAt` timestamps before serving them.
- Guarantees that all public methods return `OperationResult` with `success`, `data`,
  `error`, and `code` fields, providing a uniform result contract.
- Guarantees that a metadata document is deleted as an orphan **only** on positive
  evidence of the file's absence (`storage_file_not_found`). A read that FAILED for any
  other reason -- wrong bucket id, missing read scope, a non-JSON error whose `code` is
  `None` -- is INDETERMINATE and the operation fails instead (`_classify_storage_presence`;
  register C-231, þing-02 #329).
- Guarantees that a paired write (file + metadata document) verifies its containers
  read-only **before** the file is uploaded, so a missing container cannot leave an
  orphaned file (`_require_containers`).
- **NO LONGER GUARANTEED (þing-02 #331):** `AppwriteMetadataHandler` no longer creates
  databases or collections, and `AppWriteFileModule` no longer creates buckets. Creating
  storage is a deliberate act, not a side effect of publishing; it moved to
  `views_pipeline_core.modules.appwrite.provisioning`, which the delivery path must not
  import (asserted in a subprocess by `tests/test_import_purity.py`, #332).

---

## 4. Inputs and Assumptions

- `config: AppwriteConfig` -- dataclass with:
  - `endpoint`, `project_id`, `credentials` (str for API key, dict for session).
  - `auth_method: AuthMethod` -- `API_KEY` or `SESSION`.
  - `bucket_id`, `bucket_name`, `collection_id`, `collection_name`, `database_id`,
    `database_name` -- **all six are REQUIRED and have no defaults**. Construction raises
    `ConfigurationException` naming every missing one. Until 2026-07-31 they defaulted to
    the live production coordinates, so a caller supplying only a key operated against
    production storage without choosing to (register C-229, #324; PLATFORM-001 §4 forbids
    baking registry coordinates into dataclass defaults).
  - `cache_ttl_hours: int` (default 24).
  - `allow_metadata_only_updates: bool` (default `True`).
  - `path_manager: Optional[ModelPathManager]` -- used for cache directory resolution.
- The Appwrite server must be reachable at the configured endpoint.
- **Bucket, database and collection must already exist.** They are verified read-only
  at the first paired write and never created; a missing container raises
  `ConfigurationException` naming the container and the command that creates it:

      python -m views_pipeline_core.modules.appwrite.provisioning ensure-collection

  (Changed by þing-02 #331: on-demand creation is what forced the platform's API key to
  carry create scopes, which blocked least privilege.)

---

## 5. Outputs and Side Effects

- **Primary outputs:** `OperationResult` from every public method.
- **Side effects:**
  - Uploads files to Appwrite storage buckets.
  - Creates/updates metadata documents in Appwrite databases.
  - Writes cached files and `cache_metadata.json` to the local cache directory.
  - Verifies (never creates) bucket and collection existence, once per instance.
  - Logs at `INFO` on successful operations, `WARNING` on fallbacks, `ERROR` on
    failures.

---

## 6. Failure Modes and Loudness

- `ValueError` at construction if authentication fails.
- `AppwriteException` propagated through `OperationResult.error` for storage and
  database failures (network, permissions, quota).
- `ValueError` from `_calculate_file_hash()` if neither `file_path` nor `file_bytes`
  is provided.
- Cache failures (corrupt metadata JSON, missing files) are logged as warnings and
  fall back to fresh downloads; they do **not** raise.
- Metadata storage failures after a successful upload are returned as
  `OperationResult(success=False)` but do **not** delete the uploaded file.

---

## 7. Boundaries and Interactions

- **Depends on:**
  - `appwrite` SDK (`Client`, `Storage`, `Databases`, `Account`, `Users`, `InputFile`,
    `Query`, `Permission`, `Role`, `ID`).
  - `AuthFactory` / `AuthManager` hierarchy (internal) for pluggable auth.
  - `CacheManager` (internal) for local TTL cache.
  - `AppwriteMetadataHandler` (internal) for database/collection management.
  - `ModelPathManager` (optional) for cache directory resolution.
- **Does not depend on:**
  - Any model manager, data loader, sniffer, or pipeline stage.
  - WandB or any non-Appwrite external service.
- **Trust boundary:** Appwrite API responses are trusted for metadata fields
  (`$id`, `$updatedAt`, `sizeOriginal`). File content integrity is verified via
  SHA-256 hash on upload; there is no hash verification on download.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.modules.appwrite.file import (
    AppWriteFileModule, AppwriteConfig, AuthMethod,
)

config = AppwriteConfig(
    endpoint="https://cloud.appwrite.io/v1",
    project_id="my_project",
    credentials="my_api_key",
    auth_method=AuthMethod.API_KEY,
    bucket_id="production_forecasts",
)
manager = AppWriteFileModule(config)

# Upload with metadata and deduplication
result = manager.upload_file_with_metadata(
    bucket_id="production_forecasts",
    file_path="/output/predictions.parquet",
    filename="predictions_202401.parquet",
    metadata={"model": "ensemble_v2", "loa": "pgm"},
)
assert result.success
```

```python
# Download with caching
result = manager.download_file(
    bucket_id="production_forecasts",
    file_id="abc123",
    use_cache=True,
)
if result.success:
    local_path = result.data["path"]
```

---

## 9. Examples of Incorrect Usage

- **Ignoring `OperationResult.success`:** Every method returns an `OperationResult`.
  Using `result.data` without checking `result.success` will produce `None`-access
  errors on failure.
- **Uploading without metadata when metadata is needed downstream:** Using
  `upload_file()` instead of `upload_file_with_metadata()` skips metadata storage.
  Downstream code that queries by hash or custom fields will not find the file.
- **Constructing with invalid credentials and catching `ValueError`:** The class
  is designed to fail loud at construction. Silently swallowing the `ValueError`
  and proceeding with an unauthenticated client will produce cryptic `AppwriteException`
  failures later.

---

## 10. Test Alignment

- **Green tests:** Unit tests with mocked Appwrite SDK can verify hash computation,
  deduplication logic, cache validation, and `OperationResult` construction.
- **Beige tests:** Integration tests against a live or emulated Appwrite instance
  verify end-to-end upload/download/metadata workflows.
- **Red tests:** Tests should verify that invalid credentials raise `ValueError`,
  that duplicate uploads with `allow_metadata_only_updates=True` skip re-upload,
  and that expired cache entries trigger fresh downloads.

---

## 11. Evolution Notes (Optional)

- This class is flagged as a god class (C-35). A future decomposition may split
  it into separate upload, download, and metadata facades.
- Download-side hash verification is not currently implemented; adding it would
  close the integrity loop.
- The `upload_file_from_bytes()` and `upload_file_from_bytes_with_metadata()` paths
  mirror the file-path equivalents; consolidation into a single polymorphic upload
  method may reduce surface area.

---

## End of Contract

This document defines the **intended meaning** of `AppWriteFileModule`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
