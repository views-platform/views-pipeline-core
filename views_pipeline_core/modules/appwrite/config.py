"""config.py — extracted from modules/appwrite/file.py (M-1 audit decision).

This module contains the classes that were previously in the
2,841-LOC God module `file.py`. The original `file.py` is now a
re-export shim that preserves all existing import paths.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
from datetime import datetime, timedelta
import hashlib
import logging
from views_pipeline_core.data.model_path import ModelPathManager
from views_pipeline_core.exceptions.exceptions import ConfigurationException

logger = logging.getLogger(__name__)

# Module-level constants (moved from file.py)
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_PAGE_LIMIT = 100
# DEFAULT_METADATA_COLLECTION_NAME = "file_metadata"
MAX_ATTRIBUTE_CREATION_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0

# The number of rows Appwrite's list endpoints return when no `Query.limit` is
# supplied. It is named here so that no walk in this module ever *inherits* it: a
# request that omits its own limit is correct only for as long as the server's default
# stays 25 and the match stays under it, and neither is ours to control. Register C-241
# is what this default cost when it was left implicit.
APPWRITE_DEFAULT_PAGE_SIZE = 25

# Backstop on a paging walk. It exists for the case where the substrate ignores
# `offset` — every page then comes back full and a `len(batch) < limit` terminator never
# fires. Tripping it means the walk is INCOMPLETE, never that it is finished.
MAX_METADATA_PAGES = 1000

# Page size for the container preflight. A project holds a handful of collections, so
# this is generous — but it is stated rather than inherited, and a total beyond it is
# refused rather than silently truncated.
_CONTAINER_PAGE = 100

# Appwrite error types. These are the SERVER's own `type` strings, which the SDK
# propagates verbatim into `AppwriteException.type` and this module carries as
# `OperationResult.code` — pinned against the real SDK in
# tests/test_modules/test_appwrite_sdk_contract.py.
APPWRITE_FILE_NOT_FOUND = "storage_file_not_found"
APPWRITE_BUCKET_NOT_FOUND = "storage_bucket_not_found"

# Coordinates that address real storage. They have no safe default: a default is a
# production address reachable without a deliberate choice (register C-229, þing-02
# #324; PLATFORM-001 §4). Their values belong to the seam's coordinate registry.
_REQUIRED_COORDINATES = (
    "bucket_id",
    "bucket_name",
    "collection_id",
    "collection_name",
    "database_id",
    "database_name",
)



# Enums




class AuthMethod(Enum):
    """Authentication methods supported by AppWriteFileModule.

    Attributes:
        API_KEY: Server-side API key authentication. Requires string credentials.

    Example:
        >>> config = AppwriteConfig(
        ...     auth_method=AuthMethod.API_KEY,
        ...     credentials="my_api_key"
        ... )
    """

    API_KEY = "api_key"


class _StoragePresence(Enum):
    """Whether a file is in a bucket — a three-valued question.

    `OperationResult.success` cannot express the third state, and substituting
    "absent" for "could not tell" is what let a permission denial delete a live
    forecast's metadata (register C-231). Any branch with a destructive consequence
    must require PRESENT or ABSENT and refuse to act on INDETERMINATE.
    """

    PRESENT = "present"
    ABSENT = "absent"
    INDETERMINATE = "indeterminate"


@dataclass
class OperationResult:
    """Standard result container for all Appwrite operations.

    Provides a consistent interface for returning operation outcomes across
    all file manager methods. Supports both successful results with data
    and failures with error information.

    Attributes:
        success: Whether the operation completed successfully.
        data: Result data on success. Structure varies by operation.
        error: Human-readable error message on failure.
        code: Machine-readable status/error code (e.g., 'CREATED', 'EXISTS',
            'NOT_FOUND', or Appwrite error types).

    Example:
        >>> result = file_manager.upload_file(bucket_id, file_path)
        >>> if result.success:
        ...     file_id = result.data['$id']
        ...     print(f"Uploaded: {file_id}")
        ... else:
        ...     print(f"Error ({result.code}): {result.error}")
        >>>
        >>> # Convert to dictionary for serialization
        >>> result_dict = result.to_dict()
    """

    success: bool
    data: Any = None
    error: Optional[str] = None
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format.

        Returns:
            Dictionary with keys: success, data, error, code.
        """
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "code": self.code
        }


@dataclass
class FileMetadata:
    """Metadata structure for files stored in Appwrite.

    Contains essential file information tracked in the metadata database.
    Used internally by AppwriteMetadataHandler for document creation.

    Attributes:
        fileId: Appwrite storage file ID.
        bucketId: ID of the bucket containing the file.
        filename: Original filename as stored.
        mime_type: MIME type of the file. Defaults to 'application/octet-stream'.
        uploaded_at: ISO format timestamp of upload. Auto-generated if not provided.
        file_size: Size of the file in bytes.
        file_hash: SHA-256 hash of file contents for deduplication.

    Example:
        >>> metadata = FileMetadata(
        ...     fileId="abc123",
        ...     bucketId="my_bucket",
        ...     filename="predictions.parquet",
        ...     file_size=1048576,
        ...     file_hash="sha256_hash_here"
        ... )
    """

    fileId: str
    bucketId: str
    filename: str
    mime_type: str = "application/octet-stream"
    uploaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_size: Optional[int] = None
    file_hash: Optional[str] = None


@dataclass(frozen=True)
class AppwriteConfig:
    """Configuration for Appwrite file manager connections.

    Centralized configuration for all Appwrite connection settings, authentication,
    caching behavior, and storage/metadata identifiers.

    **FROZEN (C-240, #346)** — attributes cannot be REBOUND. Validating coordinates in
    `__post_init__` is worthless if a caller can reassign one afterwards:
    `config.bucket_id = "production_forecasts"` a line later is exactly the move C-229
    was about. Its sibling `PredictionStoreConfig` was already `frozen=True`; two configs
    for one seam now carry one guarantee.

    It does NOT deep-freeze the objects behind the attributes — `path_manager` is a live
    collaborator and remains mutable through the wrapper. Deliberate: the guarantee this
    type needs is that a validated coordinate cannot be swapped, and every coordinate is
    a string.

    Attributes:
        endpoint: Appwrite server endpoint URL (e.g., 'https://cloud.appwrite.io/v1').
        project_id: Appwrite project identifier.
        credentials: Authentication API key (a string). The dict form documented here
            belonged to SESSION auth, deleted in #344 — API_KEY is the only method.
        auth_method: Authentication method to use. Defaults to API_KEY.
        cache_dir: Local directory for file caching. Auto-generated if not provided.
        cache_ttl_hours: Hours before cached files expire. Defaults to 24.
        allow_metadata_only_updates: If True, updates only metadata when file hash
            matches existing file. Defaults to True.
        bucket_id: Default storage bucket ID. Defaults to 'production_forecasts'.
        bucket_name: Human-readable bucket name. Derived from bucket_id if not set.
        collection_name: Metadata collection name. Defaults to 'Metadata'.
        collection_id: Metadata collection ID. Defaults to 'metadata'.
        database_name: Metadata database name. Derived from collection_id if not set.
        database_id: Metadata database ID. Defaults to 'file_metadata'.
        path_manager: Optional ModelPathManager for path resolution.

    Example:
        >>> # API key configuration
        >>> config = AppwriteConfig(
        ...     endpoint="https://cloud.appwrite.io/v1",
        ...     project_id="my_project",
        ...     credentials="secret_api_key",
        ...     auth_method=AuthMethod.API_KEY,
        ...     bucket_id="forecasts",
        ...     cache_ttl_hours=48
        ... )
        >>>
        >>> # Session authentication configuration
        >>> config = AppwriteConfig(
        ...     endpoint="https://cloud.appwrite.io/v1",
        ...     project_id="my_project",
        ...     credentials={"email": "user@example.com", "password": "secret"},
        ... )
    """

    # Core connection settings
    endpoint: str
    project_id: str
    # repr=False: a dataclass renders every field, so this live Appwrite key was one
    # `logger.debug(f"{config}")`, one W&B run-config capture or one traceback rendering
    # locals away from the logs — with nothing in the type to stop it (register C-230,
    # þing-01 #325). PLATFORM-001 §5's redaction clause is multi-carrier and binding:
    # credentials are never logged, in any carrier. Endpoints and coordinates may be, and
    # deliberately still render — a repr that hid everything would push people back to
    # printing the raw object.
    credentials: Union[str, Dict[str, str]] = field(repr=False)
    
    # Authentication settings
    auth_method: AuthMethod = AuthMethod.API_KEY
    
    # Cache settings
    cache_dir: Optional[str] = None
    cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS
    allow_metadata_only_updates: bool = True  # Whether to update metadata only when file hash exists
    
    # Storage and metadata coordinates. REQUIRED — see _REQUIRED_COORDINATES below.
    # Declared Optional only so they can keep their position after the defaulted
    # fields above; omitting one is a configuration error, not a default.
    bucket_id: Optional[str] = None
    bucket_name: Optional[str] = None
    collection_name: Optional[str] = None
    collection_id: Optional[str] = None
    database_name: Optional[str] = None
    database_id: Optional[str] = None

    # Path manager
    path_manager: ModelPathManager = None

    def __post_init__(self):
        # `object.__setattr__` because the dataclass is frozen (C-240). Coercion is the
        # ONE mutation this type permits, it happens before the instance is handed to
        # anyone, and `audit/targets.py` passes `auth_method="api_key"` as a string
        # so the coercion has a live caller.
        if isinstance(self.auth_method, str):
            object.__setattr__(self, "auth_method", AuthMethod(self.auth_method))

        # Every coordinate must arrive explicitly. Until 2026-07-31 these defaulted to
        # the live production values ("production_forecasts", "file_metadata", …), so a
        # caller, test or scratch script that supplied only a key operated against
        # production storage without ever choosing to (register C-229, þing-02 #324).
        # PLATFORM-001 §4 forbids baking registry coordinates into code, examples or
        # dataclass defaults for precisely this reason.
        #
        # Empty strings count as missing: `os.getenv("X", "")` is a common way to reach
        # here with nothing, and it must not pass silently.
        missing = [name for name in _REQUIRED_COORDINATES if not getattr(self, name)]
        if missing:
            raise ConfigurationException(
                f"AppwriteConfig is missing required coordinate(s): {missing}. "
                f"These address real storage and have no safe default — pass them "
                f"explicitly from the seam's coordinate registry (PLATFORM-001 §4), "
                f"or build the config from validated environment variables via "
                f"PredictionStoreConfig.from_environment().to_appwrite_config()."
            )

def _classify_storage_presence(result: "OperationResult") -> _StoragePresence:
    """Classify a ``get_file`` outcome without collapsing failure into absence.

    Only a positive ``storage_file_not_found`` from Appwrite counts as evidence of
    absence. Everything else — a wrong bucket id, a missing read scope, a 502 whose
    body was not JSON (so ``code`` is ``None``) — is INDETERMINATE. The match is
    deliberately positive rather than a negation: an unrecognised or absent code must
    fail safe, not authorise a delete.
    """
    if result.success:
        return _StoragePresence.PRESENT
    if result.code == APPWRITE_FILE_NOT_FOUND:
        return _StoragePresence.ABSENT
    return _StoragePresence.INDETERMINATE


def exception_message(exception: Exception) -> str:
    """The message of an ``AppwriteException`` as a string, always.

    The SDK sets ``AppwriteException.message`` to whatever it was handed. For an API
    error that is a string; for a **transport** failure it is the underlying exception
    OBJECT — so ``"something" in e.message`` raises
    ``TypeError: argument of type 'ConnectTimeout' is not iterable``.

    That was unreachable until #347: before timeouts existed, a hung call never returned
    at all, so no transport exception ever reached these handlers. Bounding the calls
    made the path reachable and turned an indefinite hang into a crash — which is
    precisely why þing-01 required the hang DRILLED before a value was shipped, and why
    that sequencing was not optional.

    Six call sites compared against ``e.message`` directly. This is the coercion they all
    needed; the deeper fix is to branch on ``e.type`` instead of prose (register C-235).
    """
    message = getattr(exception, "message", None)
    return message if isinstance(message, str) else str(message if message is not None else exception)
