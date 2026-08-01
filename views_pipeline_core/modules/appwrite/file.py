"""Appwrite file management module for cloud storage operations.

This module provides a comprehensive interface for interacting with Appwrite
cloud storage, including file uploads/downloads, metadata management, caching,
and authentication. It supports both API key and session-based authentication.

The module consists of several key components:
    - AppWriteFileModule: Main interface for file operations
    - AppwriteMetadataHandler: Reads and updates file metadata documents
    - CacheManager: Local caching with TTL-based validation
    - AuthManager hierarchy: Flexible authentication (API key or session)

GOVERNING CONTRACT — this module sits on the VIEWS Appwrite *seam*, whose identity,
credential and coordinate rules are platform surface, not this repo's to set:

    PLATFORM-001, Identity, Secrets & Configuration Contract (views-appwrite),
    pinned at tag platform-001-v1.2.0:
    https://github.com/views-platform/views-appwrite/blob/platform-001-v1.2.0/docs/ADRs/platform/PLATFORM-001_identity_secrets_configuration_contract.md

    Coordinate registry (THE canonical source for bucket/collection/database ids —
    read and validate against it; never copy values into code or defaults):
    https://github.com/views-platform/views-appwrite/blob/platform-001-v1.2.0/docs/ADRs/platform/coordinate_registry.toml

Two clauses bear directly on the code below. **§6 (raise, never provision):** this module
creates no containers — provisioning lives in the sibling `provisioning.py`, runnable and
never importable from here, and a wrong coordinate fails rather than conjuring a new
production bucket. **§5 (redaction, all carriers):** credentials are never logged, in any
form. Locally: ADR-046 (amended 2026-07-31) and ADR-047 (three-destination authority).

Typical usage example:

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

    # Download with caching
    download = file_manager.download_file(
        bucket_id="my_bucket",
        file_id="abc123",
        save_path="/tmp/output.parquet",
        use_cache=True
    )

SDK RETURN-SHAPE CONTRACT (C-219 audit, 2026-07-27, #310 post-incident):
the Appwrite SDK dispatches on response Content-Type — application/json
returns a PARSED DICT, everything else raw bytes (pinned by
tests/test_modules/test_appwrite_sdk_contract.py). All API endpoints used
here serve JSON and are consumed as dicts (a shape surprise fails loud on
first subscript). The ONE bytes-expecting site platform-wide is
download_file, which coerces SDK-parsed JSON back to bytes (#310, C-217).
When adding a new SDK call, state its expected shape at the call site.
"""

from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.services.account import Account
from appwrite.services.users import Users
from appwrite.input_file import InputFile
from appwrite.id import ID
from appwrite.exception import AppwriteException
from appwrite.query import Query
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime, timedelta
import shutil
import logging
import hashlib
from views_pipeline_core.data.model_path import ModelPathManager
from views_pipeline_core.exceptions.exceptions import ConfigurationException
from abc import ABC, abstractmethod
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

# Constants
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


# Enums
class AuthMethod(Enum):
    """Authentication methods supported by AppWriteFileModule.

    Attributes:
        API_KEY: Server-side API key authentication. Requires string credentials.
        SESSION: User session authentication. Requires dict with email/password.

    Example:
        >>> config = AppwriteConfig(
        ...     auth_method=AuthMethod.API_KEY,
        ...     credentials="my_api_key"
        ... )
    """

    API_KEY = "api_key"
    SESSION = "session"


class CacheValidationResult(Enum):
    """Results of cache validation checks.

    Used by CacheManager.validate_cache() to indicate cache state.

    Attributes:
        VALID: Cache entry exists, is within TTL, and matches remote timestamp.
        INVALID_TTL: Cache entry exists but has exceeded the TTL period.
        INVALID_TIMESTAMP: Cache entry exists but remote file was updated after caching.
        NOT_FOUND: No cache entry exists for the requested file.

    Example:
        >>> validation = cache_manager.validate_cache(bucket_id, file_id)
        >>> if validation == CacheValidationResult.VALID:
        ...     # Use cached file
        ... else:
        ...     # Download fresh copy
    """

    VALID = "valid"
    INVALID_TTL = "invalid_ttl"
    INVALID_TIMESTAMP = "invalid_timestamp"
    NOT_FOUND = "not_found"

# Type Definitions
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

@dataclass
class AppwriteConfig:
    """Configuration for Appwrite file manager connections.

    Centralized configuration for all Appwrite connection settings, authentication,
    caching behavior, and storage/metadata identifiers.

    Attributes:
        endpoint: Appwrite server endpoint URL (e.g., 'https://cloud.appwrite.io/v1').
        project_id: Appwrite project identifier.
        credentials: Authentication credentials. String for API_KEY, dict with
            'email' and 'password' keys for SESSION auth.
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
        ...     auth_method=AuthMethod.SESSION
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
        if isinstance(self.auth_method, str):
            self.auth_method = AuthMethod(self.auth_method)

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

# Authentication Classes
class AuthManager(ABC):
    """Abstract base class for Appwrite authentication handlers.

    Defines the interface for authentication strategies used by AppWriteFileModule.
    Implementations must provide the setup() method to configure client authentication.

    See Also:
        ApiKeyAuth: Implementation for API key authentication.
        SessionAuth: Implementation for user session authentication.
        AuthFactory: Factory for creating appropriate AuthManager instances.
    """

    @abstractmethod
    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        """Configure authentication on the Appwrite client.

        Args:
            client: Appwrite Client instance to configure.
            credentials: Authentication credentials (format depends on implementation).

        Returns:
            OperationResult indicating success or failure of authentication setup.
        """
        pass

class ApiKeyAuth(AuthManager):
    """API key authentication handler for server-side Appwrite access.

    Uses a server API key for authentication, suitable for backend services
    and automated pipelines. Provides full access based on API key permissions.

    Example:
        >>> auth = ApiKeyAuth()
        >>> result = auth.setup(client, "my_api_key_string")
        >>> if result.success:
        ...     # Client is now authenticated
    """

    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        """Configure API key authentication on the client.

        Args:
            client: Appwrite Client instance to configure.
            credentials: API key string. Must be a string, not a dictionary.

        Returns:
            OperationResult with success=True if key was set, or success=False
            with error message if credentials format is invalid.
        """
        if not isinstance(credentials, str):
            return OperationResult(
                success=False,
                error="API key authentication requires string credentials",
                code="INVALID_CREDENTIALS"
            )
        
        client.set_key(credentials)
        return OperationResult(success=True)

class SessionAuth(AuthManager):
    """Session-based authentication handler for user-specific Appwrite access.

    Authenticates using email/password credentials to create a user session.
    Suitable for applications where user-specific permissions are needed.

    Attributes:
        account: Appwrite Account service instance after setup.
        current_user_id: ID of the authenticated user.

    Example:
        >>> auth = SessionAuth()
        >>> result = auth.setup(client, {
        ...     "email": "user@example.com",
        ...     "password": "secure_password"
        ... })
        >>> if result.success:
        ...     user_id = result.data['user_id']
    """

    def __init__(self):
        """Initialize SessionAuth with empty account and user ID."""
        self.account = None
        self.current_user_id = None

    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        """Configure session authentication by creating a user session.

        Args:
            client: Appwrite Client instance to configure.
            credentials: Dictionary with 'email' and 'password' keys.

        Returns:
            OperationResult with success=True and data containing 'user_id'
            on successful authentication, or success=False with error details.
        """
        if not isinstance(credentials, dict) or not all(k in credentials for k in ["email", "password"]):
            return OperationResult(
                success=False,
                error="Session authentication requires dict with 'email' and 'password'",
                code="INVALID_CREDENTIALS"
            )
        
        self.account = Account(client)
        client.set_key("")  # Clear API key for session auth
        
        session_result = self._create_session(credentials["email"], credentials["password"])
        if not session_result.success:
            return session_result
        
        self.current_user_id = session_result.data["user_id"]
        return OperationResult(success=True, data={"user_id": self.current_user_id})

    def _create_session(self, email: str, password: str) -> OperationResult:
        """Create an email/password session with Appwrite.

        Args:
            email: User's email address.
            password: User's password.

        Returns:
            OperationResult with session details on success, including
            session_id, user_id, and creation timestamp.
        """
        try:
            session = self.account.create_email_password_session(email, password)
            return OperationResult(
                success=True,
                data={
                    "session_id": session["$id"],
                    "user_id": session["userId"],
                    "created": session["$createdAt"]
                }
            )
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )

class AuthFactory:
    """Factory for creating authentication handler instances.

    Provides a static method to instantiate the appropriate AuthManager
    subclass based on the specified authentication method.

    Example:
        >>> auth_manager = AuthFactory.create_auth(AuthMethod.API_KEY)
        >>> result = auth_manager.setup(client, "my_api_key")
    """

    @staticmethod
    def create_auth(auth_method: AuthMethod) -> AuthManager:
        """Create an AuthManager instance for the specified method.

        Args:
            auth_method: The authentication method to use.

        Returns:
            Appropriate AuthManager subclass instance.

        Raises:
            ValueError: If auth_method is not supported.
        """
        if auth_method == AuthMethod.API_KEY:
            return ApiKeyAuth()
        elif auth_method == AuthMethod.SESSION:
            return SessionAuth()
        else:
            raise ValueError(f"Unsupported auth method: {auth_method}")

# Cache Management
@dataclass
class CacheMetadata:
    """Metadata for cached files.

    Stores information about cached files to enable validation and management.

    Attributes:
        bucket_id: ID of the bucket the file belongs to.
        file_id: Appwrite file ID.
        path: Local filesystem path to the cached file.
        cached_at: ISO format timestamp when file was cached.
        size_bytes: Size of the cached file in bytes.
        filename: Original filename.
        remote_updated_at: Remote file's last update timestamp for validation.
    """

    bucket_id: str
    file_id: str
    path: str
    cached_at: str
    size_bytes: int
    filename: str
    remote_updated_at: Optional[str] = None

class CacheManager:
    """Local file cache manager with TTL-based validation.

    Manages a local cache of downloaded files to reduce network requests.
    Supports TTL-based expiration and timestamp validation against remote files.

    The cache stores files organized by bucket ID and maintains a metadata JSON
    file to track cached files, their timestamps, and sizes.

    Attributes:
        cache_dir: Root directory for cached files.
        cache_ttl: Time-to-live duration for cached files.
        cache_metadata_file: Path to the JSON metadata file.
        cache_metadata: Dictionary mapping cache keys to CacheMetadata objects.

    Example:
        >>> from datetime import timedelta
        >>> cache = CacheManager(Path("/tmp/cache"), timedelta(hours=24))
        >>>
        >>> # Check if file is in valid cache
        >>> result = cache.validate_cache("bucket1", "file123")
        >>> if result == CacheValidationResult.VALID:
        ...     cached_path = cache.get_cached_file_path("bucket1", "file123")
    """

    def __init__(self, cache_dir: Path, cache_ttl: timedelta):
        """Initialize cache manager with directory and TTL settings.

        Args:
            cache_dir: Directory to store cached files. Will be created if needed.
            cache_ttl: Maximum age of cached files before they're considered stale.
        """
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.cache_metadata_file = cache_dir / "cache_metadata.json"
        self.cache_metadata: Dict[str, CacheMetadata] = {}
        self._load_cache_metadata()

    def _load_cache_metadata(self):
        """Load cache metadata from JSON file on disk.

        Reads the cache_metadata.json file and populates the cache_metadata
        dictionary. Silently handles missing or corrupted files.
        """
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, "r") as f:
                    data = json.load(f)
                    self.cache_metadata = {
                        k: CacheMetadata(**v) for k, v in data.items()
                    }
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}

    def _save_cache_metadata(self):
        """Save cache metadata to JSON file on disk.

        Persists the current cache_metadata dictionary to disk for
        recovery across sessions.
        """
        try:
            data = {k: v.__dict__ for k, v in self.cache_metadata.items()}
            with open(self.cache_metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, bucket_id: str, file_id: str) -> str:
        """Generate a unique cache key for a file.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.

        Returns:
            Combined key string in format 'bucket_id_file_id'.
        """
        return f"{bucket_id}_{file_id}"

    def _get_cache_path(self, bucket_id: str, file_id: str, filename: str = None) -> Path:
        """Get the filesystem path for a cached file.

        Creates the bucket subdirectory if it doesn't exist.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
            filename: Optional filename to use. Defaults to file_id.

        Returns:
            Path object pointing to the cache file location.
        """
        bucket_cache_dir = self.cache_dir / bucket_id
        bucket_cache_dir.mkdir(exist_ok=True)
        
        if filename:
            return bucket_cache_dir / filename
        return bucket_cache_dir / file_id

    def validate_cache(self, bucket_id: str, file_id: str, remote_updated_at: str = None) -> CacheValidationResult:
        """Check if a cached file is valid and usable.

        Validates cache entries based on:
        1. Existence in cache metadata
        2. Physical file existence on disk
        3. TTL expiration
        4. Remote file timestamp (if provided)

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
            remote_updated_at: Optional ISO timestamp of remote file's last update.
                If provided and newer than cache time, returns INVALID_TIMESTAMP.

        Returns:
            CacheValidationResult enum value:
                - VALID: Cache entry is usable
                - NOT_FOUND: No cache entry exists
                - INVALID_TTL: Cache has expired
                - INVALID_TIMESTAMP: Remote file is newer than cache

        Example:
            >>> result = cache.validate_cache("bucket1", "file123", "2024-01-15T10:00:00")
            >>> if result == CacheValidationResult.VALID:
            ...     # Safe to use cached file
        """
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        if cache_key not in self.cache_metadata:
            return CacheValidationResult.NOT_FOUND
        
        metadata = self.cache_metadata[cache_key]
        cache_path = Path(metadata.path)
        
        if not cache_path.exists():
            return CacheValidationResult.NOT_FOUND
        
        cached_at = datetime.fromisoformat(metadata.cached_at)
        if datetime.now() - cached_at > self.cache_ttl:
            return CacheValidationResult.INVALID_TTL
        
        if remote_updated_at:
            try:
                remote_updated = datetime.fromisoformat(remote_updated_at.replace("Z", "+00:00"))
                cached_at_aware = cached_at.replace(tzinfo=remote_updated.tzinfo)
                if remote_updated > cached_at_aware:
                    return CacheValidationResult.INVALID_TIMESTAMP
            except (ValueError, AttributeError):
                pass
        
        return CacheValidationResult.VALID

    def add_to_cache(self, bucket_id: str, file_id: str, file_path: Path, file_metadata: Dict[str, Any] = None):
        """Add or update a file in the cache.

        Records cache metadata for a downloaded file. Call this after
        successfully downloading a file from remote storage.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
            file_path: Path where the file is stored locally.
            file_metadata: Optional metadata from Appwrite including
                'name' and '$updatedAt' fields.

        Example:
            >>> cache.add_to_cache(
            ...     "my_bucket",
            ...     "file123",
            ...     Path("/cache/my_bucket/data.parquet"),
            ...     {"name": "data.parquet", "$updatedAt": "2024-01-15T10:00:00Z"}
            ... )
        """
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        self.cache_metadata[cache_key] = CacheMetadata(
            bucket_id=bucket_id,
            file_id=file_id,
            path=str(file_path),
            cached_at=datetime.now().isoformat(),
            size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            filename=file_metadata.get("name") if file_metadata else file_path.name,
            remote_updated_at=file_metadata.get("$updatedAt") if file_metadata else None
        )
        
        self._save_cache_metadata()

    def remove_from_cache(self, bucket_id: str, file_id: str):
        """Remove a file from the cache.

        Deletes both the cached file from disk and its metadata entry.
        Silently handles missing files.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
        """
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        if cache_key in self.cache_metadata:
            cache_path = Path(self.cache_metadata[cache_key].path)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError as e:
                    logger.warning(f"Failed to delete cache file {cache_path}: {e}")
            
            del self.cache_metadata[cache_key]
            self._save_cache_metadata()

    def get_cached_file_path(self, bucket_id: str, file_id: str) -> OperationResult:
        """Get the local filesystem path of a cached file.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.

        Returns:
            OperationResult with:
                - success=True and data containing 'cache_path' and 'metadata'
                - success=False if file not in cache or file missing from disk

        Example:
            >>> result = cache.get_cached_file_path("bucket1", "file123")
            >>> if result.success:
            ...     path = result.data['cache_path']
        """
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        if cache_key not in self.cache_metadata:
            return OperationResult(
                success=False,
                error="File not in cache",
                code="NOT_CACHED"
            )
        
        cache_path = Path(self.cache_metadata[cache_key].path)
        
        if not cache_path.exists():
            return OperationResult(
                success=False,
                error="Cache file missing",
                code="CACHE_FILE_MISSING"
            )
        
        return OperationResult(
            success=True,
            data={
                "cache_path": str(cache_path),
                "metadata": self.cache_metadata[cache_key].__dict__
            }
        )

    def clear_cache(self, bucket_id: str = None, older_than_hours: int = None) -> OperationResult:
        """Clear cached files matching specified criteria.

        Removes cached files from disk and metadata. Can filter by bucket
        and/or age to selectively clear cache.

        Args:
            bucket_id: Optional bucket ID to limit clearing to. If None,
                clears all buckets.
            older_than_hours: Optional age filter. Only clears files cached
                more than this many hours ago. If None, clears regardless of age.

        Returns:
            OperationResult with data containing:
                - deleted_files: Count of files deleted
                - deleted_bytes: Total bytes freed
                - errors: List of any deletion errors, or None

        Example:
            >>> # Clear all cache older than 48 hours
            >>> result = cache.clear_cache(older_than_hours=48)
            >>> print(f"Freed {result.data['deleted_bytes']} bytes")
            >>>
            >>> # Clear all cache for a specific bucket
            >>> result = cache.clear_cache(bucket_id="old_bucket")
        """
        deleted_count = 0
        deleted_bytes = 0
        errors = []
        keys_to_delete = []
        
        for cache_key, metadata in self.cache_metadata.items():
            should_delete = False
            
            if bucket_id and metadata.bucket_id != bucket_id:
                continue
            
            if older_than_hours:
                cached_at = datetime.fromisoformat(metadata.cached_at)
                if datetime.now() - cached_at < timedelta(hours=older_than_hours):
                    continue
            
            should_delete = True
            
            if should_delete:
                cache_path = Path(metadata.path)
                if cache_path.exists():
                    try:
                        size = cache_path.stat().st_size
                        cache_path.unlink()
                        deleted_count += 1
                        deleted_bytes += size
                    except OSError as e:
                        errors.append(f"Failed to delete {cache_path}: {e}")
                
                keys_to_delete.append(cache_key)
        
        for key in keys_to_delete:
            del self.cache_metadata[key]
        
        self._save_cache_metadata()
        
        return OperationResult(
            success=True,
            data={
                "deleted_files": deleted_count,
                "deleted_bytes": deleted_bytes,
                "errors": errors if errors else None
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics and usage information.

        Returns:
            Dictionary containing:
                - total_files: Number of cached files
                - total_size_bytes: Total cache size in bytes
                - total_size_mb: Total cache size in megabytes
                - cache_dir: Path to cache directory
                - by_bucket: Dict mapping bucket IDs to {files, bytes}

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Cache: {stats['total_files']} files, {stats['total_size_mb']}MB")
            >>> for bucket, info in stats['by_bucket'].items():
            ...     print(f"  {bucket}: {info['files']} files")
        """
        total_files = len(self.cache_metadata)
        total_bytes = 0
        by_bucket = {}
        
        for metadata in self.cache_metadata.values():
            bucket_id = metadata.bucket_id
            size = metadata.size_bytes
            total_bytes += size
            
            if bucket_id not in by_bucket:
                by_bucket[bucket_id] = {"files": 0, "bytes": 0}
            
            by_bucket[bucket_id]["files"] += 1
            by_bucket[bucket_id]["bytes"] += size
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "by_bucket": by_bucket
        }

# Metadata Management
class AppwriteMetadataHandler:
    """Handler for reading and writing file metadata documents in Appwrite.

    Searches, reads and updates metadata documents in an EXISTING database and
    collection.

    **It no longer creates them.** Provisioning — databases, collections, attributes —
    moved to ``views_pipeline_core.modules.appwrite.provisioning`` in þing-02 #331,
    because creating storage as a side effect of publishing a forecast is what forced
    the platform's API key to carry create scopes, and least privilege could not be
    applied while narrowing the key broke every upload. Containers are now created
    deliberately::

        python -m views_pipeline_core.modules.appwrite.provisioning ensure-collection

    and their existence is verified read-only before the first write by
    :meth:`AppWriteFileModule._require_containers`.

    Attributes:
        databases: Appwrite Databases service instance.
        config: AppwriteConfig with database/collection settings.

    Example:
        >>> handler = AppwriteMetadataHandler(databases_service, config)
        >>>
        >>> # Search for files by metadata
        >>> results = handler.search_files_by_metadata(
        ...     filters={"model": "test"},
        ...     array_filters={"targets": "ged_sb"}
        ... )
    """

    def __init__(self, databases: Databases, config: AppwriteConfig):
        """Initialize metadata handler with database service and config.

        Args:
            databases: Appwrite Databases service instance.
            config: AppwriteConfig with database/collection identifiers.
        """
        self.databases = databases
        self.config = config


    def search_files_by_metadata(
    self,
    filters: Dict[str, Any] = None,
    array_filters: Dict[str, Any] = None,
    collection_name: str = None,
    collection_id: str = None,
    database_id: str = None,
) -> OperationResult:
        """Search for files by metadata attributes.

        Queries the metadata collection with equality filters and/or array
        containment filters to find matching file metadata documents.

        Args:
            filters: Dict of attribute=value pairs for equality matching.
            array_filters: Dict of attribute=value pairs for array containment.
            collection_name: Collection name. Defaults to config.collection_name.
            collection_id: Collection ID. Defaults to config.collection_id.
            database_id: Database ID. Defaults to config.database_id.

        Returns:
            OperationResult with:
                - success=True and data containing 'documents' list and 'total' count
                - success=False if search fails

        Example:
            >>> # Find files by exact match
            >>> result = handler.search_files_by_metadata(
            ...     filters={"model": "ensemble", "loa": "pgm"}
            ... )
            >>>
            >>> # Find files where array contains value
            >>> result = handler.search_files_by_metadata(
            ...     array_filters={"targets": "ged_sb"}
            ... )
            >>>
            >>> if result.success:
            ...     for doc in result.data['documents']:
            ...         print(doc['filename'])
        """
        # Use config values as defaults
        db_id = database_id or self.config.database_id
        coll_id = collection_id or self.config.collection_id

        if not db_id or not coll_id:
            return OperationResult(
                success=False,
                error="Database ID and collection ID must be provided in config or as parameters",
                code="MISSING_CONFIG",
            )

        try:
            queries = []

            if filters:
                for attribute, value in filters.items():
                    if value is not None:
                        queries.append(Query.equal(attribute, value))

            if array_filters:
                for attribute, value in array_filters.items():
                    if value is not None:
                        queries.append(Query.contains(attribute, value))

            # C-241: `list_documents` returns APPWRITE_DEFAULT_PAGE_SIZE rows unless a
            # `Query.limit` is supplied. This method used to omit one, so a match of
            # more than 25 documents came back silently truncated — and every caller
            # treated the truncation as the whole answer. `get_latest_file_id` then
            # returned the newest of the OLDEST 25, which does not fail: it delivers a
            # stale run as though it were current. views-faoapi hit the same default
            # from the other side (their #287) and pages the same way.
            #
            # The walk terminates on an EMPTY page rather than on a short one, and
            # advances the offset by what it RECEIVED rather than by what it asked for.
            # Both matter: Appwrite may grant less than the requested limit, and a walk
            # that treats a short page as the end skips every row after it.
            documents = []
            reported_total = None
            offset = 0
            complete = False

            for _ in range(MAX_METADATA_PAGES):
                page = self.databases.list_documents(
                    db_id,
                    coll_id,
                    queries=queries
                    + [Query.limit(DEFAULT_PAGE_LIMIT), Query.offset(offset)],
                )
                batch = page.get("documents") or []
                if reported_total is None:
                    reported_total = page.get("total")
                documents.extend(batch)
                if not batch:
                    complete = True
                    break
                offset += len(batch)

            if not complete:
                # The substrate kept handing back full pages. The usual cause is an
                # ignored offset, and the one thing we must not do is return the rows
                # we happen to hold as if they were the match.
                logger.error(
                    f"Search of {coll_id!r} did not terminate within "
                    f"{MAX_METADATA_PAGES} pages; refusing to report a partial result"
                )
                return OperationResult(
                    success=False,
                    error=(
                        f"Search incomplete: walk of {coll_id!r} exceeded the "
                        f"{MAX_METADATA_PAGES}-page guard after "
                        f"{len(documents)} documents"
                    ),
                    code="SEARCH_INCOMPLETE",
                )

            # The server told us how many documents match. Enumerating a different
            # number means the walk cannot be certified, and an uncertifiable read must
            # not be handed back as an answer — that conflation is the defect class this
            # method was an instance of. A concurrent write during the walk lands here
            # too; the caller retries rather than delivering a count nobody verified.
            if reported_total is not None and len(documents) != reported_total:
                logger.error(
                    f"Search of {coll_id!r} enumerated {len(documents)} documents but "
                    f"the collection reports total={reported_total}"
                )
                return OperationResult(
                    success=False,
                    error=(
                        f"Search incomplete: enumerated {len(documents)} of a reported "
                        f"{reported_total} documents in {coll_id!r}"
                    ),
                    code="SEARCH_INCOMPLETE",
                )

            return OperationResult(
                success=True,
                data={"documents": documents, "total": len(documents)},
            )

        except AppwriteException as e:
            logger.error(f"Search failed: {e.message}")
            return OperationResult(
                success=False, error=f"Search failed: {e.message}", code=e.type
            )

    def check_file_exists_by_hash(
    self,
    file_hash: str,
    collection_name: str = None,
    collection_id: str = None,
    database_id: str = None,
) -> OperationResult:
        """Check if a file with the given hash exists in metadata.

        Searches the metadata collection for a file with matching file_hash.
        Creates the file_hash attribute if it doesn't exist in the schema.

        Args:
            file_hash: SHA-256 hash of file contents to search for.
            collection_name: Collection name. Defaults to config.collection_name.
            collection_id: Collection ID. Defaults to config.collection_id.
            database_id: Database ID. Defaults to config.database_id.

        Returns:
            OperationResult with:
                - success=True, code='FOUND_BY_HASH' and document data if found
                - success=False, code='NOT_FOUND' if no match

        Example:
            >>> file_hash = hashlib.sha256(file_bytes).hexdigest()
            >>> result = handler.check_file_exists_by_hash(file_hash)
            >>> if result.success and result.code == 'FOUND_BY_HASH':
            ...     existing_file_id = result.data['fileId']
        """
        # Use config values as defaults
        db_id = database_id or self.config.database_id
        coll_id = collection_id or self.config.collection_id

        if not db_id or not coll_id:
            return OperationResult(
                success=False,
                error="Database ID and collection ID must be provided in config or as parameters",
                code="MISSING_CONFIG",
            )
        try:
            # This is a QUERY. It used to create the database, collection and attributes
            # before searching — a read with a write's side effect, and one of the
            # reasons the platform key needed create scopes (register C-233, þing-02
            # #331). The collection is now provisioned deliberately:
            #   python -m views_pipeline_core.modules.appwrite.provisioning ensure-collection
            # If it does not exist, the read below fails loud and says so.
            # limit(1): this reads `total` for existence and `documents[0]` for the
            # answer, so one row is all it consumes. `total` is reported over the whole
            # match regardless of page size, so bounding the page cannot hide a
            # duplicate hash. Explicit because "the default happens to be enough" is
            # what C-241 was.
            search_result = self.databases.list_documents(
                db_id,
                coll_id,
                queries=[Query.equal("file_hash", file_hash), Query.limit(1)],
            )

            if search_result["total"] > 0:
                return OperationResult(
                    success=True, 
                    data=search_result["documents"][0], 
                    code="FOUND_BY_HASH"  # <-- CHANGED from "FOUND" to "FOUND_BY_HASH"
                )

            return OperationResult(success=False, code="NOT_FOUND")

        except AppwriteException as e:
            # If the file_hash attribute doesn't exist, create it and try again
            if "Attribute not found in schema: file_hash" in e.message:
                logger.info("file_hash attribute not found, creating it...")
                try:
                    self._create_attribute_by_type(
                        db_id, coll_id, "file_hash", "string", False
                    )

                    # Try the search again
                    try:
                        search_result = self.databases.list_documents(
                            db_id,
                            coll_id,
                            queries=[
                                Query.equal("file_hash", file_hash),
                                Query.limit(1),
                            ],
                        )

                        if search_result["total"] > 0:
                            return OperationResult(
                                success=True,
                                data=search_result["documents"][0],
                                code="FOUND_BY_HASH"  # <-- CHANGED here too
                            )

                        return OperationResult(success=False, code="NOT_FOUND")
                    except AppwriteException as retry_e:
                        logger.error(
                            f"Search failed after creating attribute: {retry_e.message}"
                        )
                        return OperationResult(
                            success=False,
                            error=f"Search failed: {retry_e.message}",
                            code=retry_e.type,
                        )
                except AppwriteException as create_e:
                    logger.error(
                        f"Failed to create file_hash attribute: {create_e.message}"
                    )
                    return OperationResult(
                        success=False,
                        error=f"Attribute creation failed: {create_e.message}",
                        code=create_e.type,
                    )

            logger.error(f"Search failed: {e.message}")
            return OperationResult(
                success=False, error=f"Search failed: {e.message}", code=e.type
            )

    def update_file_metadata(
        self,
        file_id: str,
        metadata_updates: Dict[str, Any],
        collection_name: str = None,
        collection_id: str = None,
        database_id: str = None
    ) -> OperationResult:
        """Update metadata for an existing file.

        Finds the metadata document by fileId and updates specified fields.

        Args:
            file_id: Appwrite file ID to update metadata for.
            metadata_updates: Dict of fields to update with new values.
            collection_name: Collection name. Defaults to config.collection_name.
            collection_id: Collection ID. Defaults to config.collection_id.
            database_id: Database ID. Defaults to config.database_id.

        Returns:
            OperationResult with:
                - success=True, code='UPDATED' and updated document data
                - success=False, code='METADATA_NOT_FOUND' if file not found

        Example:
            >>> result = handler.update_file_metadata(
            ...     file_id="abc123",
            ...     metadata_updates={"status": "validated", "score": 0.95}
            ... )
        """
        # Use config values as defaults
        db_id = database_id or self.config.database_id
        coll_id = collection_id or self.config.collection_id
        
        if not db_id or not coll_id:
            return OperationResult(
                success=False,
                error="Database ID and collection ID must be provided in config or as parameters",
                code="MISSING_CONFIG"
            )
        try:
            # fileId is unique per document by construction, and only documents[0] is
            # used below.
            search_result = self.databases.list_documents(
                database_id=db_id,
                collection_id=coll_id,
                queries=[Query.equal("fileId", file_id), Query.limit(1)],
            )
            
            if not search_result["documents"]:
                return OperationResult(
                    success=False,
                    error=f"No metadata found for file ID: {file_id}",
                    code="METADATA_NOT_FOUND"
                )
            
            document_id = search_result["documents"][0]["$id"]
            
            result = self.databases.update_document(
                database_id=db_id,
                collection_id=coll_id,
                document_id=document_id,
                data=metadata_updates
            )
            
            return OperationResult(
                success=True,
                data=result,
                code="UPDATED"
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"Metadata update failed: {e.message}",
                code=e.type
            )

# Main File Manager
class AppWriteFileModule:
    """Main interface for Appwrite file storage operations.

    Provides comprehensive file management capabilities including uploads,
    downloads, metadata tracking, caching, and bucket management. Supports
    both file path and byte-based uploads with automatic deduplication.

    Key features:
        - File uploads with hash-based deduplication
        - Metadata storage in Appwrite databases
        - Local caching with TTL validation
        - Bucket and collection management
        - Support for API key and session authentication

    Attributes:
        config: AppwriteConfig with connection and storage settings.
        client: Appwrite Client instance.
        storage: Appwrite Storage service.
        databases: Appwrite Databases service.
        users: Appwrite Users service.
        metadata_manager: AppwriteMetadataHandler for database operations.
        cache_manager: CacheManager for local file caching.
        auth_manager: AuthManager instance handling authentication.

    Example:
        >>> from views_pipeline_core.modules.appwrite import (
        ...     AppWriteFileModule, AppwriteConfig, AuthMethod
        ... )
        >>>
        >>> config = AppwriteConfig(
        ...     endpoint="https://cloud.appwrite.io/v1",
        ...     project_id="my_project",
        ...     credentials="my_api_key",
        ...     bucket_id="forecasts"
        ... )
        >>> file_manager = AppWriteFileModule(config)
        >>>
        >>> # Upload with metadata
        >>> result = file_manager.upload_file_with_metadata(
        ...     bucket_id="forecasts",
        ...     file_path="/data/predictions.parquet",
        ...     filename="predictions.parquet",
        ...     metadata={"model": "ensemble", "loa": "pgm"}
        ... )
        >>>
        >>> # Download with caching
        >>> download = file_manager.download_file(
        ...     bucket_id="forecasts",
        ...     file_id=result.data['file_id'],
        ...     use_cache=True
        ... )
    """

    def __init__(self, config: AppwriteConfig):
        """Initialize AppWriteFileModule with configuration.

        Sets up Appwrite client, authentication, and internal managers
        for metadata and caching.

        Args:
            config: AppwriteConfig with all connection and storage settings.

        Raises:
            ValueError: If authentication fails with provided credentials.

        Example:
            >>> config = AppwriteConfig(
            ...     endpoint="https://cloud.appwrite.io/v1",
            ...     project_id="my_project",
            ...     credentials="api_key"
            ... )
            >>> manager = AppWriteFileModule(config)
        """
        # if not isinstance(config.path_manager, ModelPathManager):
        #     raise ValueError("path_manager must be an instance of ModelPathManager")

        self.config = config
        self.client = Client()
        self.client.set_endpoint(config.endpoint).set_project(config.project_id)
        
        # Initialize authentication
        self.auth_manager = AuthFactory.create_auth(config.auth_method)
        auth_result = self.auth_manager.setup(self.client, config.credentials)
        if not auth_result.success:
            raise ValueError(f"Authentication failed: {auth_result.error}")
        
        # Initialize services
        self.storage = Storage(self.client)
        self.users = Users(self.client)
        self.databases = Databases(self.client)
        
        # Initialize managers
        self.metadata_manager = AppwriteMetadataHandler(self.databases, config)
        self.cache_manager = self._setup_cache()

        # Containers are verified once per process, before the first write.
        self._containers_verified = False

    def _require_containers(self, bucket_id: str, collection_id: str = None) -> None:
        """Fail loud, BEFORE any write, if a target container does not exist.

        Provisioning left the delivery path in þing-02 #331, so an upload can no longer
        conjure its own destination. Something must still notice when a destination is
        missing — and it has to notice *before* the file is uploaded, because
        ``upload_file_with_metadata`` writes the file first and the metadata document
        second. Discovering a missing collection after the upload would leave the file
        in the bucket with no index card: an orphan, which is the corruption #329
        exists to remove.

        Read-only (``get_bucket`` + ``list_collections``) and cached per instance, so
        the cost is two calls per process rather than two per upload.

        **Scope: paired writes only.** The single-write methods (``upload_file``,
        ``update_file_metadata``) deliberately do not call this. They touch one
        container, so a missing one surfaces as a loud Appwrite error with nothing
        half-written behind it — there is no pair to leave inconsistent. Add the
        precondition here if a third paired write is ever introduced.

        Raises:
            ConfigurationException: naming the missing container and the exact command
                that creates it.
        """
        if self._containers_verified:
            return

        from views_pipeline_core.exceptions.exceptions import ConfigurationException

        bucket_check = self.get_bucket(bucket_id)
        if not bucket_check.success:
            raise ConfigurationException(
                f"Appwrite bucket '{bucket_id}' is not usable: {bucket_check.error} "
                f"(code={bucket_check.code}). If it does not exist, create it "
                f"deliberately:\n"
                f"    python -m views_pipeline_core.modules.appwrite.provisioning "
                f"ensure-bucket --bucket {bucket_id}"
            )

        coll_id = collection_id or self.config.collection_id
        try:
            # This builds a membership set, so a SHORT read does not return less — it
            # returns a WRONG answer: a collection that exists but falls past the page
            # boundary reads as missing and fails the preflight. The limit is stated,
            # and a total larger than the page is refused rather than guessed at.
            collections = self.databases.list_collections(
                self.config.database_id, queries=[Query.limit(_CONTAINER_PAGE)]
            )
            listed = collections.get("collections", [])
            reported = collections.get("total")
            if reported is not None and reported > len(listed):
                raise ConfigurationException(
                    f"Database {self.config.database_id!r} reports {reported} "
                    f"collections but only {len(listed)} were listed; this preflight "
                    f"cannot confirm a container's absence from a partial read."
                )
            known = {c.get("$id") for c in listed} | {c.get("name") for c in listed}
        except AppwriteException as e:
            raise ConfigurationException(
                f"Cannot verify the Appwrite metadata collection in database "
                f"'{self.config.database_id}': {e.message} (type={e.type}). "
                f"Check the coordinates and that this key may read the database."
            ) from e

        if coll_id not in known and self.config.collection_name not in known:
            raise ConfigurationException(
                f"Appwrite metadata collection '{coll_id}' does not exist in database "
                f"'{self.config.database_id}'. Create it deliberately:\n"
                f"    python -m views_pipeline_core.modules.appwrite.provisioning "
                f"ensure-collection"
            )

        self._containers_verified = True

    def _setup_cache(self) -> CacheManager:
        """Initialize the local file cache manager.

        Creates cache directory based on config or path_manager settings.
        Falls back to a default directory if setup fails.

        Returns:
            Configured CacheManager instance.
        """
        try:
            if not self.config.cache_dir:
                cache_dir = getattr(self.config.path_manager, "cache", Path(".")) / "appwrite_cache"
            else:
                cache_dir = Path(self.config.cache_dir)
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_ttl = timedelta(hours=self.config.cache_ttl_hours)
            
            return CacheManager(cache_dir, cache_ttl)
        
        except Exception as e:
            logger.warning(f"Cache setup failed: {e}. Using default cache directory.")
            cache_dir = Path(".appwrite_cache")
            cache_dir.mkdir(exist_ok=True)
            return CacheManager(cache_dir, timedelta(hours=DEFAULT_CACHE_TTL_HOURS))

    def _calculate_file_hash(self, file_path: str = None, file_bytes: bytes = None) -> str:
        """Calculate SHA-256 hash of a file for deduplication.

        Args:
            file_path: Path to file on disk. Reads in 4KB chunks.
            file_bytes: Raw file bytes. Use for in-memory data.

        Returns:
            Hexadecimal SHA-256 hash string.

        Raises:
            ValueError: If neither file_path nor file_bytes is provided.

        Example:
            >>> hash1 = manager._calculate_file_hash(file_path="/data/file.parquet")
            >>> hash2 = manager._calculate_file_hash(file_bytes=b"file content")
        """
        sha256_hash = hashlib.sha256()
        
        if file_path:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        elif file_bytes:
            sha256_hash.update(file_bytes)
        else:
            raise ValueError("Either file_path or file_bytes must be provided")
        
        return sha256_hash.hexdigest()
    
    def _file_exists_by_hash(
    self,
    bucket_id: str,
    file_hash: str,
    filename: str = None
) -> OperationResult:
        """Check if a file exists by hash or filename.

        First checks metadata for matching hash, then falls back to
        filename search in storage. Used for deduplication during uploads.

        Args:
            bucket_id: Storage bucket to search in.
            file_hash: SHA-256 hash to search for.
            filename: Optional filename to search as fallback.

        Returns:
            OperationResult with:
                - success=True, code='FOUND_BY_HASH' if hash matches
                - success=True, code='FOUND_BY_NAME' if filename matches
                - success=False, code='NOT_FOUND' if no match
        """
        try:
            # First try to find by hash in metadata
            search_result = self.metadata_manager.check_file_exists_by_hash(
                file_hash,
                self.config.collection_name,
                self.config.collection_id,
                self.config.database_id
            )

            if search_result.success:
                return OperationResult(
                    success=True,
                    data=search_result.data,
                    code="FOUND_BY_HASH"
                )

            # The lookup did not find a match — but "found nothing" and "could not
            # look" are different answers, and only the first means there is no
            # duplicate. Reporting a failed lookup as NOT_FOUND makes the caller
            # upload a second copy of a file that is already there: a read fault
            # turned into a write (register C-232). Propagate the failure instead.
            if search_result.code != "NOT_FOUND":
                return OperationResult(
                    success=False,
                    error=(
                        f"Could not determine whether a duplicate exists: "
                        f"{search_result.error}"
                    ),
                    code=search_result.code,
                )

            # Fallback to filename check if hash not found - but use efficient query
            if filename:
                try:
                    # Use query instead of listing all files
                    result = self.storage.list_files(
                        bucket_id, 
                        [Query.equal("name", filename), Query.limit(1)]
                    )
                    
                    files = result.get("files", [])
                    if files:
                        return OperationResult(
                            success=True,
                            data=files[0],
                            code="FOUND_BY_NAME"
                        )
                except AppwriteException as query_error:
                    logger.warning(f"Filename query failed, falling back to list: {query_error}")
                    # Fallback to original list-based approach if query fails
                    all_files = []
                    offset = 0
                    limit = DEFAULT_PAGE_LIMIT
                    
                    while True:
                        result = self.storage.list_files(
                            bucket_id, [Query.limit(limit), Query.offset(offset)]
                        )
                        files_chunk = result.get("files", [])
                        all_files.extend(files_chunk)
                        
                        if len(files_chunk) < limit:
                            break
                        offset += limit
                    
                    for file in all_files:
                        if file["name"] == filename:
                            return OperationResult(
                                success=True,
                                data=file,
                                code="FOUND_BY_NAME"
                            )
            
            return OperationResult(success=False, code="NOT_FOUND")
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )

    def _build_metadata_document(
        self,
        file_id: str,
        bucket_id: str,
        filename: str,
        upload_result: Dict[str, Any],
        metadata: Dict[str, Any],
        file_hash: str = None
    ) -> Dict[str, Any]:
        """Build a metadata document for database storage.

        Combines fixed fields (fileId, bucketId, etc.) with custom metadata
        to create a complete document for the metadata collection.

        Args:
            file_id: Appwrite file ID.
            bucket_id: Storage bucket ID.
            filename: Original filename.
            upload_result: Result from file upload containing size info.
            metadata: Custom metadata fields to include.
            file_hash: Optional SHA-256 hash of file contents.

        Returns:
            Dictionary suitable for storing in Appwrite database.
            None values are filtered out.
        """
        base_document = {
            "fileId": file_id,
            "bucketId": bucket_id,
            "filename": filename,
            "mime_type": metadata.get("mime_type", "application/octet-stream"),
            "uploaded_at": datetime.now().isoformat(),
            "file_hash": file_hash,
            **metadata
        }
        
        if "data" in upload_result and "sizeOriginal" in upload_result["data"]:
            base_document["file_size"] = upload_result["data"]["sizeOriginal"]
        
        return {k: v for k, v in base_document.items() if v is not None}

    def _store_metadata_document(
        self,
        database_id: str,
        collection_id: str,
        file_id: str,
        metadata_document: Dict[str, Any]
    ) -> OperationResult:
        """Store or update a metadata document in the database.

        Creates a new document if none exists for the file_id, otherwise
        updates the existing document.

        Args:
            database_id: Target database identifier.
            collection_id: Target collection identifier.
            file_id: File ID to associate metadata with.
            metadata_document: Complete metadata document to store.

        Returns:
            OperationResult with code 'CREATED' or 'UPDATED' on success.
        """
        try:
            # Existence check plus documents[0]; one row is all that is consumed.
            existing_docs = self.databases.list_documents(
                database_id,
                collection_id,
                queries=[Query.equal("fileId", file_id), Query.limit(1)],
            )
            
            if existing_docs["total"] > 0:
                doc_id = existing_docs["documents"][0]["$id"]
                result = self.databases.update_document(
                    database_id, collection_id, doc_id, metadata_document
                )
                return OperationResult(success=True, data=result, code="UPDATED")
            else:
                result = self.databases.create_document(
                    database_id, collection_id, ID.unique(), metadata_document
                )
                return OperationResult(success=True, data=result, code="CREATED")
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )

    def upload_file(
        self,
        bucket_id: str,
        file_path: str,
        file_id: str = None,
        permissions: List[str] = None,
        check_duplicates: bool = True,
        overwrite: bool = False
    ) -> OperationResult:
        """Upload a file from disk to Appwrite storage.

        Uploads a file with optional duplicate checking and overwrite support.
        Does NOT store metadata - use upload_file_with_metadata for that.

        Args:
            bucket_id: Target storage bucket ID.
            file_path: Local path to the file to upload.
            file_id: Optional custom file ID. Auto-generated if not provided.
            permissions: Optional list of Appwrite permission strings.
            check_duplicates: Whether to check for existing files by hash/name.
                Defaults to True.
            overwrite: If True and duplicate found, delete existing and upload.
                If False and duplicate found, return existing file info.
                Defaults to False.

        Returns:
            OperationResult with:
                - success=True, code='CREATED' and file data on new upload
                - success=True, code='EXISTS' and existing file data if duplicate
                - success=False with error details on failure

        Example:
            >>> result = manager.upload_file(
            ...     bucket_id="my_bucket",
            ...     file_path="/data/output.parquet",
            ...     check_duplicates=True,
            ...     overwrite=False
            ... )
            >>> if result.success:
            ...     file_id = result.data['$id']
        """
        try:
            filename = Path(file_path).name
            file_hash = None
            
            if check_duplicates:
                file_hash = self._calculate_file_hash(file_path=file_path)
                duplicate_check = self._file_exists_by_hash(bucket_id, file_hash, filename)
                
                if duplicate_check.success:
                    existing_file = duplicate_check.data
                    
                    if overwrite:
                        delete_result = self.delete_file(bucket_id, existing_file["$id"])
                        if not delete_result.success:
                            return delete_result
                    else:
                        return OperationResult(
                            success=True,
                            data=existing_file,
                            code="EXISTS"
                        )
            
            file_id = file_id or ID.unique()
            permissions = permissions or []
            
            input_file = InputFile.from_path(file_path)
            result = self.storage.create_file(
                bucket_id=bucket_id,
                file_id=file_id,
                file=input_file,
                permissions=permissions
            )
            
            return OperationResult(
                success=True,
                data=result,
                code="CREATED"
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"Upload failed: {e.message}",
                code=e.type
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                code="UNKNOWN_ERROR"
            )

    def upload_file_from_bytes(
        self,
        bucket_id: str,
        file_bytes: bytes,
        filename: str,
        file_id: str = None,
        permissions: List[str] = None,
        check_duplicates: bool = True,
        overwrite: bool = False
    ) -> OperationResult:
        """Upload a file from bytes to Appwrite storage.

        Uploads in-memory file data with optional duplicate checking.
        Does NOT store metadata - use upload_file_from_bytes_with_metadata.

        Args:
            bucket_id: Target storage bucket ID.
            file_bytes: Raw file content as bytes.
            filename: Name to give the file in storage.
            file_id: Optional custom file ID. Auto-generated if not provided.
            permissions: Optional list of Appwrite permission strings.
            check_duplicates: Whether to check for existing files by hash/name.
            overwrite: If True and duplicate found, delete and re-upload.

        Returns:
            OperationResult with file data on success.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"col": [1, 2, 3]})
            >>> parquet_bytes = df.to_parquet()
            >>> result = manager.upload_file_from_bytes(
            ...     bucket_id="my_bucket",
            ...     file_bytes=parquet_bytes,
            ...     filename="data.parquet"
            ... )
        """
        try:
            file_hash = None
            
            if check_duplicates:
                file_hash = self._calculate_file_hash(file_bytes=file_bytes)
                duplicate_check = self._file_exists_by_hash(bucket_id, file_hash, filename)
                
                if duplicate_check.success:
                    existing_file = duplicate_check.data
                    
                    if overwrite:
                        delete_result = self.delete_file(bucket_id, existing_file["$id"])
                        if not delete_result.success:
                            return delete_result
                    else:
                        return OperationResult(
                            success=True,
                            data=existing_file,
                            code="EXISTS"
                        )
            
            file_id = file_id or ID.unique()
            permissions = permissions or []
            
            input_file = InputFile.from_bytes(file_bytes, filename=filename)
            result = self.storage.create_file(
                bucket_id=bucket_id,
                file_id=file_id,
                file=input_file,
                permissions=permissions
            )
            
            return OperationResult(
                success=True,
                data=result,
                code="CREATED"
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"Upload from bytes failed: {e.message}",
                code=e.type
            )

    def upload_file_with_metadata(
    self,
    bucket_id: str,
    file_path: str,
    filename: str,
    metadata: Dict[str, Any],
    file_id: str = None,
    permissions: List[str] = None,
    collection_name: str = None,
    collection_id: str = None
) -> OperationResult:
        """Upload a file and store metadata in the database.

        Complete upload workflow that:
        1. Calculates file hash for deduplication
        2. Checks for existing files by hash or name
        3. Handles existing files (update metadata or delete old version)
        4. Uploads the file to storage
        5. Stores metadata in the database collection

        Args:
            bucket_id: Target storage bucket ID.
            file_path: Local path to the file to upload.
            filename: Name to give the file in storage.
            metadata: Custom metadata dict to store with the file.
            file_id: Optional custom file ID.
            permissions: Optional Appwrite permission strings.
            collection_name: Metadata collection name. Defaults to config.
            collection_id: Metadata collection ID. Defaults to config.

        Returns:
            OperationResult with:
                - success=True, code='UPLOAD_SUCCESS' with file_id, document_id, metadata
                - success=True, code='METADATA_UPDATED' if only metadata was updated
                - success=False with error details on failure

        Example:
            >>> result = manager.upload_file_with_metadata(
            ...     bucket_id="forecasts",
            ...     file_path="/output/predictions.parquet",
            ...     filename="predictions_202401.parquet",
            ...     metadata={
            ...         "model": "ensemble_v2",
            ...         "loa": "pgm",
            ...         "targets": ["ged_sb", "ged_ns"]
            ...     }
            ... )
            >>> if result.success:
            ...     print(f"File ID: {result.data['file_id']}")
            ...     print(f"Document ID: {result.data['document_id']}")
        """
        # Use defaults from config if not provided
        if collection_name is None:
            collection_name = self.config.collection_name
        if collection_id is None:
            collection_id = self.config.collection_id

        # Calculate file hash for metadata
        file_hash = self._calculate_file_hash(file_path=file_path)

        # Check if file already exists by hash in metadata
        existing_metadata = self.metadata_manager.check_file_exists_by_hash(
            file_hash, collection_name, collection_id, self.config.database_id
        )

        # Verify the file exists in BOTH metadata and storage before treating the
        # metadata document as an orphan. See ``_classify_storage_presence``: a read
        # that FAILED is not evidence of absence, and only evidence of absence may
        # authorise a delete (register C-231, þing-02 #329).
        should_update_metadata_only = False
        if existing_metadata.success and existing_metadata.code == "FOUND_BY_HASH" and not file_id:
            existing_file_id = existing_metadata.data.get("fileId")

            if existing_file_id:
                file_check = self.get_file(bucket_id, existing_file_id)
                presence = _classify_storage_presence(file_check)

                if presence is _StoragePresence.PRESENT:
                    should_update_metadata_only = self.config.allow_metadata_only_updates
                    logger.info(f"File {existing_file_id} exists in both metadata and storage")

                elif presence is _StoragePresence.ABSENT:
                    # Appwrite positively confirmed the file is gone: the metadata
                    # document really is an orphan of a failed upload.
                    logger.warning(f"File {existing_file_id} found in metadata but missing from storage, will re-upload")
                    existing_doc_id = existing_metadata.data.get("$id")
                    if existing_doc_id:
                        try:
                            self.databases.delete_document(
                                database_id=self.config.database_id,
                                collection_id=collection_id,
                                document_id=existing_doc_id
                            )
                            logger.info(f"Deleted orphaned metadata document: {existing_doc_id}")
                        except AppwriteException as e:
                            return OperationResult(
                                success=False,
                                error=(
                                    f"Could not delete the orphaned metadata document "
                                    f"{existing_doc_id}: {e.message}"
                                ),
                                code=e.type,
                            )

                else:
                    # INDETERMINATE: wrong bucket id, no read scope on the bucket, or an
                    # untyped transport error. The file may well exist. Deleting its
                    # metadata document here would make a live forecast unfindable, so
                    # refuse the whole upload and say which distinction failed.
                    return OperationResult(
                        success=False,
                        error=(
                            f"Cannot determine whether file {existing_file_id} exists in "
                            f"bucket '{bucket_id}': {file_check.error}. Refusing to treat "
                            f"an unreadable file as an absent one — check the bucket id "
                            f"and that this key holds read scope on it."
                        ),
                        code=file_check.code,
                    )

        if should_update_metadata_only:
            logger.info(f"File with hash {file_hash} already exists, updating metadata only")

            # Get existing document ID
            existing_doc_id = existing_metadata.data.get("$id")
            existing_file_id = existing_metadata.data.get("fileId")

            if not existing_doc_id:
                logger.warning("Existing metadata found but no document ID available")
                # Fall through to normal upload
            else:
                # Update the metadata document
                updated_metadata = {**metadata, "file_hash": file_hash}

                update_result = self.metadata_manager.update_file_metadata(
                    file_id=existing_file_id,
                    metadata_updates=updated_metadata,
                    collection_name=collection_name,
                    collection_id=collection_id,
                    database_id=self.config.database_id
                )

                if update_result.success:
                    return OperationResult(
                        success=True,
                        data={
                            "file_id": existing_file_id,
                            "document_id": existing_doc_id,
                            "metadata": updated_metadata,
                            "message": "Metadata updated for existing file"
                        },
                        code="METADATA_UPDATED"
                    )
                else:
                    logger.warning(f"Failed to update metadata: {update_result.error}")
                    # Fall through to normal upload

        # CRITICAL: If file exists by NAME but different hash, DELETE the old one
        if existing_metadata.success and existing_metadata.code == "FOUND_BY_NAME":
            logger.info(f"File '{filename}' exists with different hash, deleting old version")
            old_file_id = existing_metadata.data.get("fileId")
            old_doc_id = existing_metadata.data.get("$id")

            if old_file_id:
                # Delete the old file from storage. If this fails we must NOT go on to
                # delete its metadata document: that would leave the old file in the
                # bucket with nothing pointing at it — unfindable, and indistinguishable
                # from a month that never ran. Same rule as the de-dup path above:
                # a failed operation is not a completed one (register C-231, þing-02 #329).
                delete_result = self.delete_file(bucket_id, old_file_id)
                if not delete_result.success:
                    presence = _classify_storage_presence(delete_result)
                    if presence is not _StoragePresence.ABSENT:
                        return OperationResult(
                            success=False,
                            error=(
                                f"Refusing to replace '{filename}': the old file "
                                f"{old_file_id} could not be removed from bucket "
                                f"'{bucket_id}' ({delete_result.error}), and deleting "
                                f"its metadata would orphan it."
                            ),
                            code=delete_result.code,
                        )
                    # Already gone from storage — the metadata is genuinely stale.
                    logger.info(f"Old file {old_file_id} was already absent from storage")

            if old_doc_id:
                # Delete the old metadata document
                try:
                    self.databases.delete_document(
                        database_id=self.config.database_id,
                        collection_id=collection_id,
                        document_id=old_doc_id
                    )
                    logger.info(f"Deleted old metadata document: {old_doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete old metadata: {str(e)}")

        # Containers must already exist — provisioning is a deliberate act (#331).
        # Verified BEFORE the upload below, so a missing collection cannot leave an
        # orphaned file in the bucket.
        self._require_containers(bucket_id, collection_id)

        # Add file_hash to metadata
        metadata["file_hash"] = file_hash

        # Upload file - DISABLE duplicate checking since we already handled it above
        upload_result = self.upload_file(
            bucket_id, 
            file_path, 
            file_id, 
            permissions, 
            check_duplicates=False,  # Don't check again - we already handled it
            overwrite=False
        )

        if not upload_result.success:
            return OperationResult(
                success=False,
                error=upload_result.error,
                code=upload_result.code
            )

        # Get the uploaded file ID
        uploaded_file_id = upload_result.data.get("$id")
        
        # Get database and collection IDs from the collection result
        database_id = self.config.database_id
        coll_id = collection_id or self.config.collection_id

        # Prepare metadata with file reference
        metadata_with_file_ref = {
            **metadata,
            "fileId": uploaded_file_id,
            "filename": filename,
            "bucketId": bucket_id,
            "uploaded_at": datetime.now().isoformat()
        }

        # Store metadata in database using _store_metadata_document
        metadata_result = self._store_metadata_document(
            database_id=database_id,
            collection_id=coll_id,
            file_id=uploaded_file_id,
            metadata_document=metadata_with_file_ref
        )

        if not metadata_result.success:
            # Metadata storage failed, but file was uploaded
            logger.error(f"File uploaded but metadata storage failed: {metadata_result.error}")
            return OperationResult(
                success=False,
                error=f"File uploaded but metadata storage failed: {metadata_result.error}",
                data={
                    "file_id": uploaded_file_id,
                    "file_data": upload_result.data
                },
                code="PARTIAL_SUCCESS"
            )

        # Success - both file and metadata stored
        return OperationResult(
            success=True,
            data={
                "file_id": uploaded_file_id,
                "document_id": metadata_result.data.get("$id"),
                "file_data": upload_result.data,
                "metadata": metadata_with_file_ref
            },
            code="UPLOAD_SUCCESS"
        )

    def upload_file_from_bytes_with_metadata(
        self,
        bucket_id: str,
        file_bytes: bytes,
        filename: str,
        metadata: Dict[str, Any],
        file_id: str = None,
        permissions: List[str] = None,
        collection_name: str = None,
        collection_id: str = None
    ) -> OperationResult:
        """Upload file bytes and store metadata in the database.

        Same as upload_file_with_metadata but accepts raw bytes instead
        of a file path. Useful for uploading in-memory data.

        Args:
            bucket_id: Target storage bucket ID.
            file_bytes: Raw file content as bytes.
            filename: Name to give the file in storage.
            metadata: Custom metadata dict to store with the file.
            file_id: Optional custom file ID.
            permissions: Optional Appwrite permission strings.
            collection_name: Metadata collection name. Defaults to config.
            collection_id: Metadata collection ID. Defaults to config.

        Returns:
            OperationResult with:
                - success=True, code='CREATED_WITH_METADATA' on new upload
                - success=True, code='EXISTS_METADATA_UPDATED' if metadata updated
                - success=False with error on failure

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"predictions": [0.1, 0.5, 0.9]})
            >>> result = manager.upload_file_from_bytes_with_metadata(
            ...     bucket_id="forecasts",
            ...     file_bytes=df.to_parquet(),
            ...     filename="predictions.parquet",
            ...     metadata={"model": "test", "loa": "pgm"}
            ... )
        """
        # Use defaults from config if not provided
        if collection_name is None:
            collection_name = self.config.collection_name
        if collection_id is None:
            collection_id = self.config.collection_id
        
        # Calculate file hash for metadata
        file_hash = self._calculate_file_hash(file_bytes=file_bytes)
        
        # Check if file already exists by hash
        existing_metadata = self.metadata_manager.check_file_exists_by_hash(
            file_hash, collection_name, collection_id, self.config.database_id
        )
        
        # Use same logic as upload_file_with_metadata for consistency
        should_update_metadata_only = (existing_metadata.success and 
                                    not file_id and 
                                    self.config.allow_metadata_only_updates)
        
        if should_update_metadata_only:
            logger.info(f"File with hash {file_hash} already exists, updating metadata only")
            
            existing_file_id = existing_metadata.data.get("fileId")
            
            # Containers must already exist — provisioning is deliberate (#331).
            self._require_containers(bucket_id, collection_id)


            # Update the metadata
            metadata_update = metadata.copy()
            metadata_update["file_hash"] = file_hash
            metadata_update["filename"] = filename
            metadata_update["uploaded_at"] = datetime.now().isoformat()
            
            update_result = self.metadata_manager.update_file_metadata(
                file_id=existing_file_id,
                metadata_updates=metadata_update,
                collection_name=collection_name,
                collection_id=collection_id,
                database_id=self.config.database_id
            )
            
            if update_result.success:
                # Get the full file info to return
                file_info = self.get_file(bucket_id, existing_file_id)
                return OperationResult(
                    success=True,
                    data={
                        **(file_info.data if file_info.success else {}),
                        "metadata": update_result.data,
                        "metadata_action": "UPDATED"
                    },
                    code="EXISTS_METADATA_UPDATED"
                )
            else:
                return OperationResult(
                    success=False,
                    error=f"Failed to update metadata: {update_result.error}",
                    code="METADATA_UPDATE_FAILED"
                )
        
        # Ensure metadata infrastructure exists
        self._require_containers(bucket_id, collection_id)

        # Add file_hash to metadata
        metadata["file_hash"] = file_hash
        
        # Upload file (this will handle duplicates based on check_duplicates parameter)
        upload_result = self.upload_file_from_bytes(
            bucket_id, 
            file_bytes, 
            filename, 
            file_id, 
            permissions, 
            check_duplicates=True,  # Let the base method handle duplicates
            overwrite=False  # Don't overwrite by default in metadata flow
        )
        
        if not upload_result.success:
            return upload_result
        
        file_id = upload_result.data["$id"]
        database_id = self.config.database_id
        coll_id = collection_id or self.config.collection_id
        
        # Create and store metadata
        try:
            metadata_document = self._build_metadata_document(
                file_id, bucket_id, filename, {"data": upload_result.data}, metadata, file_hash
            )
            
            metadata_result = self._store_metadata_document(
                database_id, coll_id, file_id, metadata_document
            )
            
            if metadata_result.success:
                upload_result.data["metadata"] = metadata_result.data
                upload_result.data["metadata_action"] = metadata_result.code
            
            return OperationResult(
                success=True,
                data=upload_result.data,
                code="CREATED_WITH_METADATA"
            )
        
        except AppwriteException as e:
            logger.error(f"Metadata handling failed: {e.message}")
            # Rollback: delete the uploaded file if metadata fails. `delete_file`
            # reports failure by RETURN VALUE, so the `except` below never sees one —
            # inspect the result, or a failed rollback leaves an orphaned file in the
            # bucket and says nothing (the fourth route to the state reconcile.py
            # exists to detect; register C-227's disease at this site).
            try:
                rollback = self.delete_file(bucket_id, file_id)
                if not rollback.success:
                    logger.error(
                        "Rollback FAILED: file %s remains in bucket '%s' with no "
                        "metadata document (code=%s, %s). Run `python -m "
                        "views_pipeline_core.modules.appwrite.reconcile` to confirm.",
                        file_id, bucket_id, rollback.code, rollback.error,
                    )
            except Exception as delete_error:
                logger.error(f"Failed to rollback file upload after metadata error: {delete_error}")

            return OperationResult(
                success=False,
                error=f"Metadata handling failed: {e.message}",
                code="METADATA_ERROR"
            )

    def download_file(
        self,
        bucket_id: str,
        file_id: str,
        save_path: str = None,
        use_cache: bool = True,
        validate_cache: bool = True
    ) -> OperationResult:
        """Download a file from Appwrite storage with caching.

        Downloads a file either from local cache (if valid) or from remote
        storage. Automatically updates cache after remote downloads.

        Args:
            bucket_id: Storage bucket containing the file.
            file_id: ID of the file to download.
            save_path: Optional path to save file to disk. If None, returns
                bytes in result data.
            use_cache: Whether to use cached file if available. Defaults to True.
            validate_cache: Whether to validate cache against remote timestamps.
                Defaults to True.

        Returns:
            OperationResult with:
                - success=True and data containing either:
                    - 'save_path' and 'from_cache' if save_path was provided
                    - 'file_bytes' and 'from_cache' if no save_path
                - code indicating source: 'SAVED_FROM_CACHE', 'RETURNED_FROM_CACHE',
                  'SAVED_FROM_REMOTE', or 'RETURNED_FROM_REMOTE'

        Example:
            >>> # Download to file
            >>> result = manager.download_file(
            ...     bucket_id="forecasts",
            ...     file_id="abc123",
            ...     save_path="/tmp/output.parquet"
            ... )
            >>>
            >>> # Download to memory
            >>> result = manager.download_file("forecasts", "abc123")
            >>> if result.success:
            ...     data = result.data['file_bytes']
            ...     print(f"From cache: {result.data['from_cache']}")
        """
        try:
            # Get file metadata for cache validation
            file_metadata = None
            if validate_cache or use_cache:
                file_info = self.get_file(bucket_id, file_id)
                if file_info.success:
                    file_metadata = file_info.data
            
            # Check cache if enabled
            if use_cache:
                remote_updated = file_metadata.get("$updatedAt") if file_metadata else None
                cache_validation = self.cache_manager.validate_cache(bucket_id, file_id, remote_updated)
                
                if cache_validation == CacheValidationResult.VALID:
                    cache_result = self.cache_manager.get_cached_file_path(bucket_id, file_id)
                    if cache_result.success:
                        cache_path = Path(cache_result.data["cache_path"])
                        
                        if save_path:
                            shutil.copy2(cache_path, save_path)
                            return OperationResult(
                                success=True,
                                data={"save_path": save_path, "from_cache": True},
                                code="SAVED_FROM_CACHE"
                            )
                        else:
                            with open(cache_path, "rb") as f:
                                file_bytes = f.read()
                            
                            return OperationResult(
                                success=True,
                                data={"file_bytes": file_bytes, "from_cache": True},
                                code="RETURNED_FROM_CACHE"
                            )
            
            # Download from remote
            file_bytes = self.storage.get_file_download(bucket_id, file_id)
            # The Appwrite SDK's Client.call() returns a PARSED DICT (response.json())
            # for any file served with Content-Type: application/json — e.g. ADR-013
            # wire *.json manifests — despite get_file_download's `-> bytes` annotation
            # (#310). Every downstream use here expects raw bytes, so coerce JSON
            # responses back to bytes. NOTE (byte-fidelity, register C-217): the
            # re-serialization is compact JSON — NOT byte-identical to the stored
            # artifact (e.g. the publisher writes manifests indent=2 + newline).
            # Semantically equivalent for every parser; never byte-compare a JSON
            # download against its shelf artifact.
            if isinstance(file_bytes, dict):
                logger.warning(
                    "download_file: JSON payload for file_id=%s arrived SDK-parsed; "
                    "re-serialized to bytes (compact form — not byte-identical to "
                    "the stored artifact, see #310/C-217).",
                    file_id,
                )
                file_bytes = json.dumps(file_bytes).encode("utf-8")

            # Determine filename for caching
            filename = file_metadata.get("name", file_id) if file_metadata else file_id
            cache_path = self.cache_manager._get_cache_path(bucket_id, file_id, filename)
            
            # Save to cache
            with open(cache_path, "wb") as f:
                f.write(file_bytes)
            
            self.cache_manager.add_to_cache(bucket_id, file_id, cache_path, file_metadata)
            
            # Handle save_path
            if save_path:
                shutil.copy2(cache_path, save_path)
                return OperationResult(
                    success=True,
                    data={"save_path": save_path, "from_cache": False},
                    code="SAVED_FROM_REMOTE"
                )
            else:
                return OperationResult(
                    success=True,
                    data={"file_bytes": file_bytes, "from_cache": False},
                    code="RETURNED_FROM_REMOTE"
                )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"Download failed: {e.message}",
                code=e.type
            )
        except IOError as e:
            return OperationResult(
                success=False,
                error=f"File operation failed: {str(e)}",
                code="IO_ERROR"
            )

    def list_files(
        self,
        bucket_id: str,
        queries: List[str] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
        order_field: str = None,
        order_type: str = "ASC"
    ) -> OperationResult:
        """List files in a storage bucket with optional filtering.

        Args:
            bucket_id: Storage bucket to list files from.
            queries: Optional list of Appwrite Query objects for filtering.
            limit: Maximum number of files to return. Defaults to 100.
            offset: Number of files to skip for pagination.
            order_field: Field name to sort by (e.g., '$createdAt').
            order_type: Sort direction, 'ASC' or 'DESC'. Defaults to 'ASC'.

        Returns:
            OperationResult with data containing:
                - 'files': List of file objects
                - 'total': Total count of matching files

        Example:
            >>> result = manager.list_files(
            ...     bucket_id="forecasts",
            ...     limit=50,
            ...     order_field="$createdAt",
            ...     order_type="DESC"
            ... )
            >>> for file in result.data['files']:
            ...     print(f"{file['name']}: {file['$id']}")
        """
        try:
            if queries is None:
                queries = []
            
            query_list = queries.copy()
            query_list.append(Query.limit(limit))
            query_list.append(Query.offset(offset))
            
            if order_field:
                if order_type.upper() == "DESC":
                    query_list.append(Query.order_desc(order_field))
                else:
                    query_list.append(Query.order_asc(order_field))
            
            result = self.storage.list_files(bucket_id, query_list)
            
            return OperationResult(
                success=True,
                data={
                    "files": result.get("files", []),
                    "total": result.get("total", 0)
                }
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"List files failed: {e.message}",
                code=e.type
            )

    def delete_file(self, bucket_id: str, file_id: str) -> OperationResult:
        """Delete a file from Appwrite storage.

        Removes the file from both remote storage and local cache.
        Note: Does not delete associated metadata documents.

        Args:
            bucket_id: Storage bucket containing the file.
            file_id: ID of the file to delete.

        Returns:
            OperationResult with code='DELETED' on success.

        Example:
            >>> result = manager.delete_file("my_bucket", "file123")
            >>> if result.success:
            ...     print("File deleted")
        """
        try:
            result = self.storage.delete_file(bucket_id, file_id)
            
            # Also remove from cache
            self.cache_manager.remove_from_cache(bucket_id, file_id)
            
            return OperationResult(
                success=True,
                data=result,
                code="DELETED"
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"Delete failed: {e.message}",
                code=e.type
            )

    def get_file(self, bucket_id: str, file_id: str) -> OperationResult:
        """Get file metadata from Appwrite storage.

        Retrieves file information (name, size, timestamps, etc.) without
        downloading the actual file content.

        Args:
            bucket_id: Storage bucket containing the file.
            file_id: ID of the file.

        Returns:
            OperationResult with file metadata in data field on success.

        Example:
            >>> result = manager.get_file("my_bucket", "file123")
            >>> if result.success:
            ...     print(f"Name: {result.data['name']}")
            ...     print(f"Size: {result.data['sizeOriginal']} bytes")
        """
        try:
            result = self.storage.get_file(bucket_id, file_id)
            return OperationResult(success=True, data=result)
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"Get file failed: {e.message}",
                code=e.type
            )

    def get_bucket(self, bucket_id: str) -> OperationResult:
        """Get bucket metadata from Appwrite storage.

        Retrieves bucket information including name, settings, and permissions.

        Args:
            bucket_id: ID of the bucket to retrieve.

        Returns:
            OperationResult with:
                - success=True and bucket data on success
                - success=False, code='storage_bucket_not_found' if not found

        Example:
            >>> result = manager.get_bucket("forecasts")
            >>> if result.success:
            ...     print(f"Bucket: {result.data['name']}")
        """
        try:
            result = self.storage.get_bucket(bucket_id)
            return OperationResult(success=True, data=result)
        
        except AppwriteException as e:
            # Check if this is a bucket not found error
            if "Storage bucket with the requested ID could not be found" in e.message:
                return OperationResult(
                    success=False,
                    error=e.message,
                    code="storage_bucket_not_found"
                )
            
            return OperationResult(
                success=False,
                error=f"Get bucket failed: {e.message}",
                code=e.type
            )

    def list_buckets(
        self,
        search: str = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0
    ) -> OperationResult:
        """List storage buckets with optional search.

        Args:
            search: Optional search string to filter buckets by name.
            limit: Maximum number of buckets to return. Defaults to 100.
            offset: Number of buckets to skip for pagination.

        Returns:
            OperationResult with data containing:
                - 'buckets': List of bucket objects
                - 'total': Total count of matching buckets

        Example:
            >>> result = manager.list_buckets(search="forecast")
            >>> for bucket in result.data['buckets']:
            ...     print(f"{bucket['name']}: {bucket['$id']}")
        """
        try:
            queries = []
            if search:
                queries.append(Query.search("name", search))
            queries.extend([Query.limit(limit), Query.offset(offset)])
            
            result = self.storage.list_buckets(queries)
            return OperationResult(
                success=True,
                data={
                    "buckets": result.get("buckets", []),
                    "total": result.get("total", 0)
                }
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"List buckets failed: {e.message}",
                code=e.type
            )


    def get_current_user(self) -> OperationResult:
        """Get current authenticated user information.

        Only available when using session authentication.

        Returns:
            OperationResult with user data including:
                - user_id: User's ID
                - email: User's email address
                - name: User's display name
                - email_verified: Whether email is verified

        Example:
            >>> result = manager.get_current_user()
            >>> if result.success:
            ...     print(f"User: {result.data['email']}")
        """
        if not isinstance(self.auth_manager, SessionAuth):
            return OperationResult(
                success=False,
                error="User information requires session authentication",
                code="AUTH_METHOD_ERROR"
            )
        
        try:
            user = self.auth_manager.account.get()
            return OperationResult(
                success=True,
                data={
                    "user_id": user["$id"],
                    "email": user.get("email"),
                    "name": user.get("name"),
                    "email_verified": user.get("emailVerification", False)
                }
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )

    def get_user_preferences(self, user_id: Optional[str] = None) -> OperationResult:
        """Get user preferences from Appwrite.

        For session auth, gets current user's preferences.
        For API key auth, requires user_id parameter.

        Args:
            user_id: Required for API key auth. Optional for session auth.

        Returns:
            OperationResult with preferences data.
                code='USER_SESSION' or code='API_KEY' indicates auth type used.

        Example:
            >>> # With session auth
            >>> result = manager.get_user_preferences()
            >>>
            >>> # With API key auth
            >>> result = manager.get_user_preferences(user_id="user123")
        """
        try:
            if isinstance(self.auth_manager, SessionAuth):
                preferences = self.auth_manager.account.get_prefs()
                return OperationResult(
                    success=True,
                    data=preferences,
                    code="USER_SESSION"
                )
            else:
                if not user_id:
                    return OperationResult(
                        success=False,
                        error="user_id parameter required for API key authentication",
                        code="MISSING_USER_ID"
                    )
                
                user_prefs = self.users.get_prefs(user_id)
                return OperationResult(
                    success=True,
                    data=user_prefs,
                    code="API_KEY"
                )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )

    def clear_cache(
        self,
        bucket_id: str = None,
        older_than_hours: int = None
    ) -> OperationResult:
        """Clear the local file cache.

        Wrapper around CacheManager.clear_cache() for convenience.

        Args:
            bucket_id: Optional bucket to limit clearing to.
            older_than_hours: Optional age filter for selective clearing.

        Returns:
            OperationResult with deletion statistics.

        Example:
            >>> # Clear all cache
            >>> result = manager.clear_cache()
            >>>
            >>> # Clear cache older than 48 hours
            >>> result = manager.clear_cache(older_than_hours=48)
        """
        return self.cache_manager.clear_cache(bucket_id, older_than_hours)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and usage information.

        Wrapper around CacheManager.get_stats() for convenience.

        Returns:
            Dictionary with cache statistics including total files,
            total size, and breakdown by bucket.

        Example:
            >>> stats = manager.get_cache_stats()
            >>> print(f"Cache: {stats['total_files']} files, {stats['total_size_mb']}MB")
        """
        return self.cache_manager.get_stats()

    def debug_collection_attributes(
        self,
        collection_id: str = None,
        database_id: str = None
    ) -> OperationResult:
        """Debug helper to list all attributes in a metadata collection.

        Useful for troubleshooting schema issues or verifying attribute creation.

        Args:
            collection_id: Collection to inspect. Defaults to config.collection_id.
            database_id: Database containing collection. Defaults to config.database_id.

        Returns:
            OperationResult with list of attributes on success.

        Example:
            >>> result = manager.debug_collection_attributes()
            >>> # Attributes are also logged at INFO level
        """
        db_id = database_id or self.config.database_id
        coll_id = collection_id or self.config.collection_id
        
        if not db_id or not coll_id:
            return OperationResult(
                success=False,
                error="Database ID and collection ID must be provided in config or as parameters",
                code="MISSING_CONFIG"
            )
        try:
            attributes = self.databases.list_attributes(db_id, coll_id)
            logger.info("Existing attributes:")
            for attr in attributes["attributes"]:
                logger.info(f"  - {attr['key']} ({attr['type']})")
            return OperationResult(success=True, data=attributes)
        
        except AppwriteException as e:
            logger.error(f"Error listing attributes: {e.message}")
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )