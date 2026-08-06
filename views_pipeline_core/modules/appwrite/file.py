"""Backward-compat re-export shim.

The classes that used to live in this 2,841-LOC God module have been
split into cohesive sub-modules (M-1 audit decision):

  * :mod:`config`       — AppwriteConfig, OperationResult, FileMetadata, _StoragePresence
    + module-level constants (APPWRITE_DEFAULT_PAGE_SIZE, APPWRITE_FILE_NOT_FOUND, etc.)
    + ``_classify_storage_presence`` helper + ``exception_message`` helper.
  * :mod:`auth`         — AuthMethod, AuthManager, ApiKeyAuth, AuthFactory
  * :mod:`cache`        — CacheValidationResult, CacheMetadata, CacheManager
  * :mod:`metadata`     — AppwriteMetadataHandler
  * :mod:`storage`      — AppWriteFileModule

This shim preserves the historical import path
``from views_pipeline_core.modules.appwrite.file import X`` so existing
callers (including tests, datastore.py, provisioning.py, and the audit
subpackage) continue to work unchanged. The Appwrite SDK classes
(``Client``, ``Storage``, ``Databases``, ``Users``, ``InputFile``,
``ID``, ``AppwriteException``, ``Query``) are also re-exported here
because the original ``file.py`` imported them at module scope and
tests patch them at this path.
"""
# Re-export the Appwrite SDK classes that the original file.py imported at module scope.
# Tests patch these at `views_pipeline_core.modules.appwrite.file.Client` etc.
from appwrite.client import Client  # noqa: F401
from appwrite.services.storage import Storage  # noqa: F401
from appwrite.services.databases import Databases  # noqa: F401
from appwrite.services.users import Users  # noqa: F401
from appwrite.input_file import InputFile  # noqa: F401
from appwrite.id import ID  # noqa: F401
from appwrite.exception import AppwriteException  # noqa: F401
from appwrite.query import Query  # noqa: F401

from views_pipeline_core.modules.appwrite.auth import (  # noqa: F401
    ApiKeyAuth,
    AuthFactory,
    AuthManager,
    AuthMethod,
)
from views_pipeline_core.modules.appwrite.cache import (  # noqa: F401
    CacheManager,
    CacheMetadata,
    CacheValidationResult,
)
from views_pipeline_core.modules.appwrite.config import (  # noqa: F401
    APPWRITE_BUCKET_NOT_FOUND,
    APPWRITE_DEFAULT_PAGE_SIZE,
    APPWRITE_FILE_NOT_FOUND,
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_PAGE_LIMIT,
    INITIAL_RETRY_DELAY,
    MAX_ATTRIBUTE_CREATION_RETRIES,
    MAX_METADATA_PAGES,
    _CONTAINER_PAGE,
    _REQUIRED_COORDINATES,
    _StoragePresence,
    _classify_storage_presence,
    AppwriteConfig,
    FileMetadata,
    OperationResult,
    exception_message,
)
from views_pipeline_core.modules.appwrite.metadata import (  # noqa: F401
    AppwriteMetadataHandler,
)
from views_pipeline_core.modules.appwrite.storage import (  # noqa: F401
    AppWriteFileModule,
)
from views_pipeline_core.modules.appwrite.transport import (  # noqa: F401
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    install_request_timeout,
    resolve_timeout_seconds,
)

__all__ = [
    "APPWRITE_BUCKET_NOT_FOUND",
    "APPWRITE_DEFAULT_PAGE_SIZE",
    "APPWRITE_FILE_NOT_FOUND",
    "AppWriteFileModule",
    "AppwriteConfig",
    "AppwriteMetadataHandler",
    "ApiKeyAuth",
    "AuthFactory",
    "AuthManager",
    "AuthMethod",
    "DEFAULT_CACHE_TTL_HOURS",
    "DEFAULT_PAGE_LIMIT",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "CacheManager",
    "CacheMetadata",
    "CacheValidationResult",
    "FileMetadata",
    "INITIAL_RETRY_DELAY",
    "MAX_ATTRIBUTE_CREATION_RETRIES",
    "MAX_METADATA_PAGES",
    "OperationResult",
    "_CONTAINER_PAGE",
    "_REQUIRED_COORDINATES",
    "_StoragePresence",
    "_classify_storage_presence",
    "install_request_timeout",
    "resolve_timeout_seconds",
]
