"""Appwrite storage seam. **Requires the optional `appwrite` extra.**

Importing this package without the extra raises with the install command rather than
letting a bare `ModuleNotFoundError: No module named 'appwrite'` surface from six frames
deep inside `file.py` — the `_require_dense_report_consumer` idiom
(`managers/reporting/stage.py`).

The SDK became optional in #345 (register C-253): three repos that never mention
Appwrite were installing it. Nothing on the delivery path imports this package eagerly,
so a consumer only reaches this check if it genuinely asked for Appwrite.
"""

try:
    import appwrite as _appwrite  # noqa: F401
except ImportError as e:  # pragma: no cover - exercised in a subprocess probe
    raise ImportError(
        "views_pipeline_core.modules.appwrite requires the optional 'appwrite' extra, "
        "which is not installed. Install it with:\n"
        "    pip install 'views-pipeline-core[appwrite]'\n"
        "or, with poetry:\n"
        "    poetry install --extras appwrite\n"
        "If you did not intend to use Appwrite, the local and views-forecasts save "
        "paths need no extra — see ADR-047 on destination authority.\n"
        f"Underlying import error: {e}"
    ) from e

from appwrite.client import Client as Client
from appwrite.exception import AppwriteException as AppwriteException
from appwrite.id import ID as ID
from appwrite.input_file import InputFile as InputFile
from appwrite.query import Query as Query
from appwrite.services.account import Account as Account
from appwrite.services.databases import Databases as Databases
from appwrite.services.storage import Storage as Storage
from appwrite.services.users import Users as Users

from .auth import (
    ApiKeyAuth as ApiKeyAuth,
    AuthFactory as AuthFactory,
    AuthManager as AuthManager,
    AuthMethod as AuthMethod,
)
from .cache import (
    CacheManager as CacheManager,
    CacheMetadata as CacheMetadata,
    CacheValidationResult as CacheValidationResult,
)
from .config import (
    APPWRITE_BUCKET_NOT_FOUND as APPWRITE_BUCKET_NOT_FOUND,
    APPWRITE_DEFAULT_PAGE_SIZE as APPWRITE_DEFAULT_PAGE_SIZE,
    APPWRITE_FILE_NOT_FOUND as APPWRITE_FILE_NOT_FOUND,
    AppwriteConfig as AppwriteConfig,
    DEFAULT_CACHE_TTL_HOURS as DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_PAGE_LIMIT as DEFAULT_PAGE_LIMIT,
    FileMetadata as FileMetadata,
    INITIAL_RETRY_DELAY as INITIAL_RETRY_DELAY,
    MAX_ATTRIBUTE_CREATION_RETRIES as MAX_ATTRIBUTE_CREATION_RETRIES,
    MAX_METADATA_PAGES as MAX_METADATA_PAGES,
    OperationResult as OperationResult,
    _CONTAINER_PAGE as _CONTAINER_PAGE,
    _REQUIRED_COORDINATES as _REQUIRED_COORDINATES,
    _StoragePresence as _StoragePresence,
    _classify_storage_presence as _classify_storage_presence,
    exception_message as exception_message,
)
from .metadata import AppwriteMetadataHandler as AppwriteMetadataHandler
from .storage import AppWriteFileModule as AppWriteFileModule
from .transport import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS as DEFAULT_REQUEST_TIMEOUT_SECONDS,
    install_request_timeout as install_request_timeout,
    resolve_timeout_seconds as resolve_timeout_seconds,
)