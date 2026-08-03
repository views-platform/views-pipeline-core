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

from .file import (
    AuthMethod as AuthMethod,
    CacheValidationResult as CacheValidationResult,
    OperationResult as OperationResult,
    FileMetadata as FileMetadata,
    AppwriteConfig as AppwriteConfig,
    AuthManager as AuthManager,
    ApiKeyAuth as ApiKeyAuth,
    AuthFactory as AuthFactory,
    CacheMetadata as CacheMetadata,
    CacheManager as CacheManager,
    AppwriteMetadataHandler as AppwriteMetadataHandler,
    AppWriteFileModule as AppWriteFileModule,
)
