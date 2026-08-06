"""Appwrite-related constants.

Centralized per the user's directive: "scan the codebase for all constants
and organise them in one centralised spot."

Sources:
  - modules/appwrite/config.py (constants block)
  - modules/appwrite/transport.py (timeout constants)
  - modules/appwrite/provisioning.py (provisioning constants)
  - modules/appwrite/audit/ (walk, render, targets, timeline constants)
  - configs/prediction_store.py (_ENV_MAP)
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple

# ---------------------------------------------------------------------------
# Cache defaults
# ---------------------------------------------------------------------------
DEFAULT_CACHE_TTL_HOURS: int = 24
DEFAULT_PAGE_LIMIT: int = 100

# ---------------------------------------------------------------------------
# Attribute creation retries
# ---------------------------------------------------------------------------
MAX_ATTRIBUTE_CREATION_RETRIES: int = 3
INITIAL_RETRY_DELAY: float = 1.0

# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------
APPWRITE_DEFAULT_PAGE_SIZE: int = 25
MAX_METADATA_PAGES: int = 1000
_CONTAINER_PAGE: int = 100

# ---------------------------------------------------------------------------
# Appwrite server error type strings (propagated by SDK into AppwriteException.type)
# ---------------------------------------------------------------------------
APPWRITE_FILE_NOT_FOUND: str = "storage_file_not_found"
APPWRITE_BUCKET_NOT_FOUND: str = "storage_bucket_not_found"

# ---------------------------------------------------------------------------
# Required coordinate keys for real storage addressing
# ---------------------------------------------------------------------------
_REQUIRED_COORDINATES: Tuple[str, ...] = (
    "bucket_id",
    "bucket_name",
    "collection_id",
    "collection_name",
    "database_id",
    "database_name",
)

# ---------------------------------------------------------------------------
# Transport timeout
# ---------------------------------------------------------------------------
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 30.0
TIMEOUT_ENV_VAR: str = "APPWRITE_REQUEST_TIMEOUT_SECONDS"
_PROXY_MARKER: str = "_views_pipeline_core_timeout_proxy"

# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------
_PROVISION_PAGE: int = 100
FIXED_METADATA_ATTRIBUTES: List[Dict[str, Any]] = [
    {"key": "file_hash", "type": "string", "size": 64},
    {"key": "model_name", "type": "string"},
    {"key": "target", "type": "string"},
    {"key": "run_type", "type": "string"},
    {"key": "timestamp", "type": "string"},
    {"key": "category", "type": "string"},
    {"key": "description", "type": "string"},
    {"key": "name", "type": "string"},
    {"key": "loa", "type": "string"},
    {"key": "type", "type": "string"},
    {"key": "targets", "type": "string"},
]

# ---------------------------------------------------------------------------
# Audit walk
# ---------------------------------------------------------------------------
PAGE_SIZE: int = 100
MAX_PAGES: int = 1000

# ---------------------------------------------------------------------------
# Audit timeline
# ---------------------------------------------------------------------------
UNDATED_KEY: str = "undated"

# ---------------------------------------------------------------------------
# Audit targets
# ---------------------------------------------------------------------------
TARGETS: Dict[str, Dict[str, Any]] = {
    "forecasts": {
        "bucket_id": "forecasts",
        "collection_id": "file_metadata",
    },
    "unfao": {
        "bucket_id": "un-fao",
        "collection_id": "file_metadata",
    },
}

_SHARED: Dict[str, str] = {
    "endpoint": "https://cloud.appwrite.io/v1",
    "project_id": "6699996669",
    "database_id": "forecasts_metadata",
}

# ---------------------------------------------------------------------------
# Audit render
# ---------------------------------------------------------------------------
_RULE: str = "-" * 68
_MAX_LISTED: int = 25

# ---------------------------------------------------------------------------
# PredictionStoreConfig env-var mapping (moved from configs/prediction_store.py)
# ---------------------------------------------------------------------------
_ENV_MAP: Dict[str, str] = {
    "endpoint": "APPWRITE_ENDPOINT",
    "project_id": "APPWRITE_PROJECT_ID",
    "bucket_id": "APPWRITE_BUCKET_ID",
    "collection_id": "APPWRITE_COLLECTION_ID",
    "database_id": "APPWRITE_DATABASE_ID",
}
