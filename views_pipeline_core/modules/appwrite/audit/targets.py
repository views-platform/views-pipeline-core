"""The two shelves this audit can point at, and how to build a reader for one.

A "shelf" is a **bucket and a collection together**. That pairing is the whole subject
of the audit, so a coordinate set with one half from one shelf and the other half from
another is not a configuration — it is a guaranteed false alarm, and this module's job
is to make it unreachable (C-250).
"""

from __future__ import annotations

import os
from typing import Optional

# The two shelves on this seam address DIFFERENT buckets and DIFFERENT collections.
# Auditing one says nothing about the other, and conflating them is how a scary number
# from the internal shelf gets read as an FAO incident.
TARGETS = {
    # internal shelf — what pipeline-core writes
    "forecasts": {
        "bucket_id": "APPWRITE_PROD_FORECASTS_BUCKET_ID",
        "bucket_name": "APPWRITE_PROD_FORECASTS_BUCKET_NAME",
        "collection_id": "APPWRITE_PROD_FORECASTS_COLLECTION_ID",
        "collection_name": "APPWRITE_PROD_FORECASTS_COLLECTION_NAME",
    },
    # FAO's outbound — what views-postprocessing's un_fao delivery writes
    "unfao": {
        "bucket_id": "APPWRITE_UNFAO_BUCKET_ID",
        "bucket_name": "APPWRITE_UNFAO_BUCKET_NAME",
        "collection_id": "APPWRITE_UNFAO_COLLECTION_ID",
        "collection_name": "APPWRITE_UNFAO_COLLECTION_NAME",
    },
}

_SHARED = {
    "endpoint": "APPWRITE_ENDPOINT",
    "project_id": "APPWRITE_DATASTORE_PROJECT_ID",
    "credentials": "APPWRITE_DATASTORE_API_KEY",
    "database_id": "APPWRITE_METADATA_DATABASE_ID",
    "database_name": "APPWRITE_METADATA_DATABASE_NAME",
}


def build_file_manager(
    target: str,
    bucket_override: Optional[str] = None,
    collection_override: Optional[str] = None,
):
    """Construct a read-only client for one of the two shelves.

    Connection settings and the shared metadata database come from the environment; the
    bucket and collection come from whichever target is named. Fails loud naming any
    variable that is missing, rather than defaulting to a live address.

    **Overrides come in pairs.** ``--bucket unfao_bucket`` alone used to leave the
    forecasts collection in place, so the audit compared FAO's 130 files against the
    forecasts collection: every file an orphan, verdict PAIRING BROKEN. Because the two
    shelves share one database, nothing errors — the mismatch is valid at the substrate
    level and wrong at every level above it. A flag that reproduces this tool's own
    false alarm on demand is worse than no flag (C-250).
    """
    from views_pipeline_core.exceptions.exceptions import ConfigurationException
    from views_pipeline_core.modules.appwrite.file import AppWriteFileModule, AppwriteConfig

    if bool(bucket_override) != bool(collection_override):
        supplied, missing = (
            ("bucket", "collection") if bucket_override else ("collection", "bucket")
        )
        raise ConfigurationException(
            f"--{supplied} was given without --{missing}. A bucket and its metadata "
            f"collection are one shelf; overriding half of the pair audits one shelf's "
            f"files against another shelf's index, which reports every file as an "
            f"orphan. Supply both, or neither."
        )

    wanted = {**_SHARED, **TARGETS[target]}
    values, missing_vars = {}, []
    for field_name, env_var in wanted.items():
        val = os.getenv(env_var)
        if not val:
            missing_vars.append(env_var)
        values[field_name] = val
    if missing_vars:
        raise ConfigurationException(
            f"Missing environment variable(s) for target {target!r}: {missing_vars}."
        )

    if bucket_override:
        values["bucket_id"] = bucket_override
        values["collection_id"] = collection_override

    return AppWriteFileModule(
        AppwriteConfig(
            path_manager=None,
            auth_method="api_key",
            cache_dir=".appwrite_audit_cache",
            **values,
        )
    )
