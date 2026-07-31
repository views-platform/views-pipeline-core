"""Deliberate provisioning of Appwrite containers. RUNNABLE, NEVER IMPORTABLE.

Creating storage — a bucket, a database, a collection, a collection's attributes — is a
**setup act performed by a person**, not a side effect of publishing a forecast.

Until þing-02 (#331) it was the latter: an ordinary upload created whatever was missing
on its way through, at five call sites, one of them inside a method whose entire job was
to *look something up*. That is why the platform's Appwrite key carries 20 scopes — the
scopes are what the delivery code demands. Least privilege could not be applied while
narrowing the key broke every upload, so this module exists to make the delivery path
read-and-write-only:

    python -m views_pipeline_core.modules.appwrite.provisioning ensure-collection
    python -m views_pipeline_core.modules.appwrite.provisioning ensure-bucket

**The delivery path must never import this module.** That is not a convention — it is
asserted in a subprocess by ``tests/test_import_purity.py`` (#332), which is why the
dependency runs one way only: ``provisioning`` imports ``file``, never the reverse.

Coordinates and credentials come from the process environment via
``PredictionStoreConfig.from_environment()``, which fails loud naming any missing
variable. Nothing is defaulted here.

**Known defect, preserved deliberately.** ``ensure_bucket``'s follow-on database creation
passes its arguments in the wrong order (register C-238). It is reproduced verbatim from
``file.py`` because a relocation must not change behaviour silently; fix it under its own
issue, with a test.

**Known limitation.** ``_create_dynamic_attributes`` derives the collection's schema from
whatever payload it is handed, which contradicts ADR-040 (Authority of Declarations Over
Inference) on a production database. Replacing it with a declared schema is deferred
behind a named trigger: a non-production Appwrite project existing, since nobody has ever
run a first bring-up against an empty one (þing-02 Á-5 / G2(h)).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from appwrite.exception import AppwriteException
from appwrite.permission import Permission
from appwrite.role import Role

from views_pipeline_core.modules.appwrite.file import (
    INITIAL_RETRY_DELAY,
    MAX_ATTRIBUTE_CREATION_RETRIES,
    OperationResult,
)

logger = logging.getLogger(__name__)

# The attributes every metadata document carries regardless of payload. Unlike the
# payload-inferred ones below, these are declared — ADR-040's preferred direction.
FIXED_METADATA_ATTRIBUTES: List[Dict[str, Any]] = [
    {"key": "fileId", "type": "string", "size": 255, "required": True},
    {"key": "bucketId", "type": "string", "size": 255, "required": True},
    {"key": "filename", "type": "string", "size": 500, "required": True},
    {"key": "file_size", "type": "integer", "required": False},
    {"key": "mime_type", "type": "string", "size": 100, "required": False},
    {"key": "uploaded_at", "type": "datetime", "required": False},
    {"key": "file_hash", "type": "string", "size": 64, "required": False},
]


def infer_attribute_type(value: Any) -> Tuple[str, bool]:
    """Infer an Appwrite attribute type from a sample value.

    Inference, and ADR-040 says declarations should win — see the module docstring.
    Retained as-is by the relocation; superseded when the schema is declared.
    """
    is_array = isinstance(value, list)
    base_value = value[0] if is_array and value else value

    if isinstance(base_value, bool):
        return "boolean", is_array
    if isinstance(base_value, int):
        return "integer", is_array
    if isinstance(base_value, float):
        return "double", is_array
    if isinstance(base_value, str):
        try:
            datetime.fromisoformat(base_value.replace("Z", "+00:00"))
            return "datetime", is_array
        except (ValueError, AttributeError):
            return "string", is_array
    return "string", is_array


class AppwriteProvisioner:
    """Creates the containers the delivery path expects to already exist.

    Constructed around a live ``AppWriteFileModule`` so it reuses that module's
    authenticated clients rather than opening a second connection with a second copy
    of the credential.
    """

    def __init__(self, file_manager):
        self._fm = file_manager
        self.databases = file_manager.databases
        self.storage = file_manager.storage
        self.config = file_manager.config

    # -- databases ---------------------------------------------------------------

    def ensure_database(
        self, database_id: str = None, database_name: str = None
    ) -> OperationResult:
        """Create the metadata database if it does not exist."""
        db_id = database_id or self.config.database_id
        db_name = database_name or self.config.database_name

        if not db_id or not db_name:
            return OperationResult(
                success=False,
                error="Database ID and name must be provided in config or as parameters",
                code="MISSING_CONFIG",
            )

        try:
            existing_databases = self.databases.list()

            for db in existing_databases.get("databases", []):
                if db["name"] == db_name or db["$id"] == db_id:
                    logger.info(f"Database '{db_name}' already exists")
                    return OperationResult(success=True, data=db, code="EXISTS")

            try:
                result = self.databases.create(
                    database_id=db_id, name=db_name, enabled=True
                )
                logger.info(f"Created new database: {db_name}")
                return OperationResult(success=True, data=result)
            except AppwriteException as create_error:
                # Hitting the database limit while the database already exists is benign.
                if "maximum number of databases" in create_error.message.lower():
                    existing_databases = self.databases.list()
                    for db in existing_databases.get("databases", []):
                        if db["$id"] == db_id:
                            logger.warning(
                                f"Database limit reached, but database '{db_id}' exists"
                            )
                            return OperationResult(success=True, data=db, code="EXISTS")
                raise create_error

        except AppwriteException as e:
            logger.error(f"Database operation failed: {e.message}")
            return OperationResult(
                success=False,
                error=f"Database operation failed: {e.message}",
                code=e.type,
            )

    # -- collections and attributes ----------------------------------------------

    def ensure_collection(
        self,
        metadata: Dict[str, Any] = None,
        collection_name: str = None,
        collection_id: str = None,
        database_id: str = None,
    ) -> OperationResult:
        """Create the metadata collection (and its database) if they do not exist."""
        db_id = database_id or self.config.database_id
        coll_name = collection_name or self.config.collection_name
        coll_id = collection_id or self.config.collection_id

        if not db_id or not coll_name or not coll_id:
            return OperationResult(
                success=False,
                error=(
                    "Database ID, collection name, and collection ID must be provided "
                    "in config or as parameters"
                ),
                code="MISSING_CONFIG",
            )

        db_result = self.ensure_database(db_id, self.config.database_name)
        if not db_result.success:
            return db_result

        try:
            existing_collections = self.databases.list_collections(db_id)

            for collection in existing_collections.get("collections", []):
                if collection["$id"] == coll_id or collection["name"] == coll_name:
                    if metadata:
                        self.ensure_attributes(db_id, collection["$id"], metadata)
                    return OperationResult(
                        success=True,
                        data={
                            "collection": collection,
                            "database_id": db_id,
                            "collection_id": collection["$id"],
                        },
                        code="EXISTS",
                    )

            result = self.databases.create_collection(
                database_id=db_id,
                collection_id=coll_id,
                name=coll_name,
                permissions=[
                    Permission.read(Role.any()),
                    Permission.create(Role.any()),
                    Permission.update(Role.any()),
                    Permission.delete(Role.any()),
                ],
                document_security=False,
                enabled=True,
            )

            self.ensure_attributes(db_id, coll_id, metadata or {})

            return OperationResult(
                success=True,
                data={
                    "collection": result,
                    "database_id": db_id,
                    "collection_id": result["$id"],
                },
            )

        except AppwriteException as e:
            logger.error(f"Collection creation failed: {e.message}")
            return OperationResult(
                success=False,
                error=f"Collection creation failed: {e.message}",
                code=e.type,
            )

    def ensure_attributes(
        self,
        database_id: str,
        collection_id: str,
        metadata: Dict[str, Any],
        max_retries: int = MAX_ATTRIBUTE_CREATION_RETRIES,
        initial_delay: float = INITIAL_RETRY_DELAY,
    ) -> OperationResult:
        """Create the fixed attributes, plus any inferred from ``metadata``."""
        delay = initial_delay
        existing_attribute_keys = set()

        for attempt in range(1, max_retries + 1):
            try:
                existing_attributes = self.databases.list_attributes(
                    database_id, collection_id
                )
                existing_attribute_keys = {
                    attr["key"] for attr in existing_attributes["attributes"]
                }
                logger.debug(f"Found {len(existing_attribute_keys)} existing attributes")
                break
            except AppwriteException as e:
                # NOTE: matches the human-readable message rather than e.type — same
                # defect class as C-179, preserved by the relocation (register C-235).
                if "collection_not_found" in e.message and attempt < max_retries:
                    logger.warning(
                        f"Collection not ready (attempt {attempt}/{max_retries}), "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Failed to list attributes: {e.message}")
                    return OperationResult(
                        success=False,
                        error=f"Failed to list attributes: {e.message}",
                        code=e.type,
                    )

        for attr in FIXED_METADATA_ATTRIBUTES:
            if attr["key"] in existing_attribute_keys:
                continue
            try:
                self._create_fixed_attribute(database_id, collection_id, attr)
            except AppwriteException as e:
                if "attribute already exists" not in e.message.lower():
                    logger.warning(
                        f"Failed to create fixed attribute {attr['key']}: {e.message}"
                    )

        for key, value in metadata.items():
            if key in existing_attribute_keys:
                continue
            try:
                attr_type, is_array = infer_attribute_type(value)
                self._create_attribute_by_type(
                    database_id, collection_id, key, attr_type, is_array
                )
            except AppwriteException as e:
                logger.error(f"Failed to create attribute '{key}': {e.message}")

        return OperationResult(success=True)

    def _create_fixed_attribute(
        self, database_id: str, collection_id: str, attr: Dict[str, Any]
    ):
        attr_creators = {
            "string": lambda: self.databases.create_string_attribute(
                database_id, collection_id, attr["key"], attr["size"], attr["required"]
            ),
            "integer": lambda: self.databases.create_integer_attribute(
                database_id, collection_id, attr["key"], attr["required"]
            ),
            "datetime": lambda: self.databases.create_datetime_attribute(
                database_id, collection_id, attr["key"], attr["required"]
            ),
        }
        if attr["type"] in attr_creators:
            attr_creators[attr["type"]]()
            logger.debug(f"Created {attr['type']} attribute: {attr['key']}")

    def _create_attribute_by_type(
        self,
        database_id: str,
        collection_id: str,
        key: str,
        attr_type: str,
        is_array: bool,
    ):
        common_args = [database_id, collection_id, key]
        try:
            if attr_type == "string":
                result = self.databases.create_string_attribute(
                    *common_args, size=255, required=False, array=is_array
                )
            elif attr_type == "integer":
                result = self.databases.create_integer_attribute(
                    *common_args, required=False, array=is_array
                )
            elif attr_type == "boolean" and not is_array:
                result = self.databases.create_boolean_attribute(
                    *common_args, required=False
                )
            else:
                result = self.databases.create_string_attribute(
                    *common_args, size=255, required=False, array=is_array
                )
                logger.warning(
                    f"Unsupported type {attr_type} for {key}, defaulted to string"
                )
            logger.debug(f"Created {attr_type} attribute '{key}' (array: {is_array})")
            return result
        except AppwriteException as e:
            if "already exists" in e.message.lower():
                logger.debug(f"Attribute '{key}' already exists")
                return None
            raise e

    # -- buckets -----------------------------------------------------------------

    def ensure_bucket(
        self,
        bucket_id: str,
        name: str = None,
        permissions: List[str] = None,
        file_security: bool = True,
        enabled: bool = True,
        maximum_file_size: int = None,
        allowed_file_extensions: List[str] = None,
        encryption: bool = False,
        compression: str = "none",
        antivirus: bool = True,
        create_metadata_db: bool = True,
    ) -> OperationResult:
        """Create a storage bucket, and optionally its metadata database."""
        try:
            if permissions is None:
                permissions = []
            if allowed_file_extensions is None:
                allowed_file_extensions = []

            result = self.storage.create_bucket(
                bucket_id=bucket_id,
                name=name,
                permissions=permissions,
                file_security=file_security,
                enabled=enabled,
                maximum_file_size=maximum_file_size,
                allowed_file_extensions=allowed_file_extensions,
                encryption=encryption,
                compression=compression,
                antivirus=antivirus,
            )

            if create_metadata_db:
                # C-238: the arguments are swapped — the signature is
                # (database_id, database_name) and this passes (bucket_name, database_id).
                # Reproduced verbatim from file.py: a relocation must not change
                # behaviour silently. Fix under its own issue, with a test.
                db_result = self.ensure_database(name, self.config.database_id)
                if db_result.success:
                    result["metadata_database"] = db_result.data

            return OperationResult(success=True, data=result, code="CREATED")

        except AppwriteException as e:
            return OperationResult(success=False, error=e.message, code=e.type)


def build_provisioner(bucket_override: Optional[str] = None) -> AppwriteProvisioner:
    """Construct a provisioner from the validated process environment."""
    from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
    from views_pipeline_core.modules.appwrite.file import (
        AppWriteFileModule,
        AppwriteConfig,
    )

    store = PredictionStoreConfig.from_environment()
    config = AppwriteConfig(
        path_manager=None,
        endpoint=store.endpoint,
        project_id=store.project_id,
        credentials=store.api_key,
        auth_method="api_key",
        cache_dir=".appwrite_provisioning_cache",
        bucket_id=bucket_override or store.bucket_id,
        bucket_name=store.bucket_name,
        collection_id=store.collection_id,
        collection_name=store.collection_name,
        database_id=store.database_id,
        database_name=store.database_name,
    )
    return AppwriteProvisioner(AppWriteFileModule(config))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m views_pipeline_core.modules.appwrite.provisioning",
        description=(
            "Create Appwrite containers deliberately. The delivery path no longer "
            "provisions; run this once when setting up a project or a new bucket."
        ),
    )
    parser.add_argument(
        "action",
        choices=["ensure-bucket", "ensure-database", "ensure-collection"],
        help="What to create if it does not already exist.",
    )
    parser.add_argument("--bucket", default=None, help="Bucket id. Defaults to the env var.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    provisioner = build_provisioner(args.bucket)
    bucket_id = args.bucket or provisioner.config.bucket_id

    if args.action == "ensure-bucket":
        result = provisioner.ensure_bucket(
            bucket_id=bucket_id, name=provisioner.config.bucket_name
        )
    elif args.action == "ensure-database":
        result = provisioner.ensure_database()
    else:
        result = provisioner.ensure_collection()

    if result.success:
        logger.info("%s: OK (%s)", args.action, result.code or "CREATED")
        return 0
    logger.error("%s FAILED: code=%s error=%s", args.action, result.code, result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
