"""metadata.py — extracted from modules/appwrite/file.py (M-1 audit decision).

This module contains the classes that were previously in the
2,841-LOC God module `file.py`. The original `file.py` is now a
re-export shim that preserves all existing import paths.
"""

from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.services.users import Users
from appwrite.input_file import InputFile
from appwrite.id import ID
from appwrite.exception import AppwriteException
from appwrite.query import Query
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from views_pipeline_core.modules.appwrite.config import (
    AppwriteConfig,
    FileMetadata,
    OperationResult,
    exception_message,
    MAX_METADATA_PAGES,
    DEFAULT_PAGE_LIMIT,
)

logger = logging.getLogger(__name__)


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
            if "Attribute not found in schema: file_hash" in exception_message(e):
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


