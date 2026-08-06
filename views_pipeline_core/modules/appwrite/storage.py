"""storage.py — extracted from modules/appwrite/file.py (M-1 audit decision).

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
from views_pipeline_core.modules.appwrite.transport import (
    install_request_timeout,
    resolve_timeout_seconds,
)
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
import shutil
import logging
from views_pipeline_core.exceptions.exceptions import ConfigurationException
from views_pipeline_core.modules.appwrite.config import (
    AppwriteConfig,
    OperationResult,
    FileMetadata,
    DEFAULT_PAGE_LIMIT,
    MAX_METADATA_PAGES,
    _CONTAINER_PAGE,
    APPWRITE_FILE_NOT_FOUND,
    APPWRITE_BUCKET_NOT_FOUND,
    _REQUIRED_COORDINATES,
    _StoragePresence,
    _classify_storage_presence,
    exception_message,
)
from views_pipeline_core.modules.appwrite.cache import CacheManager, CacheValidationResult, CacheMetadata
from views_pipeline_core.modules.appwrite.metadata import AppwriteMetadataHandler
from views_pipeline_core.modules.appwrite.auth import AuthFactory

logger = logging.getLogger(__name__)


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
        # C-15 / #347: bound every HTTP call before the client can make one. The SDK
        # offers no timeout hook, so this installs one at its transport reference — see
        # modules/appwrite/transport.py for why that is the narrowest option available.
        # Idempotent, so constructing several managers does not stack proxies.
        install_request_timeout()

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
                    # C-258. This walk decides whether a duplicate EXISTS, so a short
                    # read does not return fewer files — it returns NOT_FOUND, which the
                    # caller reads as "no duplicate" and uploads one. Same rule as
                    # #341: terminate on an EMPTY page (Appwrite may grant fewer rows
                    # than asked), advance by what was RECEIVED, bound the loop, and
                    # certify against the substrate's own total before answering.
                    all_files = []
                    offset = 0
                    limit = DEFAULT_PAGE_LIMIT
                    reported_total = None
                    complete = False

                    for _ in range(MAX_METADATA_PAGES):
                        result = self.storage.list_files(
                            bucket_id, [Query.limit(limit), Query.offset(offset)]
                        )
                        if reported_total is None:
                            reported_total = result.get("total")
                        files_chunk = result.get("files", [])
                        all_files.extend(files_chunk)
                        if not files_chunk:
                            complete = True
                            break
                        offset += len(files_chunk)

                    if not complete or (
                        reported_total is not None and len(all_files) != reported_total
                    ):
                        # Never NOT_FOUND from a walk that did not finish: "I could not
                        # look" must not be delivered as "it is not there".
                        logger.error(
                            f"Duplicate check for {filename!r} could not enumerate "
                            f"bucket {bucket_id!r}: collected {len(all_files)} of a "
                            f"reported {reported_total}"
                        )
                        return OperationResult(
                            success=False,
                            error=(
                                f"Duplicate check incomplete: enumerated "
                                f"{len(all_files)} of a reported {reported_total} files "
                                f"in {bucket_id!r}"
                            ),
                            code="LISTING_INCOMPLETE",
                        )

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
            # bucket and says nothing (the fourth route to the state the
            # `modules/appwrite/audit/` package exists to detect; register C-227's
            # disease at this site).
            try:
                rollback = self.delete_file(bucket_id, file_id)
                if not rollback.success:
                    logger.error(
                        "Rollback FAILED: file %s remains in bucket '%s' with no "
                        "metadata document (code=%s, %s). Run `python -m "
                        "views_pipeline_core.modules.appwrite.audit` to confirm.",
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
            if "Storage bucket with the requested ID could not be found" in exception_message(e):
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



    def get_user_preferences(self, user_id: str) -> OperationResult:
        """Get a user's preferences from Appwrite. Requires API-key authentication.

        `user_id` became required when session auth was deleted (#344): it was optional
        only because a session could supply the current user implicitly. There is no
        implicit user any more, so a caller that omits it is asking a question this
        method cannot answer.

        Args:
            user_id: The user whose preferences to read.

        Returns:
            OperationResult with the preferences, code='API_KEY'.
        """
        try:
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


