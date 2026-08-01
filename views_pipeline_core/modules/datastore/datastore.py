"""Datastore module for managing prediction files and metadata via Appwrite.

This module provides a high-level interface for uploading, downloading, searching,
and managing prediction files stored in Appwrite cloud storage. It handles metadata
management, file versioning and caching. It does NOT create buckets or collections.

GOVERNING CONTRACT — the VIEWS Appwrite seam's identity, credential and coordinate rules
are platform surface:

    PLATFORM-001 (views-appwrite), pinned at tag platform-001-v1.2.0:
    https://github.com/views-platform/views-appwrite/blob/platform-001-v1.2.0/docs/ADRs/platform/PLATFORM-001_identity_secrets_configuration_contract.md

    Coordinate registry (THE canonical source for ids — never copied into code):
    https://github.com/views-platform/views-appwrite/blob/platform-001-v1.2.0/docs/ADRs/platform/coordinate_registry.toml

**Callers: `upload_data()` reports failure by RETURN VALUE, not by exception.** The SDK's
`AppwriteException` is converted to `OperationResult(success=False)` inside the storage
module, so an `except` around this call will not fire. Inspect the result (ADR-046 §1 as
amended 2026-07-31; register C-227). Locally: ADR-046, ADR-047.

Typical usage example:

    from views_pipeline_core.modules.datastore import DatastoreModule
    from views_pipeline_core.modules.appwrite import AppwriteConfig

    config = AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="my_project",
        credentials="my_api_key",
        bucket_id="predictions"
    )
    datastore = DatastoreModule(config)

    # Upload a prediction file
    result = datastore.upload_data(
        file="/path/to/predictions.parquet",
        filename="model_predictions.parquet",
        loa="pgm",
        name="fatalities_model",
        type="model",
        targets=["pred_ged_sb"],
        category="forecast"
    )

    # Download the latest prediction
    latest = datastore.download_latest_file(filters={"loa": "pgm"})
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from views_pipeline_core.modules.appwrite import AppwriteConfig, AppWriteFileModule, OperationResult
import logging
import pandas as pd

import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetadataSearchIncomplete(RuntimeError):
    """The metadata lookup could not be certified, so its result is not an answer.

    Raised where a caller would otherwise receive an empty list or ``None`` — values
    that mean "there is no such file" and must never be produced by a lookup that
    merely failed. See register C-241 and the Cluster J entry: the recurring defect on
    this platform is a system that cannot distinguish "no" from "I could not tell" and
    answers anyway.

    It subclasses ``RuntimeError`` so that existing broad handlers still catch it,
    while a caller that wants to retry the read specifically can.
    """


class FileMetadata:
    """Metadata container for prediction files with validation.

    This class encapsulates and validates metadata required for prediction files
    stored in the datastore. It ensures type safety and valid category values
    before allowing file uploads.

    Attributes:
        loa: Level of analysis (e.g., 'pgm' for PRIO-GRID-month, 'cm' for country-month).
        name: Model name or identifier for the prediction.
        type: Type of model (e.g., 'model', 'postprocessor', 'ensemble').
        targets: List of target variable names (e.g., ['pred_ged_sb', 'pred_ged_ns']).
        category: Either 'forecast' or 'historical' to categorize the prediction.
        description: Optional human-readable description of the prediction file.

    Raises:
        TypeError: If any argument has an incorrect type.
        ValueError: If category is not 'forecast' or 'historical'.

    Example:
        >>> metadata = FileMetadata(
        ...     loa="pgm",
        ...     name="fatalities_model_v2",
        ...     type="model",
        ...     targets=["pred_ged_sb", "pred_ged_ns", "pred_ged_os"],
        ...     category="forecast",
        ...     description="Monthly fatality predictions for PRIO-GRID cells"
        ... )
        >>> metadata.to_dict()
        {'loa': 'pgm', 'name': 'fatalities_model_v2', 'type': 'model', ...}
    """

    def __init__(
        self,
        loa: str,
        name: str,
        type: str,
        targets: List[str],
        category: str,
        description: Optional[str] = None,
    ):
        """Initialize FileMetadata with validation.

        Args:
            loa: Level of analysis identifier (e.g., 'pgm', 'cm').
            name: Model name or identifier.
            type: Type of model (e.g., 'model', 'postprocessor', 'ensemble').
            targets: List of target variable names being predicted.
            category: Must be either 'forecast' or 'historical'.
            description: Optional description of the file.

        Raises:
            TypeError: If loa, name, or type are not strings, if targets is not
                a list of strings, or if description is not a string/None.
            ValueError: If category is not 'forecast' or 'historical'.
        """
        if not isinstance(loa, str):
            raise TypeError("loa must be a string")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(type, str):
            raise TypeError("type must be a string")
        if not isinstance(targets, list) or not all(
            isinstance(t, str) for t in targets
        ):
            raise TypeError("targets must be a list of strings")
        if description is not None and not isinstance(description, str):
            raise TypeError("description must be a string or None")
        if category not in ["forecast", "historical"]:
            raise ValueError(f"category must be either 'forecast' or 'historical'. Got: {category}")

        self.loa = loa
        self.name = name
        self.type = type
        self.targets = targets
        self.description = description
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format for storage.

        Returns:
            Dict containing all metadata fields. The 'description' field is
            only included if it has a non-empty value.

        Example:
            >>> metadata = FileMetadata(
            ...     loa="pgm", name="model", type="model",
            ...     targets=["ged_sb"], category="forecast"
            ... )
            >>> metadata.to_dict()
            {'loa': 'pgm', 'name': 'model', 'type': 'model',
             'targets': ['ged_sb'], 'category': 'forecast'}
        """
        data = {
            "loa": self.loa,
            "name": self.name,
            "type": self.type,
            "targets": self.targets,
            "category": self.category,
        }
        if self.description:
            data["description"] = self.description
        return data


class DatastoreModule:
    """High-level interface for managing prediction files in Appwrite storage.

    DatastoreModule provides a simplified API for uploading, downloading, searching,
    and managing prediction files. It wraps the lower-level AppWriteFileModule and
    handles metadata management, file versioning, and automatic bucket creation.

    The module supports:
        - File uploads with automatic metadata extraction and storage
        - Searching predictions by metadata filters (loa, type, targets, etc.)
        - Downloading files with intelligent caching
        - Listing and managing predictions for specific models
        - Automatic bucket creation when needed

    Attributes:
        model_path: ModelPathManager instance for path resolution.

    Example:
        >>> from views_pipeline_core.modules.appwrite import AppwriteConfig
        >>> config = AppwriteConfig(
        ...     endpoint="https://cloud.appwrite.io/v1",
        ...     project_id="views_project",
        ...     credentials="api_key_here",
        ...     bucket_id="forecasts",
        ...     collection_name="Predictions"
        ... )
        >>> datastore = DatastoreModule(config)
        >>>
        >>> # Upload a prediction file
        >>> result = datastore.upload_data(
        ...     file="/data/predictions.parquet",
        ...     filename="pgm_forecast_202401.parquet",
        ...     loa="pgm",
        ...     name="ensemble_model",
        ...     type="model",
        ...     targets=["ged_sb"],
        ...     category="forecast"
        ... )
        >>>
        >>> # Search for predictions
        >>> predictions = datastore.get_predictions_by_metadata(
        ...     filters={"loa": "pgm", "category": "forecast"}
        ... )
        >>>
        >>> # Download the latest file
        >>> download = datastore.download_latest_file(
        ...     filters={"type": "model"},
        ...     save_path="/tmp/latest_prediction.parquet"
        ... )
    """

    def __init__(self, appwrite_file_manager_config: AppwriteConfig):
        """Initialize DatastoreModule with Appwrite configuration.

        Args:
            appwrite_file_manager_config: AppwriteConfig instance containing
                connection settings, authentication credentials, bucket/collection
                identifiers, and optional path manager.

        Example:
            >>> config = AppwriteConfig(
            ...     endpoint="https://cloud.appwrite.io/v1",
            ...     project_id="my_project",
            ...     credentials="my_api_key",
            ...     bucket_id="production_forecasts"
            ... )
            >>> datastore = DatastoreModule(config)
        """
        self.model_path = appwrite_file_manager_config.path_manager
        self.__appwrite_file_manager_config = appwrite_file_manager_config
        self.__appwrite_file_manager = AppWriteFileModule(
            self.__appwrite_file_manager_config
        )

    def upload_predictions(
        self,
        file: Union[Path, str, pd.DataFrame],
        filename: str,
        loa: str,
        name: Optional[str],
        type: str,
        targets: List[str],
        category: str,
        description: Optional[str] = None,
    ) -> OperationResult:
        """Upload a prediction file with metadata (DEPRECATED).

        .. deprecated::
            Use :meth:`upload_data` instead. This method is maintained for
            backward compatibility and simply delegates to upload_data.

        Args:
            file: Path to the file or DataFrame to upload.
            filename: Name to give the file in storage.
            loa: Level of analysis (e.g., 'pgm', 'cm').
            name: Model name. If None, uses model_path.model_name.
            type: Type of model (e.g., 'model', 'postprocessor', 'ensemble').
            targets: List of target variable names.
            category: Either 'forecast' or 'historical'.
            description: Optional description of the prediction.

        Returns:
            OperationResult with success status and uploaded file data.

        Raises:
            NotImplementedError: If file is a DataFrame (not yet supported).
            TypeError: If file is not a Path, str, or DataFrame.
        """
        logger.warning("upload_predictions is deprecated. Use upload_data instead.")
        return self.upload_data(
            file=file,
            filename=filename,
            loa=loa,
            name=name,
            type=type,
            targets=targets,
            category=category,
            description=description,
        )
    
    def upload_data(
        self,
        file: Union[Path, str, pd.DataFrame],
        filename: str,
        loa: str,
        name: Optional[str],
        type: str,
        targets: List[str],
        category: str,
        description: Optional[str] = None,
    ) -> OperationResult:
        """Upload a data file with associated metadata to Appwrite storage.

        Uploads a file to the configured Appwrite bucket and stores metadata
        in the associated database collection. Automatically creates the bucket
        if it doesn't exist. Handles duplicate detection via file hashing.

        Args:
            file: Path to the file to upload. Can be a Path object or string.
                DataFrame uploads are not yet implemented.
            filename: Name to give the file in storage (e.g., 'predictions.parquet').
            loa: Level of analysis identifier (e.g., 'pgm' for PRIO-GRID-month).
            name: Model name or identifier. If None, uses model_path.model_name.
            type: Type of model (e.g., 'model', 'postprocessor', 'ensemble').
            targets: List of target variable names being predicted.
            category: Must be 'forecast' or 'historical'.
            description: Optional human-readable description.

        Returns:
            OperationResult with:
                - success: True if upload succeeded
                - data: Dictionary containing file_id, document_id, and metadata
                - code: 'UPLOAD_SUCCESS', 'METADATA_UPDATED', or error code
                - error: Error message if success is False

        Raises:
            NotImplementedError: If file is a pandas DataFrame.
            TypeError: If file is not a Path, str, or DataFrame.

        Example:
            >>> result = datastore.upload_data(
            ...     file="/data/output/pgm_predictions.parquet",
            ...     filename="pgm_forecast_202401.parquet",
            ...     loa="pgm",
            ...     name="fatalities_ensemble",
            ...     type="ensemble",
            ...     targets=["pred_ged_sb", "pred_ged_ns"],
            ...     category="forecast",
            ...     description="January 2024 ensemble predictions"
            ... )
            >>> if result.success:
            ...     print(f"Uploaded with ID: {result.data['file_id']}")
        """
        if name is None:
            name = self.model_path.model_name
        metadata = FileMetadata(
            loa=loa, name=name, type=type, targets=targets, description=description, category=category
        ).to_dict()
        if isinstance(file, pd.DataFrame):
            raise NotImplementedError(
                "Uploading a DataFrame directly is not implemented."
            )
        elif isinstance(file, (Path, str)):
            file_path = str(file)
            upload_result = self.__appwrite_file_manager.upload_file_with_metadata(
                bucket_id=self.__appwrite_file_manager_config.bucket_id,
                filename=filename,
                file_path=file_path,
                metadata=metadata,
                collection_name=self.__appwrite_file_manager_config.collection_name,
                collection_id=self.__appwrite_file_manager_config.collection_id,
            ).to_dict()
        else:
            raise TypeError("file must be a Path, str, or pd.DataFrame")

        # A missing bucket used to be CREATED here and the upload retried into it —
        # so a mistyped or renamed coordinate silently provisioned a new bucket in
        # production and published the forecast where nobody reads (register C-228,
        # þing-02 #331). Provisioning is now a deliberate act; a wrong coordinate
        # fails and says which one:
        #   python -m views_pipeline_core.modules.appwrite.provisioning ensure-bucket
        if upload_result.get("code") == "storage_bucket_not_found":
            bucket_id = self.__appwrite_file_manager_config.bucket_id
            logger.error(
                "Appwrite bucket '%s' does not exist. Refusing to create it from the "
                "delivery path — run `python -m "
                "views_pipeline_core.modules.appwrite.provisioning ensure-bucket "
                "--bucket %s` if this bucket is genuinely new, or correct "
                "APPWRITE_PROD_FORECASTS_BUCKET_ID if it is a typo.",
                bucket_id,
                bucket_id,
            )

        return OperationResult(**upload_result)

    def get_predictions_by_metadata(
        self, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get predictions by metadata filters.
        
        Args:
            filters: Dictionary of metadata fields to filter by. If None, returns all predictions.
                    If provided, will be merged with model name filter.
        
        Returns:
            List of prediction metadata documents, sorted by creation date (newest first)
        """
        # Start with model name filter if available
        if filters is None:
            filters = {}
        
        # Add model name to filters if model_path has it
        if hasattr(self.model_path, 'model_name') and self.model_path.model_name:
            filters["name"] = self.model_path.model_name
        
        logger.info(f"Searching for predictions with filters: {filters}")
        
        # FIXED: Use correct attribute name
        search_result = (
            self.__appwrite_file_manager.metadata_manager.search_files_by_metadata(
                filters=filters if filters else None,
                collection_name=self.__appwrite_file_manager_config.collection_name,  
                collection_id=self.__appwrite_file_manager_config.collection_id,      
                database_id=self.__appwrite_file_manager_config.database_id,          
            ).to_dict()
        )

        if not search_result.get("success", False):
            # Returning [] here would tell every caller "no predictions match", which is
            # a statement about the shelf rather than about the lookup. `get_latest_file_id`
            # would then hand back None and the FAO delivery would report nothing to
            # deliver — a false negative to an external counterparty, produced by a
            # failure we had already detected. C-241, Cluster J.
            error_msg = search_result.get("error", "Unknown error")
            code = search_result.get("code", "UNKNOWN")
            logger.error(f"Metadata search failed ({code}) with filters {filters}: {error_msg}")
            raise MetadataSearchIncomplete(
                f"metadata search failed ({code}) for filters {filters}: {error_msg}"
            )

        documents = search_result.get("data", {}).get("documents", [])
        logger.info(f"Found {len(documents)} prediction files")
        
        if len(documents) == 0:
            logger.warning(f"No files found matching filters: {filters}")
            logger.info("Try calling list_all_predictions_unfiltered() to see all files in the bucket")
        
        return sorted(
            documents,
            key=lambda x: x.get("$createdAt", ""),
            reverse=True,
        )

    def download_prediction(
        self,
        file_id: str,
        save_path: Union[Path, str] = None,
        use_cache: bool = True,
        validate_cache: bool = True,
    ) -> OperationResult:
        """Download a prediction file by its ID.

        Downloads a file from Appwrite storage with optional caching support.
        Can return file bytes or save directly to disk.

        Args:
            file_id: The Appwrite file ID to download.
            save_path: Optional path where the file should be saved. If None,
                returns file bytes in the result data.
            use_cache: Whether to check local cache before downloading.
                Defaults to True.
            validate_cache: Whether to validate cache freshness against
                remote file timestamps. Defaults to True.

        Returns:
            OperationResult with:
                - success: True if download succeeded
                - data: Dict with 'save_path' or 'file_bytes' and 'from_cache' flag
                - code: 'SAVED_FROM_CACHE', 'RETURNED_FROM_CACHE',
                       'SAVED_FROM_REMOTE', or 'RETURNED_FROM_REMOTE'

        Example:
            >>> # Download to memory
            >>> result = datastore.download_prediction(file_id="abc123")
            >>> if result.success:
            ...     file_bytes = result.data['file_bytes']
            >>>
            >>> # Download to file
            >>> result = datastore.download_prediction(
            ...     file_id="abc123",
            ...     save_path="/tmp/prediction.parquet"
            ... )
        """
        download_result = self.__appwrite_file_manager.download_file(
            bucket_id=self.__appwrite_file_manager_config.bucket_id,
            file_id=file_id,
            save_path=save_path,
            use_cache=use_cache,
            validate_cache=validate_cache,
        )
        return download_result

    def get_latest_file_id(self, filters: Dict[str, Any]) -> Optional[str]:
        """Get the file ID of the most recently uploaded prediction matching filters.

        Searches predictions by metadata and returns the file ID of the newest
        matching file based on creation timestamp.

        Args:
            filters: Dictionary of metadata fields to filter by.
                Common filters include 'loa', 'type', 'category', 'targets'.

        Returns:
            The file ID string of the latest matching file, or None if no
            files match the given filters.

        Example:
            >>> file_id = datastore.get_latest_file_id(
            ...     filters={"loa": "pgm", "category": "forecast"}
            ... )
            >>> if file_id:
            ...     print(f"Latest file: {file_id}")
            ... else:
            ...     print("No matching files found")
        """
        files_list = self.get_predictions_by_metadata(filters=filters)
        if len(files_list) == 0:
            logger.warning(f"No files found matching the given filters: {filters}")
            return None
        
        latest_file = files_list[0]
        file_id = latest_file.get("fileId", None)
        
        if file_id:
            logger.info(f"Latest file ID: {file_id} (created: {latest_file.get('$createdAt')})")
        else:
            logger.warning(f"Latest file found but missing 'fileId' field: {latest_file}")
        
        return file_id

    def download_latest_file(
        self,
        filters: Dict[str, Any] = {},
        save_path: Union[Path, str] = None,
        use_cache: bool = True,
        validate_cache: bool = True,
    ) -> OperationResult:
        """Download the most recently uploaded prediction matching filters.

        Convenience method that combines get_latest_file_id and download_prediction
        to fetch the newest file matching the given metadata filters.

        Args:
            filters: Dictionary of metadata fields to filter by. Defaults to
                empty dict which applies model name filter only.
            save_path: Optional path where the file should be saved. If None,
                returns file bytes in the result data.
            use_cache: Whether to check local cache before downloading.
                Defaults to True.
            validate_cache: Whether to validate cache freshness against
                remote file timestamps. Defaults to True.

        Returns:
            OperationResult with downloaded file data.

        Raises:
            FileNotFoundError: If no files match the given filters.

        Example:
            >>> # Download latest forecast to file
            >>> result = datastore.download_latest_file(
            ...     filters={"loa": "pgm", "category": "forecast"},
            ...     save_path="/tmp/latest_forecast.parquet"
            ... )
            >>> if result.success:
            ...     print(f"Downloaded to: {result.data['save_path']}")
            >>>
            >>> # Download latest to memory
            >>> result = datastore.download_latest_file(
            ...     filters={"category": "historical"}
            ... )
            >>> file_bytes = result.data['file_bytes']
        """
        latest_file_id = self.get_latest_file_id(filters=filters)
        if latest_file_id is None:
            error_msg = f"No files found matching the given filters: {filters}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Downloading latest file: {latest_file_id}")
        return self.download_prediction(
            file_id=latest_file_id,
            save_path=save_path,
            use_cache=use_cache,
            validate_cache=validate_cache,
        )
    
    def get_file_metadata(self, file_id: str) -> OperationResult:
        """
        Get metadata for a specific file from the metadata collection.
        
        Args:
            file_id: The ID of the file to get metadata for
            
        Returns:
            OperationResult with the file metadata
        """
        try:
            search_result = self.__appwrite_file_manager.metadata_manager.search_files_by_metadata(
                filters={"fileId": file_id},
                collection_name=self.__appwrite_file_manager_config.collection_name,
                collection_id=self.__appwrite_file_manager_config.collection_id,
                database_id=self.__appwrite_file_manager_config.database_id,
            )
            
            if not search_result.success:
                return OperationResult(
                    success=False,
                    error=f"Failed to get metadata for file {file_id}: {search_result.error}",
                    code=search_result.code
                )
            
            documents = search_result.data.get("documents", [])
            if not documents:
                return OperationResult(
                    success=False,
                    error=f"No metadata found for file {file_id}",
                    code="NOT_FOUND"
                )
            
            return OperationResult(
                success=True,
                data=documents[0],
                code="FOUND"
            )
        
        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                code="UNKNOWN_ERROR"
            )
    
    def update_prediction_metadata(
        self,
        file_id: str,
        metadata_updates: Dict[str, Any]
    ) -> OperationResult:
        """
        Update metadata for a specific prediction file.
        
        Args:
            file_id: The ID of the file to update
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            OperationResult with the update status
        """
        return self.__appwrite_file_manager.metadata_manager.update_file_metadata(
            file_id=file_id,
            metadata_updates=metadata_updates,
            collection_name=self.__appwrite_file_manager_config.collection_name,
            collection_id=self.__appwrite_file_manager_config.collection_id,
            database_id=self.__appwrite_file_manager_config.database_id,
        )
    
    def delete_prediction(self, file_id: str) -> OperationResult:
        """
        Delete a prediction file and its metadata.
        
        Args:
            file_id: The ID of the file to delete
            
        Returns:
            OperationResult with the deletion status
        """
        # Delete the file from storage
        delete_result = self.__appwrite_file_manager.delete_file(
            bucket_id=self.__appwrite_file_manager_config.bucket_id,
            file_id=file_id
        )
        
        if not delete_result.success:
            logger.error(f"Failed to delete file {file_id}: {delete_result.error}")
        
        return delete_result
    
    def list_all_predictions(
        self,
    ) -> List[Dict[str, Any]]:
        """
        List all predictions for the current model.
        
        Returns:
            List of prediction metadata documents
        """
        filters = {"name": self.model_path.model_name}
        return self.get_predictions_by_metadata(filters=filters)
    
    # Debug
    def list_all_predictions_unfiltered(self) -> List[Dict[str, Any]]:
        """
        List all predictions in the bucket without any filters.
        Useful for debugging when filtered searches return no results.
        
        Returns:
            List of all prediction metadata documents
        """
        logger.info("Listing all predictions without filters")
        
        search_result = (
            self.__appwrite_file_manager.metadata_manager.search_files_by_metadata(
                filters=None,  # No filters
                collection_name=self.__appwrite_file_manager_config.collection_name,
                collection_id=self.__appwrite_file_manager_config.collection_id,
                database_id=self.__appwrite_file_manager_config.database_id,
            ).to_dict()
        )

        if not search_result.get("success", False):
            # Same swallow as `get_predictions_by_metadata`, and worse here: this method
            # is what an operator reaches for to answer "what is actually on the shelf?".
            # An empty list is the most misleading possible reply to that question when
            # the read failed. C-241, Cluster J.
            error_msg = search_result.get("error", "Unknown error")
            code = search_result.get("code", "UNKNOWN")
            logger.error(f"Unfiltered metadata search failed ({code}): {error_msg}")
            raise MetadataSearchIncomplete(
                f"unfiltered metadata search failed ({code}): {error_msg}"
            )

        documents = search_result.get("data", {}).get("documents", [])
        logger.info(f"Found {len(documents)} total prediction files")
        
        return sorted(
            documents,
            key=lambda x: x.get("$createdAt", ""),
            reverse=True,
        )