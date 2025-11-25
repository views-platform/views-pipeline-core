from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.modules.appwrite import AppwriteConfig, AppWriteFileModule, OperationResult
import logging
import pandas as pd
import os

import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# APPWRITE_DATASTORE_PROJECT_ID = os.getenv("APPWRITE_DATASTORE_PROJECT_ID")
# APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
# APPWRITE_DATASTORE_API_KEY = os.getenv("APPWRITE_DATASTORE_API_KEY")


class FileMetadata:
    def __init__(
        self,
        loa: str,
        name: str,
        type: str,
        targets: List[str],
        category: str,
        description: Optional[str] = None,
    ):
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
    def __init__(self, appwrite_file_manager_config: AppwriteConfig):
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
        # if name is None:
        #     name = self.model_path.model_name
        # metadata = FileMetadata(
        #     loa=loa, name=name, type=type, targets=targets, description=description, category=category
        # ).to_dict()
        # if isinstance(file, pd.DataFrame):
        #     raise NotImplementedError(
        #         "Uploading a DataFrame directly is not implemented."
        #     )
        # elif isinstance(file, (Path, str)):
        #     file_path = str(file)
        #     upload_result = self.__appwrite_file_manager.upload_file_with_metadata(
        #         bucket_id=self.__appwrite_file_manager_config.bucket_id,
        #         filename=filename,
        #         file_path=file_path,
        #         metadata=metadata,
        #         collection_name=self.__appwrite_file_manager_config.collection_name,
        #         collection_id=self.__appwrite_file_manager_config.collection_id,
        #     ).to_dict()
        # else:
        #     raise TypeError("file must be a Path, str, or pd.DataFrame")

        # if upload_result.get("code") == "storage_bucket_not_found":
        #     logger.info(
        #         f"Bucket '{self.__appwrite_file_manager_config.bucket_id}' not found. Creating it..."
        #     )
        #     try:
        #         self.__appwrite_file_manager.create_bucket(
        #             bucket_id=self.__appwrite_file_manager_config.bucket_id,
        #             name=self.__appwrite_file_manager_config.bucket_name,
        #         )
        #     except Exception as e:
        #         logger.error(f"Failed to create bucket: {e}")
        #         return OperationResult(success=False, error=str(e))

        #     upload_result = self.__appwrite_file_manager.upload_file_with_metadata(
        #         bucket_id=self.__appwrite_file_manager_config.bucket_id,
        #         file_path=file_path,
        #         filename=filename,
        #         metadata=metadata,
        #         collection_name=self.__appwrite_file_manager_config.collection_name,
        #         collection_id=self.__appwrite_file_manager_config.collection_id,
        #     ).to_dict()
        
        # return OperationResult(**upload_result)
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
    
    # Same as upload_predictions but for generic data. Will be refactored later.
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

        if upload_result.get("code") == "storage_bucket_not_found":
            logger.info(
                f"Bucket '{self.__appwrite_file_manager_config.bucket_id}' not found. Creating it..."
            )
            try:
                self.__appwrite_file_manager.create_bucket(
                    bucket_id=self.__appwrite_file_manager_config.bucket_id,
                    name=self.__appwrite_file_manager_config.bucket_name,
                )
            except Exception as e:
                logger.error(f"Failed to create bucket: {e}")
                return OperationResult(success=False, error=str(e))

            upload_result = self.__appwrite_file_manager.upload_file_with_metadata(
                bucket_id=self.__appwrite_file_manager_config.bucket_id,
                file_path=file_path,
                filename=filename,
                metadata=metadata,
                collection_name=self.__appwrite_file_manager_config.collection_name,
                collection_id=self.__appwrite_file_manager_config.collection_id,
            ).to_dict()
        
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
            logger.warning(f"Search failed with filters: {filters}")
            error_msg = search_result.get("error", "Unknown error")
            logger.error(f"Search error: {error_msg}")
            return []
        
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
        download_result = self.__appwrite_file_manager.download_file(
            bucket_id=self.__appwrite_file_manager_config.bucket_id,
            file_id=file_id,
            save_path=save_path,
            use_cache=use_cache,
            validate_cache=validate_cache,
        )
        return download_result

    def get_latest_file_id(self, filters: Dict[str, Any]) -> Optional[str]:
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
            logger.error(f"Search error: {search_result.get('error', 'Unknown error')}")
            return []
        
        documents = search_result.get("data", {}).get("documents", [])
        logger.info(f"Found {len(documents)} total prediction files")
        
        return sorted(
            documents,
            key=lambda x: x.get("$createdAt", ""),
            reverse=True,
        )