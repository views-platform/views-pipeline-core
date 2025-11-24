from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.services.account import Account
from appwrite.services.users import Users
from appwrite.input_file import InputFile
from appwrite.id import ID
from appwrite.role import Role
from appwrite.permission import Permission
from appwrite.exception import AppwriteException
from appwrite.query import Query
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime, timedelta
import shutil
import time
import logging
import hashlib
from functools import wraps
from views_pipeline_core.managers.model import ModelPathManager
from abc import ABC, abstractmethod
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_PAGE_LIMIT = 100
# DEFAULT_METADATA_COLLECTION_NAME = "file_metadata"
MAX_ATTRIBUTE_CREATION_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0


# Enums
class AuthMethod(Enum):
    API_KEY = "api_key"
    SESSION = "session"

class CacheValidationResult(Enum):
    VALID = "valid"
    INVALID_TTL = "invalid_ttl"
    INVALID_TIMESTAMP = "invalid_timestamp"
    NOT_FOUND = "not_found"

# Type Definitions
@dataclass
class OperationResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "code": self.code
        }

@dataclass
class FileMetadata:
    fileId: str
    bucketId: str
    filename: str
    mime_type: str = "application/octet-stream"
    uploaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_size: Optional[int] = None
    file_hash: Optional[str] = None

@dataclass
class AppwriteConfig:
    # Core connection settings
    endpoint: str
    project_id: str
    credentials: Union[str, Dict[str, str]]
    
    # Authentication settings
    auth_method: AuthMethod = AuthMethod.API_KEY
    
    # Cache settings
    cache_dir: Optional[str] = None
    cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS
    allow_metadata_only_updates: bool = True  # Whether to update metadata only when file hash exists
    
    # Storage settings
    bucket_id: str = "production_forecasts"
    bucket_name: Optional[str] = None  # Will default to bucket_id with spaces if not provided
    
    # Metadata settings
    collection_name: str = "Metadata"  #"Pipeline Forecasts"
    collection_id: Optional[str] = "metadata"  # Custom collection ID override
    database_name: Optional[str] = "File Metadata" #"File Metadata"  # Will default to bucket_name + " Metadata" if not provided
    database_id: Optional[str] = "file_metadata"  # Custom database ID override
    
    # Path manager
    path_manager: ModelPathManager = None
    
    def __post_init__(self):
        if isinstance(self.auth_method, str):
            self.auth_method = AuthMethod(self.auth_method)
        
        # Set defaults for derived values
        if not self.bucket_name:
            self.bucket_name = self.bucket_id.replace("_", " ").title()
        
        if not self.database_name:
            self.database_name = self.collection_id.replace("_", " ").title()

# Authentication Classes
class AuthManager(ABC):
    @abstractmethod
    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        pass

class ApiKeyAuth(AuthManager):
    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        if not isinstance(credentials, str):
            return OperationResult(
                success=False,
                error="API key authentication requires string credentials",
                code="INVALID_CREDENTIALS"
            )
        
        client.set_key(credentials)
        return OperationResult(success=True)

class SessionAuth(AuthManager):
    def __init__(self):
        self.account = None
        self.current_user_id = None
    
    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        if not isinstance(credentials, dict) or not all(k in credentials for k in ["email", "password"]):
            return OperationResult(
                success=False,
                error="Session authentication requires dict with 'email' and 'password'",
                code="INVALID_CREDENTIALS"
            )
        
        self.account = Account(client)
        client.set_key("")  # Clear API key for session auth
        
        session_result = self._create_session(credentials["email"], credentials["password"])
        if not session_result.success:
            return session_result
        
        self.current_user_id = session_result.data["user_id"]
        return OperationResult(success=True, data={"user_id": self.current_user_id})
    
    def _create_session(self, email: str, password: str) -> OperationResult:
        try:
            session = self.account.create_email_password_session(email, password)
            return OperationResult(
                success=True,
                data={
                    "session_id": session["$id"],
                    "user_id": session["userId"],
                    "created": session["$createdAt"]
                }
            )
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )

class AuthFactory:
    @staticmethod
    def create_auth(auth_method: AuthMethod) -> AuthManager:
        if auth_method == AuthMethod.API_KEY:
            return ApiKeyAuth()
        elif auth_method == AuthMethod.SESSION:
            return SessionAuth()
        else:
            raise ValueError(f"Unsupported auth method: {auth_method}")

# Cache Management
@dataclass
class CacheMetadata:
    bucket_id: str
    file_id: str
    path: str
    cached_at: str
    size_bytes: int
    filename: str
    remote_updated_at: Optional[str] = None

class CacheManager:
    def __init__(self, cache_dir: Path, cache_ttl: timedelta):
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.cache_metadata_file = cache_dir / "cache_metadata.json"
        self.cache_metadata: Dict[str, CacheMetadata] = {}
        self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, "r") as f:
                    data = json.load(f)
                    self.cache_metadata = {
                        k: CacheMetadata(**v) for k, v in data.items()
                    }
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}
    
    def _save_cache_metadata(self):
        try:
            data = {k: v.__dict__ for k, v in self.cache_metadata.items()}
            with open(self.cache_metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, bucket_id: str, file_id: str) -> str:
        return f"{bucket_id}_{file_id}"
    
    def _get_cache_path(self, bucket_id: str, file_id: str, filename: str = None) -> Path:
        bucket_cache_dir = self.cache_dir / bucket_id
        bucket_cache_dir.mkdir(exist_ok=True)
        
        if filename:
            return bucket_cache_dir / filename
        return bucket_cache_dir / file_id
    
    def validate_cache(self, bucket_id: str, file_id: str, remote_updated_at: str = None) -> CacheValidationResult:
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        if cache_key not in self.cache_metadata:
            return CacheValidationResult.NOT_FOUND
        
        metadata = self.cache_metadata[cache_key]
        cache_path = Path(metadata.path)
        
        if not cache_path.exists():
            return CacheValidationResult.NOT_FOUND
        
        cached_at = datetime.fromisoformat(metadata.cached_at)
        if datetime.now() - cached_at > self.cache_ttl:
            return CacheValidationResult.INVALID_TTL
        
        if remote_updated_at:
            try:
                remote_updated = datetime.fromisoformat(remote_updated_at.replace("Z", "+00:00"))
                cached_at_aware = cached_at.replace(tzinfo=remote_updated.tzinfo)
                if remote_updated > cached_at_aware:
                    return CacheValidationResult.INVALID_TIMESTAMP
            except (ValueError, AttributeError):
                pass
        
        return CacheValidationResult.VALID
    
    def add_to_cache(self, bucket_id: str, file_id: str, file_path: Path, file_metadata: Dict[str, Any] = None):
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        self.cache_metadata[cache_key] = CacheMetadata(
            bucket_id=bucket_id,
            file_id=file_id,
            path=str(file_path),
            cached_at=datetime.now().isoformat(),
            size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            filename=file_metadata.get("name") if file_metadata else file_path.name,
            remote_updated_at=file_metadata.get("$updatedAt") if file_metadata else None
        )
        
        self._save_cache_metadata()
    
    def remove_from_cache(self, bucket_id: str, file_id: str):
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        if cache_key in self.cache_metadata:
            cache_path = Path(self.cache_metadata[cache_key].path)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError as e:
                    logger.warning(f"Failed to delete cache file {cache_path}: {e}")
            
            del self.cache_metadata[cache_key]
            self._save_cache_metadata()
    
    def get_cached_file_path(self, bucket_id: str, file_id: str) -> OperationResult:
        cache_key = self._get_cache_key(bucket_id, file_id)
        
        if cache_key not in self.cache_metadata:
            return OperationResult(
                success=False,
                error="File not in cache",
                code="NOT_CACHED"
            )
        
        cache_path = Path(self.cache_metadata[cache_key].path)
        
        if not cache_path.exists():
            return OperationResult(
                success=False,
                error="Cache file missing",
                code="CACHE_FILE_MISSING"
            )
        
        return OperationResult(
            success=True,
            data={
                "cache_path": str(cache_path),
                "metadata": self.cache_metadata[cache_key].__dict__
            }
        )
    
    def clear_cache(self, bucket_id: str = None, older_than_hours: int = None) -> OperationResult:
        deleted_count = 0
        deleted_bytes = 0
        errors = []
        keys_to_delete = []
        
        for cache_key, metadata in self.cache_metadata.items():
            should_delete = False
            
            if bucket_id and metadata.bucket_id != bucket_id:
                continue
            
            if older_than_hours:
                cached_at = datetime.fromisoformat(metadata.cached_at)
                if datetime.now() - cached_at < timedelta(hours=older_than_hours):
                    continue
            
            should_delete = True
            
            if should_delete:
                cache_path = Path(metadata.path)
                if cache_path.exists():
                    try:
                        size = cache_path.stat().st_size
                        cache_path.unlink()
                        deleted_count += 1
                        deleted_bytes += size
                    except OSError as e:
                        errors.append(f"Failed to delete {cache_path}: {e}")
                
                keys_to_delete.append(cache_key)
        
        for key in keys_to_delete:
            del self.cache_metadata[key]
        
        self._save_cache_metadata()
        
        return OperationResult(
            success=True,
            data={
                "deleted_files": deleted_count,
                "deleted_bytes": deleted_bytes,
                "errors": errors if errors else None
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        total_files = len(self.cache_metadata)
        total_bytes = 0
        by_bucket = {}
        
        for metadata in self.cache_metadata.values():
            bucket_id = metadata.bucket_id
            size = metadata.size_bytes
            total_bytes += size
            
            if bucket_id not in by_bucket:
                by_bucket[bucket_id] = {"files": 0, "bytes": 0}
            
            by_bucket[bucket_id]["files"] += 1
            by_bucket[bucket_id]["bytes"] += size
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "by_bucket": by_bucket
        }

# Metadata Management
class AppwriteMetadataHandler:
    def __init__(self, databases: Databases, config: AppwriteConfig):
        self.databases = databases
        self.config = config
    
    # def _get_metadata_database_name(self, bucket_name: str) -> str:
    #     return f"{bucket_name} Metadata"
    
    # def _get_metadata_database_id(self, bucket_name: str) -> str:
    #     clean_name = "".join(c for c in bucket_name if c.isalnum())
    #     return f"{clean_name}_metadata".lower()
    
    # def _get_collection_id(self, bucket_name: str, collection_name: str, collection_id: str = None) -> str:
    #     # Use custom collection_id if provided, otherwise generate one
    #     if collection_id:
    #         return collection_id
        
    #     clean_bucket_name = "".join(c for c in bucket_name if c.isalnum()).lower()
    #     return f"{collection_name}"
    
    def create_database_if_not_exists(self, database_id: str = None, database_name: str = None) -> OperationResult:
        # Use config values as defaults
        db_id = database_id or self.config.database_id
        db_name = database_name or self.config.database_name
        
        if not db_id or not db_name:
            return OperationResult(
                success=False,
                error="Database ID and name must be provided in config or as parameters",
                code="MISSING_CONFIG"
            )
        
        try:
            existing_databases = self.databases.list()
            
            for db in existing_databases.get("databases", []):
                if db["name"] == db_name or db["$id"] == db_id:
                    logger.info(f"Database '{db_name}' already exists")
                    return OperationResult(
                        success=True,
                        data=db,
                        code="EXISTS"
                    )
            
            # Only try to create if it doesn't exist
            try:
                result = self.databases.create(
                    database_id=db_id,
                    name=db_name,
                    enabled=True
                )
                
                logger.info(f"Created new database: {db_name}")
                return OperationResult(
                    success=True,
                    data=result
                )
            except AppwriteException as create_error:
                # If we hit the database limit but the database exists, that's okay
                if "maximum number of databases" in create_error.message.lower():
                    # Try to find the database again
                    existing_databases = self.databases.list()
                    for db in existing_databases.get("databases", []):
                        if db["$id"] == db_id:
                            logger.warning(f"Database limit reached, but database '{db_id}' exists")
                            return OperationResult(
                                success=True,
                                data=db,
                                code="EXISTS"
                            )
                raise create_error
        
        except AppwriteException as e:
            logger.error(f"Database operation failed: {e.message}")
            return OperationResult(
                success=False,
                error=f"Database operation failed: {e.message}",
                code=e.type
            )
    
    def _infer_attribute_type(self, value: Any) -> Tuple[str, bool]:
        is_array = isinstance(value, list)
        base_value = value[0] if is_array and value else value
        
        if isinstance(base_value, bool):
            return "boolean", is_array
        elif isinstance(base_value, int):
            return "integer", is_array
        elif isinstance(base_value, float):
            return "double", is_array
        elif isinstance(base_value, datetime):
            return "datetime", is_array
        elif isinstance(base_value, str):
            try:
                datetime.fromisoformat(base_value.replace("Z", "+00:00"))
                return "datetime", is_array
            except (ValueError, AttributeError):
                return "string", is_array
        else:
            return "string", is_array
    
    def _create_dynamic_attributes(
        self,
        database_id: str,
        collection_id: str,
        metadata: Dict[str, Any],
        max_retries: int = MAX_ATTRIBUTE_CREATION_RETRIES,
        initial_delay: float = INITIAL_RETRY_DELAY
    ) -> OperationResult:
        fixed_attributes = [
            {"key": "fileId", "type": "string", "size": 255, "required": True},
            {"key": "bucketId", "type": "string", "size": 255, "required": True},
            {"key": "filename", "type": "string", "size": 500, "required": True},
            {"key": "file_size", "type": "integer", "required": False},
            {"key": "mime_type", "type": "string", "size": 100, "required": False},
            {"key": "uploaded_at", "type": "datetime", "required": False},
            {"key": "file_hash", "type": "string", "size": 64, "required": False},
        ]
        
        delay = initial_delay
        existing_attribute_keys = set()
        
        for attempt in range(1, max_retries + 1):
            try:
                existing_attributes = self.databases.list_attributes(database_id, collection_id)
                existing_attribute_keys = {attr["key"] for attr in existing_attributes["attributes"]}
                logger.debug(f"Found {len(existing_attribute_keys)} existing attributes")
                break
            except AppwriteException as e:
                if "collection_not_found" in e.message and attempt < max_retries:
                    logger.warning(f"Collection not ready (attempt {attempt}/{max_retries}), retrying in {delay}s")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Failed to list attributes: {e.message}")
                    return OperationResult(
                        success=False,
                        error=f"Failed to list attributes: {e.message}",
                        code=e.type
                    )
        
        for attr in fixed_attributes:
            if attr["key"] in existing_attribute_keys:
                continue
            
            try:
                self._create_single_attribute(database_id, collection_id, attr)
            except AppwriteException as e:
                if "attribute already exists" not in e.message.lower():
                    logger.warning(f"Failed to create fixed attribute {attr['key']}: {e.message}")
        
        for key, value in metadata.items():
            if key in existing_attribute_keys:
                continue
            
            try:
                attr_type, is_array = self._infer_attribute_type(value)
                self._create_attribute_by_type(database_id, collection_id, key, attr_type, is_array)
            except AppwriteException as e:
                logger.error(f"Failed to create attribute '{key}': {e.message}")
        
        return OperationResult(success=True)
    
    def _create_single_attribute(self, database_id: str, collection_id: str, attr: Dict[str, Any]):
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
        is_array: bool
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
                logger.warning(f"Unsupported type {attr_type} for {key}, defaulted to string")
            
            logger.debug(f"Created {attr_type} attribute '{key}' (array: {is_array})")
            return result
        except AppwriteException as e:
            # If the attribute already exists, that's fine - just return
            if "already exists" in e.message.lower():
                logger.debug(f"Attribute '{key}' already exists")
                return None
            # Otherwise, re-raise the exception
            raise e
    
    def create_metadata_collection_if_not_exists(
        self,
        metadata: Dict[str, Any] = None,
        collection_name: str = None,
        collection_id: str = None,
        database_id: str = None
    ) -> OperationResult:
        # Use config values as defaults
        db_id = database_id or self.config.database_id
        coll_name = collection_name or self.config.collection_name
        coll_id = collection_id or self.config.collection_id
        
        if not db_id or not coll_name or not coll_id:
            return OperationResult(
                success=False,
                error="Database ID, collection name, and collection ID must be provided in config or as parameters",
                code="MISSING_CONFIG"
            )
        
        db_result = self.create_database_if_not_exists(db_id, self.config.database_name)
        if not db_result.success:
            return db_result
        
        try:
            existing_collections = self.databases.list_collections(db_id)
            
            for collection in existing_collections.get("collections", []):
                if collection["$id"] == coll_id or collection["name"] == coll_name:
                    if metadata:
                        self._create_dynamic_attributes(db_id, collection["$id"], metadata)
                    
                    return OperationResult(
                        success=True,
                        data={
                            "collection": collection,
                            "database_id": db_id,
                            "collection_id": collection["$id"]
                        },
                        code="EXISTS"
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
            
            self._create_dynamic_attributes(db_id, coll_id, metadata or {})
            
            return OperationResult(
                success=True,
                data={
                    "collection": result,
                    "database_id": db_id,
                    "collection_id": result["$id"]
                }
            )
        
        except AppwriteException as e:
            logger.error(f"Collection creation failed: {e.message}")
            return OperationResult(
                success=False,
                error=f"Collection creation failed: {e.message}",
                code=e.type
            )
        
    def search_files_by_metadata(
    self,
    filters: Dict[str, Any] = None,
    array_filters: Dict[str, Any] = None,
    collection_name: str = None,
    collection_id: str = None,
    database_id: str = None,
) -> OperationResult:
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

            result = self.databases.list_documents(db_id, coll_id, queries=queries)

            return OperationResult(
                success=True,
                data={"documents": result["documents"], "total": result["total"]},
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
            # First ensure the collection exists
            collection_result = self.create_metadata_collection_if_not_exists(
                {}, collection_name, collection_id, database_id
            )

            if not collection_result.success:
                return collection_result

            # Now search for the file by hash
            search_result = self.databases.list_documents(
                db_id, coll_id, queries=[Query.equal("file_hash", file_hash)]
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
            if "Attribute not found in schema: file_hash" in e.message:
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
                            queries=[Query.equal("file_hash", file_hash)],
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
            search_result = self.databases.list_documents(
                database_id=db_id,
                collection_id=coll_id,
                queries=[Query.equal("fileId", file_id)]
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
class AppWriteFileModule:
    def __init__(self, config: AppwriteConfig):
        # if not isinstance(config.path_manager, ModelPathManager):
        #     raise ValueError("path_manager must be an instance of ModelPathManager")
        
        self.config = config
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
    
    def _setup_cache(self) -> CacheManager:
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
    
    # def _file_exists_by_hash(
    #     self,
    #     bucket_id: str,
    #     file_hash: str,
    #     filename: str = None
    # ) -> OperationResult:
    #     try:
    #         # First try to find by hash in metadata
    #         # bucket_info = self.get_bucket(bucket_id)
    #         # if not bucket_info.success:
    #         #     return OperationResult(success=False, error="Bucket not found")
            
    #         # bucket_name = bucket_info.data["name"]
    #         # search_result = self.metadata_manager.check_file_exists_by_hash(
    #         #     bucket_name, file_hash, self.config.collection_name, self.config.collection_id, self.config.database_id
    #         # )
    #         search_result = self.metadata_manager.check_file_exists_by_hash(
    #             file_hash, 
    #             self.config.collection_name, 
    #             self.config.collection_id, 
    #             self.config.database_id
    #         )
            
    #         if search_result.success:
    #             return OperationResult(
    #                 success=True,
    #                 data=search_result.data,
    #                 code="FOUND_BY_HASH"
    #             )
            
    #         # Fallback to filename check if hash not found
    #         if filename:
    #             all_files = []
    #             offset = 0
    #             limit = DEFAULT_PAGE_LIMIT
                
    #             while True:
    #                 result = self.storage.list_files(
    #                     bucket_id, [Query.limit(limit), Query.offset(offset)]
    #                 )
    #                 files_chunk = result.get("files", [])
    #                 all_files.extend(files_chunk)
                    
    #                 if len(files_chunk) < limit:
    #                     break
    #                 offset += limit
                
    #             for file in all_files:
    #                 if file["name"] == filename:
    #                     return OperationResult(
    #                         success=True,
    #                         data=file,
    #                         code="FOUND_BY_NAME"
    #                     )
            
    #         return OperationResult(success=False, code="NOT_FOUND")
        
    #     except AppwriteException as e:
    #         return OperationResult(
    #             success=False,
    #             error=e.message,
    #             code=e.type
    #         )
    def _file_exists_by_hash(
    self,
    bucket_id: str,
    file_hash: str,
    filename: str = None
) -> OperationResult:
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
                    all_files = []
                    offset = 0
                    limit = DEFAULT_PAGE_LIMIT
                    
                    while True:
                        result = self.storage.list_files(
                            bucket_id, [Query.limit(limit), Query.offset(offset)]
                        )
                        files_chunk = result.get("files", [])
                        all_files.extend(files_chunk)
                        
                        if len(files_chunk) < limit:
                            break
                        offset += limit
                    
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
        try:
            existing_docs = self.databases.list_documents(
                database_id, collection_id, queries=[Query.equal("fileId", file_id)]
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
        """
        Upload a file to Appwrite storage and store its metadata in a database collection.

        Args:
            bucket_id: The ID of the bucket to upload to
            file_path: Path to the file to upload
            filename: Name to give the file in storage
            metadata: Dictionary of metadata to store
            file_id: Optional file ID (if None, one will be generated)
            permissions: Optional list of permissions for the file
            collection_name: Optional collection name (defaults to config)
            collection_id: Optional collection ID (defaults to config)

        Returns:
            OperationResult with success status and data/error information
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

        # CRITICAL FIX: Verify file exists in BOTH metadata AND storage
        should_update_metadata_only = False
        if existing_metadata.success and existing_metadata.code == "FOUND_BY_HASH" and not file_id:
            existing_file_id = existing_metadata.data.get("fileId")
            
            # Verify the file actually exists in storage
            if existing_file_id:
                file_check = self.get_file(bucket_id, existing_file_id)
                
                if file_check.success:
                    # File exists in both metadata and storage
                    should_update_metadata_only = self.config.allow_metadata_only_updates
                    logger.info(f"File {existing_file_id} exists in both metadata and storage")
                else:
                    # Metadata exists but file missing from storage - clean up metadata
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
                        except Exception as e:
                            logger.warning(f"Failed to delete orphaned metadata: {str(e)}")

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
                # Delete the old file from storage
                delete_result = self.delete_file(bucket_id, old_file_id)
                if not delete_result.success:
                    logger.warning(f"Failed to delete old file from storage: {delete_result.error}")
                    # Continue anyway - the upload might still work

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

        # Ensure metadata infrastructure exists
        collection_result = self.metadata_manager.create_metadata_collection_if_not_exists(
            metadata, collection_name, collection_id, self.config.database_id
        )
        if not collection_result.success:
            return OperationResult(
                success=False,
                error=collection_result.error,
                code=collection_result.code
            )

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
        database_id = collection_result.data.get("database_id") or self.config.database_id
        coll_id = collection_result.data.get("collection_id") or collection_id

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

    
    # def upload_file_with_metadata(
    #     self,
    #     bucket_id: str,
    #     file_path: str,
    #     filename: str,
    #     metadata: Dict[str, Any],
    #     file_id: str = None,
    #     permissions: List[str] = None,
    #     collection_name: str = None,
    #     collection_id: str = None
    # ) -> OperationResult:
    #     # Use defaults from config if not provided
    #     if collection_name is None:
    #         collection_name = self.config.collection_name
    #     if collection_id is None:
    #         collection_id = self.config.collection_id
        
    #     # Calculate file hash for metadata
    #     file_hash = self._calculate_file_hash(file_path=file_path)
        
    #     # Check if file already exists by hash
    #     existing_metadata = self.metadata_manager.check_file_exists_by_hash(
    #         file_hash, collection_name, collection_id, self.config.database_id
    #     )
        
    #     if existing_metadata.success and not file_id:
    #         # File already exists - update its metadata instead of uploading again
    #         logger.info(f"File with hash {file_hash} already exists, updating metadata only")
            
    #         existing_file_id = existing_metadata.data.get("fileId")
            
    #         # Ensure collection exists with new metadata fields
    #         collection_result = self.metadata_manager.create_metadata_collection_if_not_exists(
    #             metadata, collection_name, collection_id, self.config.database_id
    #         )
    #         if not collection_result.success:
    #             return OperationResult(
    #                 success=False,
    #                 error=collection_result.error,
    #                 code=collection_result.code
    #             )
            
    #         # Update the metadata
    #         metadata_update = metadata.copy()
    #         metadata_update["file_hash"] = file_hash
    #         metadata_update["filename"] = filename
    #         metadata_update["uploaded_at"] = datetime.now().isoformat()
            
    #         update_result = self.metadata_manager.update_file_metadata(
    #             file_id=existing_file_id,
    #             metadata_updates=metadata_update,
    #             collection_name=collection_name,
    #             collection_id=collection_id,
    #             database_id=self.config.database_id
    #         )
            
    #         if update_result.success:
    #             # Get the full file info to return
    #             file_info = self.get_file(bucket_id, existing_file_id)
    #             return OperationResult(
    #                 success=True,
    #                 data={
    #                     **(file_info.data if file_info.success else {}),
    #                     "metadata": update_result.data,
    #                     "metadata_action": "UPDATED"
    #                 },
    #                 code="EXISTS_METADATA_UPDATED"
    #             )
    #         else:
    #             return OperationResult(
    #                 success=False,
    #                 error=f"Failed to update metadata: {update_result.error}",
    #                 code="METADATA_UPDATE_FAILED"
    #             )
        
    #     # Ensure metadata infrastructure exists
    #     collection_result = self.metadata_manager.create_metadata_collection_if_not_exists(
    #         metadata, collection_name, collection_id, self.config.database_id
    #     )
    #     if not collection_result.success:
    #         return OperationResult(
    #             success=False,
    #             error=collection_result.error,
    #             code=collection_result.code
    #         )
        
    #     # Add file_hash to metadata
    #     metadata["file_hash"] = file_hash
        
    #     # Upload file
    #     upload_result = self.upload_file(bucket_id, file_path, file_id, permissions)
    #     if not upload_result.success:
    #         return upload_result
        
    #     file_id = upload_result.data["$id"]
    #     database_id = collection_result.data["database_id"]
    #     coll_id = collection_result.data["collection_id"]
        
    #     # Create and store metadata
    #     try:
    #         metadata_document = self._build_metadata_document(
    #             file_id, bucket_id, filename, {"data": upload_result.data}, metadata, file_hash
    #         )
            
    #         metadata_result = self._store_metadata_document(
    #             database_id, coll_id, file_id, metadata_document
    #         )
            
    #         if metadata_result.success:
    #             upload_result.data["metadata"] = metadata_result.data
    #             upload_result.data["metadata_action"] = metadata_result.code
            
    #         return OperationResult(
    #             success=True,
    #             data=upload_result.data,
    #             code="CREATED_WITH_METADATA"
    #         )
        
    #     except AppwriteException as e:
    #         logger.error(f"Metadata handling failed: {e.message}")
    #         return OperationResult(
    #             success=True,
    #             data=upload_result.data,
    #             error=f"Metadata handling failed: {e.message}",
    #             code="METADATA_ERROR"
    #         )
    
#     def upload_file_from_bytes_with_metadata(
#     self,
#     bucket_id: str,
#     file_bytes: bytes,
#     filename: str,
#     metadata: Dict[str, Any],
#     file_id: str = None,
#     permissions: List[str] = None,
#     collection_name: str = None,
#     collection_id: str = None
# ) -> OperationResult:
#         # Use defaults from config if not provided
#         if collection_name is None:
#             collection_name = self.config.collection_name
#         if collection_id is None:
#             collection_id = self.config.collection_id
        
#         # Calculate file hash for metadata
#         file_hash = self._calculate_file_hash(file_bytes=file_bytes)
        
#         # Check if file already exists by hash - REMOVED bucket_name parameter
#         existing_metadata = self.metadata_manager.check_file_exists_by_hash(
#             file_hash, collection_name, collection_id, self.config.database_id
#         )
        
#         if existing_metadata.success and not file_id:
#             # File already exists, return existing metadata
#             return OperationResult(
#                 success=True,
#                 data=existing_metadata.data,
#                 code="EXISTS"
#             )
        
#         # Ensure metadata infrastructure exists - REMOVED bucket_name parameter
#         collection_result = self.metadata_manager.create_metadata_collection_if_not_exists(
#             metadata, collection_name, collection_id, self.config.database_id
#         )
#         if not collection_result.success:
#             return OperationResult(
#                 success=False,
#                 error=collection_result.error,
#                 code=collection_result.code
#             )
        
#         # Add file_hash to metadata
#         metadata["file_hash"] = file_hash
        
#         # Upload file
#         upload_result = self.upload_file_from_bytes(
#             bucket_id, file_bytes, filename, file_id, permissions
#         )
#         if not upload_result.success:
#             return upload_result
        
#         file_id = upload_result.data["$id"]
#         database_id = collection_result.data["database_id"]
#         coll_id = collection_result.data["collection_id"]
        
#         # Create and store metadata
#         try:
#             metadata_document = self._build_metadata_document(
#                 file_id, bucket_id, filename, {"data": upload_result.data}, metadata, file_hash
#             )
            
#             metadata_result = self._store_metadata_document(
#                 database_id, coll_id, file_id, metadata_document
#             )
            
#             if metadata_result.success:
#                 upload_result.data["metadata"] = metadata_result.data
#                 upload_result.data["metadata_action"] = metadata_result.code
            
#             return OperationResult(
#                 success=True,
#                 data=upload_result.data,
#                 code="CREATED_WITH_METADATA"
#             )
        
#         except AppwriteException as e:
#             logger.error(f"Metadata handling failed: {e.message}")
#             return OperationResult(
#                 success=True,
#                 data=upload_result.data,
#                 error=f"Metadata handling failed: {e.message}",
#                 code="METADATA_ERROR"
#             )
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
            
            # Ensure collection exists with new metadata fields
            collection_result = self.metadata_manager.create_metadata_collection_if_not_exists(
                metadata, collection_name, collection_id, self.config.database_id
            )
            if not collection_result.success:
                return OperationResult(
                    success=False,
                    error=collection_result.error,
                    code=collection_result.code
                )
            
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
        collection_result = self.metadata_manager.create_metadata_collection_if_not_exists(
            metadata, collection_name, collection_id, self.config.database_id
        )
        if not collection_result.success:
            return OperationResult(
                success=False,
                error=collection_result.error,
                code=collection_result.code
            )
        
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
        database_id = collection_result.data["database_id"]
        coll_id = collection_result.data["collection_id"]
        
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
            # Rollback: delete the uploaded file if metadata fails
            try:
                self.delete_file(bucket_id, file_id)
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
        try:
            result = self.storage.get_bucket(bucket_id)
            return OperationResult(success=True, data=result)
        
        except AppwriteException as e:
            # Check if this is a bucket not found error
            if "Storage bucket with the requested ID could not be found" in e.message:
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
    
    def create_bucket(
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
        create_metadata_db: bool = True
    ) -> OperationResult:
        # Use default name from config if not provided
        if name is None:
            name = self.config.bucket_name
        
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
                antivirus=antivirus
            )
            
            # Automatically create metadata database if requested
            if create_metadata_db:
                db_result = self.metadata_manager.create_database_if_not_exists(name, self.config.database_id)
                if db_result.success:
                    result["metadata_database"] = db_result.data
            
            return OperationResult(
                success=True,
                data=result,
                code="CREATED"
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )
    
    def get_current_user(self) -> OperationResult:
        if not isinstance(self.auth_manager, SessionAuth):
            return OperationResult(
                success=False,
                error="User information requires session authentication",
                code="AUTH_METHOD_ERROR"
            )
        
        try:
            user = self.auth_manager.account.get()
            return OperationResult(
                success=True,
                data={
                    "user_id": user["$id"],
                    "email": user.get("email"),
                    "name": user.get("name"),
                    "email_verified": user.get("emailVerification", False)
                }
            )
        
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=e.message,
                code=e.type
            )
    
    def get_user_preferences(self, user_id: Optional[str] = None) -> OperationResult:
        try:
            if isinstance(self.auth_manager, SessionAuth):
                preferences = self.auth_manager.account.get_prefs()
                return OperationResult(
                    success=True,
                    data=preferences,
                    code="USER_SESSION"
                )
            else:
                if not user_id:
                    return OperationResult(
                        success=False,
                        error="user_id parameter required for API key authentication",
                        code="MISSING_USER_ID"
                    )
                
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
        return self.cache_manager.clear_cache(bucket_id, older_than_hours)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache_manager.get_stats()
    
    def debug_collection_attributes(
        self,
        collection_id: str = None,
        database_id: str = None
    ) -> OperationResult:
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