from views_faoapi.managers.model import APIManager, APIPathManager
import logging

from views_faoapi.managers.appwrite import AppWriteFileManager, AppwriteConfig

from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional, Dict
from contextlib import asynccontextmanager
import os
import io
import uvicorn
import signal
import sys
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

class FAOApiManager(APIManager):
    """
    Manages the FAO API lifecycle including startup, shutdown, and maintenance.
    """

    def __init__(
        self,
        model_path: APIPathManager,
        wandb_notifications: bool = False,
    ) -> None:
        """
        Initializes the FAOApiManager.

        Args:
            model_path (APIPathManager): The path manager for the API.
            wandb_notifications (bool, optional): Enable or disable Weights & Biases notifications. Defaults to False.
        """
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
        )

        logger.info(f"{str(self._model_path.dotenv)}")
        load_dotenv(dotenv_path=self._model_path.dotenv)
        
        # Cache for AppWriteFileManager instances (keyed by API key hash)
        self._manager_cache: Dict[str, AppWriteFileManager] = {}
        
        # Define lifespan context manager before creating FastAPI app
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup logic
            logger.info("FAO API is starting up...")
            # Add any startup logic here (e.g., cache warming, DB connections)
            
            yield
            
            # Shutdown logic
            logger.info("FAO API is shutting down...")
            # Clear manager cache
            self._manager_cache.clear()
            logger.info("Cleared AppWrite manager cache")
        
        self.app = FastAPI(
            title="AppWrite File Server",
            description="FastAPI service for retrieving files from Appwrite",
            version="1.0.0",
            lifespan=lifespan
        )
        
        self.appwrite_manager: Optional[AppWriteFileManager] = None
        self._server_config = {
            "host": self.configs.get("host", "0.0.0.0"),
            "port": int(self.configs.get("port", 80)),
            "reload": self.configs.get("reload", "false").lower() == "true",
            "workers": int(self.configs.get("workers", 1)),
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register routes
        self._register_routes()

    def _get_appwrite_manager(self, x_api_key: str = Header(..., description="Appwrite API Key")) -> AppWriteFileManager:
        """
        Dependency to get or create an AppWriteFileManager instance for the provided API key.
        Managers are cached to avoid recreating them for each request.
        """
        cache_key = x_api_key
        
        if cache_key not in self._manager_cache:
            endpoint = os.getenv("APPWRITE_ENDPOINT")
            project_id = os.getenv("APPWRITE_DATASTORE_PROJECT_ID")
            
            if not project_id:
                raise HTTPException(
                    status_code=500,
                    detail="Appwrite configuration missing. Set APPWRITE_PROJECT_ID environment variable."
                )
            
            try:
                # Create AppwriteConfig with the new structure
                config = AppwriteConfig(
                    endpoint=endpoint,
                    project_id=project_id,
                    credentials=x_api_key,
                    auth_method="api_key",
                    path_manager=self._model_path,
                    cache_dir=str(self._model_path.cache / f".appwrite_cache_{hash(x_api_key) % 10000}"),
                    cache_ttl_hours=24,
                    bucket_id=os.getenv("APPWRITE_UNFAO_BUCKET_ID"),
                    bucket_name=os.getenv("APPWRITE_UNFAO_BUCKET_NAME"),
                    collection_name=os.getenv("APPWRITE_UNFAO_COLLECTION_NAME"),
                    collection_id=os.getenv("APPWRITE_UNFAO_COLLECTION_ID"),
                    database_name=os.getenv("APPWRITE_DATABASE_NAME"),
                    database_id=os.getenv("APPWRITE_DATABASE_ID")
                )
                
                manager = AppWriteFileManager(config)
                
                # Validate the API key by making a simple API call
                validation_result = manager.list_buckets(limit=1)
                if not validation_result.success:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Invalid API key: {validation_result.get('error', 'Authentication failed')}"
                    )
                
                self._manager_cache[cache_key] = manager
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid API key or Appwrite configuration: {str(e)}"
                )
        
        return self._manager_cache[cache_key]

    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "AppWrite File Server", 
                "note": "Include 'X-API-Key' header with your Appwrite API key in all requests",
                "endpoints": {
                    "files": "/files/{bucket_id}",
                    "file_download": "/files/{bucket_id}/{file_id}/download",
                    "file_cached": "/files/{bucket_id}/{file_id}/cached",
                    "file_info": "/files/{bucket_id}/{file_id}/info",
                    "cache_stats": "/cache/stats",
                    "clear_cache": "/cache (DELETE)",
                    "health": "/health"
                }
            }

        @self.app.get("/files/{bucket_id}")
        async def list_files(
            bucket_id: str,
            limit: int = Query(100, ge=1, le=1000),
            offset: int = Query(0, ge=0),
            search: Optional[str] = None,
            manager: AppWriteFileManager = Depends(self._get_appwrite_manager)
        ):
            """List all files in a bucket with pagination."""
            try:
                queries = []
                if search:
                    queries.append(f"search('name','{search}')")
                
                result = manager.list_files(
                    bucket_id=bucket_id,
                    queries=queries,
                    limit=limit,
                    offset=offset
                )
                
                if not result.success:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Error listing files: {result.get('error', 'Unknown error')}"
                    )
                
                return {
                    "bucket_id": bucket_id,
                    "files": result.data.get("files", []),
                    "total": result.data.get("total", 0),
                    "pagination": {
                        "limit": limit,
                        "offset": offset
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/files/{bucket_id}/{file_id}/info")
        async def get_file_info(
            bucket_id: str,
            file_id: str,
            manager: AppWriteFileManager = Depends(self._get_appwrite_manager)
        ):
            """Get file metadata without downloading the file."""
            try:
                result = manager.get_file(bucket_id, file_id)
                
                if not result.success:
                    raise HTTPException(
                        status_code=404,
                        detail=f"File not found: {result.get('error', 'Unknown error')}"
                    )
                
                return {
                    "file_id": file_id,
                    "bucket_id": bucket_id,
                    "metadata": result.data
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/files/{bucket_id}/{file_id}/download")
        async def download_file(
            bucket_id: str,
            file_id: str,
            use_cache: bool = Query(True, description="Use cached version if available"),
            download: bool = Query(False, description="Force file download with attachment header"),
            manager: AppWriteFileManager = Depends(self._get_appwrite_manager)
        ):
            """Download a file from Appwrite storage."""
            try:
                file_info = manager.get_file(bucket_id, file_id)
                if not file_info.success:
                    raise HTTPException(
                        status_code=404,
                        detail=f"File not found: {file_info.get('error', 'Unknown error')}"
                    )
                
                metadata = file_info.data
                filename = metadata.get('name', file_id)
                
                result = manager.download_file(
                    bucket_id=bucket_id,
                    file_id=file_id,
                    use_cache=use_cache
                )
                
                if not result.success:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Download failed: {result.get('error', 'Unknown error')}"
                    )
                
                # Determine media type
                media_type = "application/octet-stream"
                file_extension = os.path.splitext(filename)[1].lower()
                extension_to_media_type = {
                    '.pdf': 'application/pdf',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.txt': 'text/plain',
                    '.json': 'application/json',
                    '.zip': 'application/zip',
                }
                media_type = extension_to_media_type.get(file_extension, "application/octet-stream")
                
                file_bytes = result.data.get("file_bytes")
                
                if download:
                    return StreamingResponse(
                        io.BytesIO(file_bytes),
                        media_type=media_type,
                        headers={"Content-Disposition": f"attachment; filename={filename}"}
                    )
                else:
                    return StreamingResponse(
                        io.BytesIO(file_bytes),
                        media_type=media_type,
                        headers={"Content-Disposition": f"inline; filename={filename}"}
                    )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/files/{bucket_id}/{file_id}/cached")
        async def get_cached_file(
            bucket_id: str,
            file_id: str,
            manager: AppWriteFileManager = Depends(self._get_appwrite_manager)
        ):
            """Get file from cache if available (fastest option)."""
            try:
                result = manager.cache_manager.get_cached_file_path(bucket_id, file_id)
                
                if not result.success:
                    raise HTTPException(
                        status_code=404,
                        detail=f"File not in cache: {result.get('error', 'Unknown error')}"
                    )
                
                file_info = manager.get_file(bucket_id, file_id)
                if file_info.success:
                    filename = file_info.data.get('name', file_id)
                else:
                    filename = file_id
                
                return FileResponse(
                    path=result.data.get("cache_path"),
                    filename=filename,
                    media_type='application/octet-stream'
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/cache/stats")
        async def get_cache_stats(manager: AppWriteFileManager = Depends(self._get_appwrite_manager)):
            """Get cache statistics for the current API key."""
            try:
                stats = manager.get_cache_stats()
                return stats
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/cache")
        async def clear_cache(
            bucket_id: Optional[str] = None,
            older_than_hours: Optional[int] = None,
            manager: AppWriteFileManager = Depends(self._get_appwrite_manager)
        ):
            """Clear file cache for the current API key."""
            try:
                result = manager.clear_cache(
                    bucket_id=bucket_id,
                    older_than_hours=older_than_hours
                )
                return result.to_dict()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_check(manager: AppWriteFileManager = Depends(self._get_appwrite_manager)):
            """Health check to verify Appwrite connection."""
            try:
                result = manager.list_buckets(limit=1)
                return {
                    "status": "healthy",
                    "appwrite_connected": result.success,
                    "cache_stats": manager.get_cache_stats()
                }
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Service unhealthy: {str(e)}"
                )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._is_running = False
        self._shutdown()
        sys.exit(0)

    def _startup(self):
        """Initialize and start the API server."""
        logger.info("Starting FAO API server...")
        
        try:
            # Start the server
            logger.info(f"Starting server on {self._server_config['host']}:{self._server_config['port']}")
            uvicorn.run(
                self.app,
                host=self._server_config["host"],
                port=self._server_config["port"],
                reload=self._server_config["reload"],
                workers=self._server_config["workers"],
                log_level="info"
            )
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise

    def _shutdown(self):
        """Gracefully shutdown the API server."""
        logger.info("Shutting down FAO API server...")
        
        try:
            # Clear manager cache
            self._manager_cache.clear()
            if hasattr(self._model_path, 'cache') and self._model_path.cache.exists():
                    import shutil
                    shutil.rmtree(self._model_path.cache)
                    self._model_path.cache.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared cache")
                
            # Close any open file handles or database connections
            logger.info("Cleaning up resources...")
            
            self._is_running = False
            logger.info("FAO API server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    def _health_check(self):
        """Perform health checks on the API server."""
        logger.info("Performing health check...")
        
        health_status = {
            "status": "healthy",
            "checks": {}
        }
        
        try:
            # Check if server is running
            health_status["checks"]["server"] = "running" if self._is_running else "stopped"
            
            # Check manager cache
            health_status["checks"]["manager_cache_size"] = len(self._manager_cache)
            
            # Check cache directory
            if hasattr(self._model_path, 'cache'):
                cache_exists = self._model_path.cache.exists()
                health_status["checks"]["cache"] = "available" if cache_exists else "unavailable"
                
            logger.info(f"Health check complete: {health_status}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            return health_status

    def _maintenance(self):
        """Perform maintenance tasks on the API."""
        logger.info("Starting maintenance tasks...")
        
        try:
            # Clear cache if specified in configs
            if self.configs.get('clear_cache', False):
                logger.info("Clearing cache...")
                if hasattr(self._model_path, 'cache') and self._model_path.cache.exists():
                    import shutil
                    shutil.rmtree(self._model_path.cache)
                    self._model_path.cache.mkdir(parents=True, exist_ok=True)
                    logger.info("Cache cleared successfully")
            
            # Clear manager cache
            if self.configs.get('clear_manager_cache', False):
                logger.info("Clearing manager cache...")
                self._manager_cache.clear()
                logger.info("Manager cache cleared successfully")
                
            logger.info("Maintenance tasks completed successfully")
            
        except Exception as e:
            logger.error(f"Maintenance tasks failed: {e}")
            raise