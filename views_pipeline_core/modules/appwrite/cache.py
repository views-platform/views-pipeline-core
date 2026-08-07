"""cache.py — extracted from modules/appwrite/file.py (M-1 audit decision).

This module contains the classes that were previously in the
2,841-LOC God module `file.py`. The original `file.py` is now a
re-export shim that preserves all existing import paths.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import shutil
import json
import logging
from views_pipeline_core.modules.appwrite.config import AppwriteConfig, OperationResult

logger = logging.getLogger(__name__)


class CacheValidationResult(Enum):
    """Results of cache validation checks.

    Used by CacheManager.validate_cache() to indicate cache state.

    Attributes:
        VALID: Cache entry exists, is within TTL, and matches remote timestamp.
        INVALID_TTL: Cache entry exists but has exceeded the TTL period.
        INVALID_TIMESTAMP: Cache entry exists but remote file was updated after caching.
        NOT_FOUND: No cache entry exists for the requested file.

    Example:
        >>> validation = cache_manager.validate_cache(bucket_id, file_id)
        >>> if validation == CacheValidationResult.VALID:
        ...     # Use cached file
        ... else:
        ...     # Download fresh copy
    """

    VALID = "valid"
    INVALID_TTL = "invalid_ttl"
    INVALID_TIMESTAMP = "invalid_timestamp"
    NOT_FOUND = "not_found"

# Type Definitions


@dataclass
class CacheMetadata:
    """Metadata for cached files.

    Stores information about cached files to enable validation and management.

    Attributes:
        bucket_id: ID of the bucket the file belongs to.
        file_id: Appwrite file ID.
        path: Local filesystem path to the cached file.
        cached_at: ISO format timestamp when file was cached.
        size_bytes: Size of the cached file in bytes.
        filename: Original filename.
        remote_updated_at: Remote file's last update timestamp for validation.
    """

    bucket_id: str
    file_id: str
    path: str
    cached_at: str
    size_bytes: int
    filename: str
    remote_updated_at: Optional[str] = None


class CacheManager:
    """Local file cache manager with TTL-based validation.

    Manages a local cache of downloaded files to reduce network requests.
    Supports TTL-based expiration and timestamp validation against remote files.

    The cache stores files organized by bucket ID and maintains a metadata JSON
    file to track cached files, their timestamps, and sizes.

    Attributes:
        cache_dir: Root directory for cached files.
        cache_ttl: Time-to-live duration for cached files.
        cache_metadata_file: Path to the JSON metadata file.
        cache_metadata: Dictionary mapping cache keys to CacheMetadata objects.

    Example:
        >>> from datetime import timedelta
        >>> cache = CacheManager(Path("/tmp/cache"), timedelta(hours=24))
        >>>
        >>> # Check if file is in valid cache
        >>> result = cache.validate_cache("bucket1", "file123")
        >>> if result == CacheValidationResult.VALID:
        ...     cached_path = cache.get_cached_file_path("bucket1", "file123")
    """

    def __init__(self, cache_dir: Path, cache_ttl: timedelta):
        """Initialize cache manager with directory and TTL settings.

        Args:
            cache_dir: Directory to store cached files. Will be created if needed.
            cache_ttl: Maximum age of cached files before they're considered stale.
        """
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.cache_metadata_file = cache_dir / "cache_metadata.json"
        self.cache_metadata: Dict[str, CacheMetadata] = {}
        self._load_cache_metadata()

    def _load_cache_metadata(self):
        """Load cache metadata from JSON file on disk.

        Reads the cache_metadata.json file and populates the cache_metadata
        dictionary. Silently handles missing or corrupted files.
        """
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
        """Save cache metadata to JSON file on disk.

        Persists the current cache_metadata dictionary to disk for
        recovery across sessions.
        """
        try:
            data = {k: v.__dict__ for k, v in self.cache_metadata.items()}
            with open(self.cache_metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, bucket_id: str, file_id: str) -> str:
        """Generate a unique cache key for a file.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.

        Returns:
            Combined key string in format 'bucket_id_file_id'.
        """
        return f"{bucket_id}_{file_id}"

    def _get_cache_path(self, bucket_id: str, file_id: str, filename: str = None) -> Path:
        """Get the filesystem path for a cached file.

        Creates the bucket subdirectory if it doesn't exist.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
            filename: Optional filename to use. Defaults to file_id.

        Returns:
            Path object pointing to the cache file location.
        """
        bucket_cache_dir = self.cache_dir / bucket_id
        bucket_cache_dir.mkdir(exist_ok=True)
        
        if filename:
            return bucket_cache_dir / filename
        return bucket_cache_dir / file_id

    def validate_cache(self, bucket_id: str, file_id: str, remote_updated_at: str = None) -> CacheValidationResult:
        """Check if a cached file is valid and usable.

        Validates cache entries based on:
        1. Existence in cache metadata
        2. Physical file existence on disk
        3. TTL expiration
        4. Remote file timestamp (if provided)

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
            remote_updated_at: Optional ISO timestamp of remote file's last update.
                If provided and newer than cache time, returns INVALID_TIMESTAMP.

        Returns:
            CacheValidationResult enum value:
                - VALID: Cache entry is usable
                - NOT_FOUND: No cache entry exists
                - INVALID_TTL: Cache has expired
                - INVALID_TIMESTAMP: Remote file is newer than cache

        Example:
            >>> result = cache.validate_cache("bucket1", "file123", "2024-01-15T10:00:00")
            >>> if result == CacheValidationResult.VALID:
            ...     # Safe to use cached file
        """
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
        """Add or update a file in the cache.

        Records cache metadata for a downloaded file. Call this after
        successfully downloading a file from remote storage.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
            file_path: Path where the file is stored locally.
            file_metadata: Optional metadata from Appwrite including
                'name' and '$updatedAt' fields.

        Example:
            >>> cache.add_to_cache(
            ...     "my_bucket",
            ...     "file123",
            ...     Path("/cache/my_bucket/data.parquet"),
            ...     {"name": "data.parquet", "$updatedAt": "2024-01-15T10:00:00Z"}
            ... )
        """
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
        """Remove a file from the cache.

        Deletes both the cached file from disk and its metadata entry.
        Silently handles missing files.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.
        """
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
        """Get the local filesystem path of a cached file.

        Args:
            bucket_id: Storage bucket identifier.
            file_id: File identifier.

        Returns:
            OperationResult with:
                - success=True and data containing 'cache_path' and 'metadata'
                - success=False if file not in cache or file missing from disk

        Example:
            >>> result = cache.get_cached_file_path("bucket1", "file123")
            >>> if result.success:
            ...     path = result.data['cache_path']
        """
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
        """Clear cached files matching specified criteria.

        Removes cached files from disk and metadata. Can filter by bucket
        and/or age to selectively clear cache.

        Args:
            bucket_id: Optional bucket ID to limit clearing to. If None,
                clears all buckets.
            older_than_hours: Optional age filter. Only clears files cached
                more than this many hours ago. If None, clears regardless of age.

        Returns:
            OperationResult with data containing:
                - deleted_files: Count of files deleted
                - deleted_bytes: Total bytes freed
                - errors: List of any deletion errors, or None

        Example:
            >>> # Clear all cache older than 48 hours
            >>> result = cache.clear_cache(older_than_hours=48)
            >>> print(f"Freed {result.data['deleted_bytes']} bytes")
            >>>
            >>> # Clear all cache for a specific bucket
            >>> result = cache.clear_cache(bucket_id="old_bucket")
        """
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
        """Get cache statistics and usage information.

        Returns:
            Dictionary containing:
                - total_files: Number of cached files
                - total_size_bytes: Total cache size in bytes
                - total_size_mb: Total cache size in megabytes
                - cache_dir: Path to cache directory
                - by_bucket: Dict mapping bucket IDs to {files, bytes}

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Cache: {stats['total_files']} files, {stats['total_size_mb']}MB")
            >>> for bucket, info in stats['by_bucket'].items():
            ...     print(f"  {bucket}: {info['files']} files")
        """
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