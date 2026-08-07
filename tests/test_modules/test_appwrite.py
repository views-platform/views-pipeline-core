import json
import logging

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

from views_pipeline_core.modules.appwrite import (
    AppWriteFileModule,
    AppwriteConfig,
    AuthMethod,
    OperationResult,
    CacheValidationResult,
    CacheManager,
    CacheMetadata,  # Add this import
    AppwriteMetadataHandler,
    AuthFactory,
    ApiKeyAuth,
)
from appwrite.exception import AppwriteException


# Fixtures
@pytest.fixture
def mock_path_manager():
    """Mock ModelPathManager"""
    manager = Mock()
    manager.cache = Path("/tmp/test_cache")
    return manager


@pytest.fixture
def api_key_config(mock_path_manager):
    """Basic API key configuration"""
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials="test_api_key",
        auth_method=AuthMethod.API_KEY,
        cache_dir="/tmp/test_cache",
        path_manager=mock_path_manager,
        bucket_id="test_bucket",
        bucket_name="Test Bucket",
        collection_id="test_collection",
        collection_name="Test Collection",
        database_id="test_database",
        database_name="Test Database",
    )




@pytest.fixture
def mock_client():
    """Mock Appwrite Client"""
    with patch("views_pipeline_core.modules.appwrite.storage.Client") as mock:
        client = mock.return_value
        client.set_endpoint.return_value = client
        client.set_project.return_value = client
        client.set_key.return_value = client
        yield client


@pytest.fixture
def mock_storage():
    """Mock Storage service"""
    with patch("views_pipeline_core.modules.appwrite.storage.Storage") as mock:
        yield mock.return_value


@pytest.fixture
def mock_databases():
    """Mock Databases service"""
    with patch("views_pipeline_core.modules.appwrite.storage.Databases") as mock:
        yield mock.return_value


@pytest.fixture
def mock_account():
    """Mock Account service"""
    with patch("views_pipeline_core.modules.appwrite.Account") as mock:
        yield mock.return_value


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


# Test AppwriteConfig
class TestAppwriteConfig:
    def test_coordinates_have_no_defaults(self, mock_path_manager):
        """#324/C-229 — these used to default to the live production coordinates.

        Full coverage of the replacement contract lives in
        test_appwrite_config_coordinates.py; this pins the headline here, where the
        old "defaults apply" test used to sit.
        """
        from views_pipeline_core.exceptions.exceptions import ConfigurationException

        with pytest.raises(ConfigurationException) as exc:
            AppwriteConfig(
                endpoint="https://cloud.appwrite.io/v1",
                project_id="test_project",
                credentials="test_key",
                path_manager=mock_path_manager,
            )
        assert "bucket_id" in str(exc.value)

    def test_config_with_custom_values(self, mock_path_manager):
        config = AppwriteConfig(
            endpoint="https://cloud.appwrite.io/v1",
            project_id="test_project",
            credentials="test_key",
            cache_ttl_hours=48,
            path_manager=mock_path_manager,
            bucket_id="my_custom_bucket",
            bucket_name="My Custom Bucket",
            collection_id="my_collection",
            collection_name="My Collection",
            database_id="my_database",
            database_name="My Database",
        )

        assert config.bucket_name == "My Custom Bucket"
        assert config.database_name == "My Database"
        assert config.collection_name == "My Collection"
        assert config.cache_ttl_hours == 48

    def test_config_auth_method_string_conversion(self, mock_path_manager):
        config = AppwriteConfig(
            endpoint="https://cloud.appwrite.io/v1",
            project_id="test_project",
            credentials="test_key",
            auth_method="api_key",
            path_manager=mock_path_manager,
            bucket_id="test_bucket",
            bucket_name="Test Bucket",
            collection_id="test_collection",
            collection_name="Test Collection",
            database_id="test_database",
            database_name="Test Database",
        )
        
        assert isinstance(config.auth_method, AuthMethod)
        assert config.auth_method == AuthMethod.API_KEY


# Test Authentication
class TestAuthFactory:
    def test_create_api_key_auth(self):
        auth = AuthFactory.create_auth(AuthMethod.API_KEY)
        assert isinstance(auth, ApiKeyAuth)


    def test_unsupported_auth_method(self):
        with pytest.raises(ValueError):
            AuthFactory.create_auth("invalid_method")


class TestApiKeyAuth:
    def test_setup_success(self, mock_client):
        auth = ApiKeyAuth()
        result = auth.setup(mock_client, "test_api_key")
        
        assert result.success
        mock_client.set_key.assert_called_once_with("test_api_key")

    def test_setup_invalid_credentials(self, mock_client):
        auth = ApiKeyAuth()
        result = auth.setup(mock_client, {"invalid": "type"})
        
        assert not result.success
        assert result.code == "INVALID_CREDENTIALS"




# Test CacheManager
class TestCacheManager:
    def test_cache_manager_initialization(self, temp_cache_dir):
        cache_ttl = timedelta(hours=24)
        manager = CacheManager(temp_cache_dir, cache_ttl)
        
        assert manager.cache_dir == temp_cache_dir
        assert manager.cache_ttl == cache_ttl
        assert manager.cache_metadata == {}

    def test_add_to_cache(self, temp_cache_dir):
        manager = CacheManager(temp_cache_dir, timedelta(hours=24))
        
        test_file = temp_cache_dir / "test.txt"
        test_file.write_text("test content")
        
        manager.add_to_cache(
            "bucket1",
            "file123",
            test_file,
            {"name": "test.txt", "$updatedAt": "2025-10-22T12:00:00.000Z"}
        )
        
        cache_key = "bucket1_file123"
        assert cache_key in manager.cache_metadata
        assert manager.cache_metadata[cache_key].filename == "test.txt"

    def test_validate_cache_valid(self, temp_cache_dir):
        manager = CacheManager(temp_cache_dir, timedelta(hours=24))
        
        test_file = temp_cache_dir / "bucket1" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test content")
        
        manager.add_to_cache("bucket1", "file123", test_file)
        
        result = manager.validate_cache("bucket1", "file123")
        assert result == CacheValidationResult.VALID

    def test_validate_cache_not_found(self, temp_cache_dir):
        manager = CacheManager(temp_cache_dir, timedelta(hours=24))
        
        result = manager.validate_cache("bucket1", "nonexistent")
        assert result == CacheValidationResult.NOT_FOUND

    def test_validate_cache_invalid_ttl(self, temp_cache_dir):
        manager = CacheManager(temp_cache_dir, timedelta(hours=1))
        
        test_file = temp_cache_dir / "bucket1" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test content")
        
        # Fix: Use CacheMetadata object instead of dict
        cache_key = "bucket1_file123"
        manager.cache_metadata[cache_key] = CacheMetadata(
            bucket_id="bucket1",
            file_id="file123",
            path=str(test_file),
            cached_at=(datetime.now() - timedelta(hours=2)).isoformat(),
            size_bytes=test_file.stat().st_size,
            filename="test.txt",
        )
        
        result = manager.validate_cache("bucket1", "file123")
        assert result == CacheValidationResult.INVALID_TTL

    def test_remove_from_cache(self, temp_cache_dir):
        manager = CacheManager(temp_cache_dir, timedelta(hours=24))
        
        test_file = temp_cache_dir / "bucket1" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test content")
        
        manager.add_to_cache("bucket1", "file123", test_file)
        assert "bucket1_file123" in manager.cache_metadata
        
        manager.remove_from_cache("bucket1", "file123")
        assert "bucket1_file123" not in manager.cache_metadata
        assert not test_file.exists()

    def test_clear_cache_all(self, temp_cache_dir):
        manager = CacheManager(temp_cache_dir, timedelta(hours=24))
        
        # Add multiple files
        for i in range(3):
            test_file = temp_cache_dir / "bucket1" / f"test{i}.txt"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(f"content {i}")
            manager.add_to_cache("bucket1", f"file{i}", test_file)
        
        result = manager.clear_cache()
        
        assert result.success
        assert result.data["deleted_files"] == 3
        assert len(manager.cache_metadata) == 0

    def test_get_cache_stats(self, temp_cache_dir):
        manager = CacheManager(temp_cache_dir, timedelta(hours=24))
        
        test_file = temp_cache_dir / "bucket1" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test content")
        
        manager.add_to_cache("bucket1", "file123", test_file)
        
        stats = manager.get_stats()
        
        assert stats["total_files"] == 1
        assert "bucket1" in stats["by_bucket"]
        assert stats["by_bucket"]["bucket1"]["files"] == 1


# Test AppwriteMetadataHandler
class TestMetadataManager:
    @pytest.fixture
    def metadata_manager(self, mock_databases, api_key_config):
        return AppwriteMetadataHandler(mock_databases, api_key_config)

    def test_search_files_by_metadata(self, metadata_manager, mock_databases):
        # `return_value` cannot express paging: it hands back the same page for every
        # offset, which is a substrate that ignores `offset` — and the walk added in
        # #341 correctly refuses to certify such a read. A side_effect that empties out
        # is the minimum faithful double. Real paging behaviour is covered in
        # tests/test_modules/test_appwrite_pagination.py against a substrate double
        # built from the SDK's own query encoding.
        pages = [
            {"documents": [{"fileId": "file123", "filename": "test.txt"}], "total": 1},
            {"documents": [], "total": 1},
        ]
        mock_databases.list_documents.side_effect = pages

        result = metadata_manager.search_files_by_metadata(
            filters={"filename": "test.txt"}
        )

        assert result.success
        assert result.data["total"] == 1
        assert len(result.data["documents"]) == 1


# Test AppWriteFileModule
class TestAppWriteFileManager:
    @pytest.fixture
    def file_manager(self, api_key_config):
        with patch("views_pipeline_core.modules.appwrite.storage.Client"), \
             patch("views_pipeline_core.modules.appwrite.storage.Storage"), \
             patch("views_pipeline_core.modules.appwrite.storage.Databases"), \
             patch("views_pipeline_core.modules.appwrite.storage.Users"):
            manager = AppWriteFileModule(api_key_config)
            yield manager

    def test_calculate_file_hash_from_path(self, file_manager, tmp_path):
        test_file = tmp_path / "test.txt"
        test_content = b"test content"
        test_file.write_bytes(test_content)
        
        file_hash = file_manager._calculate_file_hash(file_path=str(test_file))
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        assert file_hash == expected_hash

    def test_calculate_file_hash_from_bytes(self, file_manager):
        test_content = b"test content"
        
        file_hash = file_manager._calculate_file_hash(file_bytes=test_content)
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        assert file_hash == expected_hash

    def test_upload_file_success(self, file_manager, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        file_manager.storage.create_file.return_value = {
            "$id": "file123",
            "name": "test.txt",
            "sizeOriginal": 12,
        }
        
        with patch("views_pipeline_core.modules.appwrite.storage.InputFile"):
            result = file_manager.upload_file(
                "bucket1",
                str(test_file),
                check_duplicates=False
            )
        
        assert result.success
        assert result.data["$id"] == "file123"

    def test_upload_file_duplicate_exists(self, file_manager, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Mock duplicate detection
        file_manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(
                success=True,
                data={"$id": "existing_file"},
                code="FOUND"
            )
        )
        
        result = file_manager.upload_file(
            "bucket1",
            str(test_file),
            check_duplicates=True,
            overwrite=False
        )
        
        assert result.success
        assert result.code == "EXISTS"

    def test_upload_file_from_bytes(self, file_manager):
        test_content = b"test content"
        
        file_manager.storage.create_file.return_value = {
            "$id": "file123",
            "name": "test.txt",
            "sizeOriginal": len(test_content),
        }
        
        with patch("views_pipeline_core.modules.appwrite.storage.InputFile"):
            result = file_manager.upload_file_from_bytes(
                "bucket1",
                test_content,
                "test.txt",
                check_duplicates=False
            )
        
        assert result.success
        assert result.data["$id"] == "file123"

    def test_download_file_from_cache(self, file_manager, temp_cache_dir):
        # Setup cache
        file_manager.cache_manager.validate_cache = Mock(
            return_value=CacheValidationResult.VALID
        )
        file_manager.cache_manager.get_cached_file_path = Mock(
            return_value=OperationResult(
                success=True,
                data={"cache_path": str(temp_cache_dir / "test.txt")}
            )
        )
        
        cached_file = temp_cache_dir / "test.txt"
        cached_file.write_text("cached content")
        
        result = file_manager.download_file("bucket1", "file123", use_cache=True)
        
        assert result.success
        assert result.data["from_cache"]

    def test_download_file_from_remote(self, file_manager):
        file_manager.storage.get_file_download.return_value = b"remote content"
        file_manager.get_file = Mock(
            return_value=OperationResult(
                success=True,
                data={"name": "test.txt", "$updatedAt": "2025-10-22T12:00:00Z"}
            )
        )
        
        result = file_manager.download_file("bucket1", "file123", use_cache=False)
        
        assert result.success
        assert result.data["file_bytes"] == b"remote content"
        assert not result.data["from_cache"]


    def test_download_file_json_dict_coerced_to_bytes(self, file_manager, caplog):
        """#310: the Appwrite SDK returns a PARSED DICT for application/json
        files (e.g. ADR-013 wire manifests); download_file must coerce to bytes
        instead of crashing at the cache write — and must say, loudly, that the
        re-serialized bytes are not byte-identical to the stored artifact."""
        manifest = {
            "contract_version": "1.5",
            "run_id": "run0",
            "shards": [{"name": "a.tap.zip", "sha256": "abc", "time_id": 543}],
            "sidecar_sha256": None,
        }
        file_manager.storage.get_file_download.return_value = manifest
        file_manager.get_file = Mock(
            return_value=OperationResult(
                success=True,
                data={"name": "run0__lr_ged_sb__manifest.json",
                      "$updatedAt": "2026-07-27T12:00:00Z"},
            )
        )

        with caplog.at_level(logging.WARNING):
            result = file_manager.download_file(
                "bucket1", "manifest1", use_cache=False
            )

        assert result.success
        payload = result.data["file_bytes"]
        assert isinstance(payload, bytes)
        assert json.loads(payload) == manifest  # semantic round-trip
        assert "not byte-identical" in caplog.text  # fidelity caveat is loud

    def test_download_file_bytes_payload_emits_no_fidelity_warning(
        self, file_manager, caplog
    ):
        """The coercion branch must not touch the normal binary path."""
        file_manager.storage.get_file_download.return_value = b"binary shard"
        file_manager.get_file = Mock(
            return_value=OperationResult(
                success=True,
                data={"name": "a.tap.zip", "$updatedAt": "2026-07-27T12:00:00Z"},
            )
        )
        with caplog.at_level(logging.WARNING):
            result = file_manager.download_file("bucket1", "shard1", use_cache=False)
        assert result.success
        assert result.data["file_bytes"] == b"binary shard"
        assert "not byte-identical" not in caplog.text

    def test_list_files(self, file_manager):
        file_manager.storage.list_files.return_value = {
            "files": [
                {"$id": "file1", "name": "test1.txt"},
                {"$id": "file2", "name": "test2.txt"},
            ],
            "total": 2,
        }
        
        result = file_manager.list_files("bucket1")
        
        assert result.success
        assert len(result.data["files"]) == 2
        assert result.data["total"] == 2

    def test_delete_file(self, file_manager):
        file_manager.storage.delete_file.return_value = {}
        file_manager.cache_manager.remove_from_cache = Mock()
        
        result = file_manager.delete_file("bucket1", "file123")
        
        assert result.success
        assert result.code == "DELETED"
        file_manager.cache_manager.remove_from_cache.assert_called_once_with(
            "bucket1", "file123"
        )

    def test_get_file(self, file_manager):
        file_manager.storage.get_file.return_value = {
            "$id": "file123",
            "name": "test.txt",
            "sizeOriginal": 100,
        }
        
        result = file_manager.get_file("bucket1", "file123")
        
        assert result.success
        assert result.data["$id"] == "file123"

    def test_upload_file_with_metadata(self, file_manager, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Mock the upload and metadata creation
        file_manager.upload_file = Mock(
            return_value=OperationResult(
                success=True,
                data={"$id": "file123", "name": "test.txt", "sizeOriginal": 12}
            )
        )
        # Containers are verified, never created, since #331.
        file_manager._require_containers = Mock(return_value=None)
        file_manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(success=False, code="NOT_FOUND")
        )
        file_manager._store_metadata_document = Mock(
            return_value=OperationResult(success=True, data={})
        )
        
        metadata = {"custom_field": "custom_value"}
        
        result = file_manager.upload_file_with_metadata(
            "bucket1",
            str(test_file),
            "test.txt",
            metadata
        )
        
        assert result.success

    def test_clear_cache(self, file_manager):
        file_manager.cache_manager.clear_cache = Mock(
            return_value=OperationResult(
                success=True,
                data={"deleted_files": 5, "deleted_bytes": 1000}
            )
        )
        
        result = file_manager.clear_cache()
        
        assert result.success
        assert result.data["deleted_files"] == 5

    def test_get_cache_stats(self, file_manager):
        file_manager.cache_manager.get_stats = Mock(
            return_value={
                "total_files": 10,
                "total_size_bytes": 5000,
                "total_size_mb": 0.005,
            }
        )
        
        stats = file_manager.get_cache_stats()
        
        assert stats["total_files"] == 10
        assert stats["total_size_mb"] == 0.005


# Test Error Handling
class TestErrorHandling:
    def test_appwrite_exception_handling(self, api_key_config):
        with patch("views_pipeline_core.modules.appwrite.storage.Client"), \
             patch("views_pipeline_core.modules.appwrite.storage.Storage") as mock_storage, \
             patch("views_pipeline_core.modules.appwrite.storage.Databases"), \
             patch("views_pipeline_core.modules.appwrite.storage.Users"):
            
            manager = AppWriteFileModule(api_key_config)
            mock_storage.return_value.get_file.side_effect = AppwriteException(
                "File not found", 404, "storage_file_not_found"
            )
            
            result = manager.get_file("bucket1", "nonexistent")
            
            assert not result.success
            assert result.code == "storage_file_not_found"



# Test OperationResult
class TestOperationResult:
    def test_operation_result_success(self):
        result = OperationResult(success=True, data={"key": "value"})
        
        assert result.success
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_operation_result_failure(self):
        result = OperationResult(
            success=False,
            error="Something went wrong",
            code="ERROR_CODE"
        )
        
        assert not result.success
        assert result.error == "Something went wrong"
        assert result.code == "ERROR_CODE"

    def test_operation_result_to_dict(self):
        result = OperationResult(
            success=True,
            data={"test": "data"},
            code="SUCCESS"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["data"] == {"test": "data"}
        assert result_dict["code"] == "SUCCESS"