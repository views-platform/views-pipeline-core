import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from views_pipeline_core.managers.prediction import (
    PredictionMetadata,
    PredictionStoreManager,
)
from views_pipeline_core.managers.appwrite import (
    AppwriteConfig,
    OperationResult,
    AuthMethod,
)


# Test PredictionMetadata
class TestPredictionMetadata:
    def test_valid_initialization(self):
        """Test creating PredictionMetadata with valid parameters"""
        metadata = PredictionMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1", "target2"],
            description="Test description"
        )
        
        assert metadata.loa == "country"
        assert metadata.name == "test_model"
        assert metadata.type == "fatalities"
        assert metadata.targets == ["target1", "target2"]
        assert metadata.description == "Test description"

    def test_initialization_without_description(self):
        """Test creating PredictionMetadata without optional description"""
        metadata = PredictionMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"]
        )
        
        assert metadata.description is None

    def test_invalid_loa_type(self):
        """Test that non-string loa raises TypeError"""
        with pytest.raises(TypeError, match="loa must be a string"):
            PredictionMetadata(
                loa=123,
                name="test_model",
                type="fatalities",
                targets=["target1"]
            )

    def test_invalid_name_type(self):
        """Test that non-string name raises TypeError"""
        with pytest.raises(TypeError, match="name must be a string"):
            PredictionMetadata(
                loa="country",
                name=["invalid"],
                type="fatalities",
                targets=["target1"]
            )

    def test_invalid_type_type(self):
        """Test that non-string type raises TypeError"""
        with pytest.raises(TypeError, match="type must be a string"):
            PredictionMetadata(
                loa="country",
                name="test_model",
                type=123,
                targets=["target1"]
            )

    def test_invalid_targets_not_list(self):
        """Test that non-list targets raises TypeError"""
        with pytest.raises(TypeError, match="targets must be a list of strings"):
            PredictionMetadata(
                loa="country",
                name="test_model",
                type="fatalities",
                targets="target1"
            )

    def test_invalid_targets_non_string_elements(self):
        """Test that list with non-string elements raises TypeError"""
        with pytest.raises(TypeError, match="targets must be a list of strings"):
            PredictionMetadata(
                loa="country",
                name="test_model",
                type="fatalities",
                targets=[1, 2, 3]
            )

    def test_invalid_description_type(self):
        """Test that non-string, non-None description raises TypeError"""
        with pytest.raises(TypeError, match="description must be a string or None"):
            PredictionMetadata(
                loa="country",
                name="test_model",
                type="fatalities",
                targets=["target1"],
                description=123
            )

    def test_to_dict_with_description(self):
        """Test to_dict method with description"""
        metadata = PredictionMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1", "target2"],
            description="Test description"
        )
        
        result = metadata.to_dict()
        
        assert result == {
            "loa": "country",
            "name": "test_model",
            "type": "fatalities",
            "targets": ["target1", "target2"],
            "description": "Test description"
        }

    def test_to_dict_without_description(self):
        """Test to_dict method without description"""
        metadata = PredictionMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"]
        )
        
        result = metadata.to_dict()
        
        assert result == {
            "loa": "country",
            "name": "test_model",
            "type": "fatalities",
            "targets": ["target1"]
        }
        assert "description" not in result


# Fixtures for PredictionStoreManager tests
@pytest.fixture
def mock_path_manager():
    """Mock ModelPathManager"""
    manager = Mock()
    manager.cache = Path("/tmp/test_cache")
    manager.model_name = "test_model"
    return manager


@pytest.fixture
def mock_config(mock_path_manager):
    """Mock AppwriteConfig"""
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials="test_api_key",
        auth_method=AuthMethod.API_KEY,
        bucket_id="test_bucket",
        bucket_name="Test Bucket",
        collection_name="Test Collection",
        collection_id="test_collection",
        database_id="test_database",
        cache_dir="/tmp/test_cache",
        path_manager=mock_path_manager,
    )


@pytest.fixture
def mock_appwrite_manager():
    """Mock AppWriteFileManager"""
    with patch("views_pipeline_core.managers.prediction.AppWriteFileManager") as mock:
        manager = mock.return_value
        
        # Mock metadata_manager
        manager.metadata_manager = Mock()
        
        yield manager


@pytest.fixture
def prediction_store(mock_config, mock_appwrite_manager):
    """Create PredictionStoreManager instance with mocked dependencies"""
    with patch("views_pipeline_core.managers.prediction.AppWriteFileManager", return_value=mock_appwrite_manager):
        store = PredictionStoreManager(mock_config)
        yield store


# Test PredictionStoreManager
class TestPredictionStoreManager:
    def test_initialization(self, mock_config, mock_appwrite_manager):
        """Test PredictionStoreManager initialization"""
        with patch("views_pipeline_core.managers.prediction.AppWriteFileManager", return_value=mock_appwrite_manager):
            store = PredictionStoreManager(mock_config)
            
            assert store.model_path == mock_config.path_manager
            assert store._PredictionStoreManager__appwrite_file_managerconfig == mock_config
            assert store._PredictionStoreManager__appwrite_file_manager == mock_appwrite_manager

    def test_upload_predictions_from_path_success(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test successful upload from file path"""
        # Create a test file
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test content")
        
        # Mock successful upload
        mock_appwrite_manager.upload_file_with_metadata.return_value = OperationResult(
            success=True,
            data={"$id": "file123", "name": "test.parquet"},
            code="CREATED"
        )
        
        result = prediction_store.upload_predictions(
            file=test_file,
            filename="test.parquet",
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1", "target2"],
            description="Test upload"
        )
        
        assert result.success
        assert result.data["$id"] == "file123"
        
        # Verify the metadata passed to upload
        call_args = mock_appwrite_manager.upload_file_with_metadata.call_args
        assert call_args[1]["filename"] == "test.parquet"
        assert call_args[1]["metadata"]["loa"] == "country"
        assert call_args[1]["metadata"]["name"] == "test_model"
        assert call_args[1]["metadata"]["type"] == "fatalities"
        assert call_args[1]["metadata"]["targets"] == ["target1", "target2"]

    def test_upload_predictions_from_string_path(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test upload from string path"""
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test content")
        
        mock_appwrite_manager.upload_file_with_metadata.return_value = OperationResult(
            success=True,
            data={"$id": "file123"},
            code="CREATED"
        )
        
        result = prediction_store.upload_predictions(
            file=str(test_file),
            filename="test.parquet",
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"]
        )
        
        assert result.success

    def test_upload_predictions_dataframe_not_implemented(self, prediction_store):
        """Test that uploading DataFrame raises NotImplementedError"""
        df = pd.DataFrame({"col": [1, 2, 3]})
        
        with pytest.raises(NotImplementedError, match="Uploading a DataFrame directly is not implemented"):
            prediction_store.upload_predictions(
                file=df,
                filename="test.parquet",
                loa="country",
                name="test_model",
                type="fatalities",
                targets=["target1"]
            )

    def test_upload_predictions_invalid_file_type(self, prediction_store):
        """Test that invalid file type raises TypeError"""
        with pytest.raises(TypeError, match="file must be a Path, str, or pd.DataFrame"):
            prediction_store.upload_predictions(
                file=123,
                filename="test.parquet",
                loa="country",
                name="test_model",
                type="fatalities",
                targets=["target1"]
            )

    def test_upload_predictions_bucket_not_found_creates_bucket(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test that bucket is created when not found"""
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test content")
        
        # First upload fails with bucket not found
        mock_appwrite_manager.upload_file_with_metadata.side_effect = [
            OperationResult(
                success=False,
                error="Bucket not found",
                code="storage_bucket_not_found"
            ),
            OperationResult(
                success=True,
                data={"$id": "file123"},
                code="CREATED"
            )
        ]
        
        # Mock successful bucket creation
        mock_appwrite_manager.create_bucket.return_value = OperationResult(
            success=True,
            data={"$id": "test_bucket"},
            code="CREATED"
        )
        
        result = prediction_store.upload_predictions(
            file=test_file,
            filename="test.parquet",
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"]
        )
        
        assert result.success
        mock_appwrite_manager.create_bucket.assert_called_once()

    def test_upload_predictions_bucket_creation_fails(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test handling of bucket creation failure"""
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test content")
        
        # Upload fails with bucket not found
        mock_appwrite_manager.upload_file_with_metadata.return_value = OperationResult(
            success=False,
            error="Bucket not found",
            code="storage_bucket_not_found"
        )
        
        # Bucket creation fails
        mock_appwrite_manager.create_bucket.side_effect = Exception("Creation failed")
        
        result = prediction_store.upload_predictions(
            file=test_file,
            filename="test.parquet",
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"]
        )
        
        assert not result.success
        assert "Creation failed" in result.error

    def test_get_predictions_by_metadata_with_filters(self, prediction_store, mock_appwrite_manager):
        """Test getting predictions with filters"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=True,
            data={
                "documents": [
                    {"fileId": "file1", "$createdAt": "2025-10-22T12:00:00.000Z"},
                    {"fileId": "file2", "$createdAt": "2025-10-22T11:00:00.000Z"},
                ],
                "total": 2
            }
        )
        
        filters = {"type": "fatalities", "loa": "country"}
        result = prediction_store.get_predictions_by_metadata(filters=filters)
        
        assert len(result) == 2
        # Should be sorted by creation date, newest first
        assert result[0]["fileId"] == "file1"
        assert result[1]["fileId"] == "file2"
        
        # Verify filters include model name
        call_args = mock_appwrite_manager.metadata_manager.search_files_by_metadata.call_args
        assert call_args[1]["filters"]["name"] == "test_model"
        assert call_args[1]["filters"]["type"] == "fatalities"

    def test_get_predictions_by_metadata_no_filters(self, prediction_store, mock_appwrite_manager):
        """Test getting predictions without additional filters"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=True,
            data={
                "documents": [{"fileId": "file1", "$createdAt": "2025-10-22T12:00:00.000Z"}],
                "total": 1
            }
        )
        
        result = prediction_store.get_predictions_by_metadata()
        
        assert len(result) == 1
        
        # Should still include model name filter
        call_args = mock_appwrite_manager.metadata_manager.search_files_by_metadata.call_args
        assert call_args[1]["filters"]["name"] == "test_model"

    def test_get_predictions_by_metadata_search_fails(self, prediction_store, mock_appwrite_manager):
        """Test handling of search failure"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=False,
            error="Search failed",
            code="ERROR"
        )
        
        result = prediction_store.get_predictions_by_metadata()
        
        assert result == []

    def test_get_predictions_by_metadata_no_results(self, prediction_store, mock_appwrite_manager):
        """Test getting predictions when no results found"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=True,
            data={"documents": [], "total": 0}
        )
        
        result = prediction_store.get_predictions_by_metadata()
        
        assert result == []

    def test_download_prediction(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test downloading a prediction file"""
        save_path = tmp_path / "downloaded.parquet"
        
        mock_appwrite_manager.download_file.return_value = OperationResult(
            success=True,
            data={"save_path": str(save_path)},
            code="SAVED"
        )
        
        result = prediction_store.download_prediction(
            file_id="file123",
            save_path=str(save_path),
            use_cache=True,
            validate_cache=True
        )
        
        assert result.success
        assert result.data["save_path"] == str(save_path)
        
        # Verify correct parameters passed
        mock_appwrite_manager.download_file.assert_called_once_with(
            bucket_id="test_bucket",
            file_id="file123",
            save_path=str(save_path),
            use_cache=True,
            validate_cache=True
        )

    def test_download_prediction_no_save_path(self, prediction_store, mock_appwrite_manager):
        """Test downloading prediction to memory"""
        mock_appwrite_manager.download_file.return_value = OperationResult(
            success=True,
            data={"file_bytes": b"test content"},
            code="RETURNED"
        )
        
        result = prediction_store.download_prediction(file_id="file123")
        
        assert result.success
        assert result.data["file_bytes"] == b"test content"

    def test_get_latest_file_id_success(self, prediction_store, mock_appwrite_manager):
        """Test getting latest file ID"""
        prediction_store.get_predictions_by_metadata = Mock(
            return_value=[
                {"fileId": "file1", "$createdAt": "2025-10-22T12:00:00.000Z"},
                {"fileId": "file2", "$createdAt": "2025-10-22T11:00:00.000Z"},
            ]
        )
        
        file_id = prediction_store.get_latest_file_id(filters={"type": "fatalities"})
        
        assert file_id == "file1"

    def test_get_latest_file_id_no_files(self, prediction_store):
        """Test getting latest file ID when no files found"""
        prediction_store.get_predictions_by_metadata = Mock(return_value=[])
        
        file_id = prediction_store.get_latest_file_id(filters={"type": "fatalities"})
        
        assert file_id is None

    def test_get_latest_file_id_missing_fileid(self, prediction_store):
        """Test getting latest file ID when fileId field is missing"""
        prediction_store.get_predictions_by_metadata = Mock(
            return_value=[{"$createdAt": "2025-10-22T12:00:00.000Z"}]
        )
        
        file_id = prediction_store.get_latest_file_id(filters={})
        
        assert file_id is None

    def test_download_latest_file_success(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test downloading the latest file"""
        save_path = tmp_path / "latest.parquet"
        
        prediction_store.get_latest_file_id = Mock(return_value="file123")
        mock_appwrite_manager.download_file.return_value = OperationResult(
            success=True,
            data={"save_path": str(save_path)},
            code="SAVED"
        )
        
        result = prediction_store.download_latest_file(
            filters={"type": "fatalities"},
            save_path=str(save_path)
        )
        
        assert result.success
        prediction_store.get_latest_file_id.assert_called_once_with(filters={"type": "fatalities"})

    def test_download_latest_file_no_files_found(self, prediction_store):
        """Test downloading latest file when none found"""
        prediction_store.get_latest_file_id = Mock(return_value=None)
        
        with pytest.raises(FileNotFoundError, match="No files found matching the given filters"):
            prediction_store.download_latest_file(filters={"type": "fatalities"})

    def test_download_latest_file_with_empty_filters(self, prediction_store, mock_appwrite_manager):
        """Test downloading latest file with empty filters"""
        prediction_store.get_latest_file_id = Mock(return_value="file123")
        mock_appwrite_manager.download_file.return_value = OperationResult(
            success=True,
            data={"file_bytes": b"content"},
            code="RETURNED"
        )
        
        result = prediction_store.download_latest_file(filters={})
        
        assert result.success

    def test_get_file_metadata_success(self, prediction_store, mock_appwrite_manager):
        """Test getting file metadata"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=True,
            data={
                "documents": [
                    {
                        "fileId": "file123",
                        "filename": "test.parquet",
                        "loa": "country",
                        "type": "fatalities"
                    }
                ]
            }
        )
        
        result = prediction_store.get_file_metadata("file123")
        
        assert result.success
        assert result.data["fileId"] == "file123"
        assert result.code == "FOUND"

    def test_get_file_metadata_not_found(self, prediction_store, mock_appwrite_manager):
        """Test getting metadata for non-existent file"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=True,
            data={"documents": []}
        )
        
        result = prediction_store.get_file_metadata("nonexistent")
        
        assert not result.success
        assert result.code == "NOT_FOUND"

    def test_get_file_metadata_search_fails(self, prediction_store, mock_appwrite_manager):
        """Test getting metadata when search fails"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=False,
            error="Search failed",
            code="ERROR"
        )
        
        result = prediction_store.get_file_metadata("file123")
        
        assert not result.success
        assert "Search failed" in result.error

    def test_update_prediction_metadata(self, prediction_store, mock_appwrite_manager):
        """Test updating prediction metadata"""
        mock_appwrite_manager.metadata_manager.update_file_metadata.return_value = OperationResult(
            success=True,
            data={"$id": "doc123"},
            code="UPDATED"
        )
        
        result = prediction_store.update_prediction_metadata(
            file_id="file123",
            metadata_updates={"description": "Updated description"}
        )
        
        assert result.success
        assert result.code == "UPDATED"
        
        mock_appwrite_manager.metadata_manager.update_file_metadata.assert_called_once_with(
            file_id="file123",
            metadata_updates={"description": "Updated description"},
            collection_name="Test Collection",
            collection_id="test_collection",
            database_id="test_database"
        )

    def test_delete_prediction_success(self, prediction_store, mock_appwrite_manager):
        """Test deleting a prediction"""
        mock_appwrite_manager.delete_file.return_value = OperationResult(
            success=True,
            code="DELETED"
        )
        
        result = prediction_store.delete_prediction("file123")
        
        assert result.success
        assert result.code == "DELETED"
        
        mock_appwrite_manager.delete_file.assert_called_once_with(
            bucket_id="test_bucket",
            file_id="file123"
        )

    def test_delete_prediction_fails(self, prediction_store, mock_appwrite_manager):
        """Test deletion failure"""
        mock_appwrite_manager.delete_file.return_value = OperationResult(
            success=False,
            error="Deletion failed",
            code="ERROR"
        )
        
        result = prediction_store.delete_prediction("file123")
        
        assert not result.success

    def test_list_all_predictions(self, prediction_store):
        """Test listing all predictions for current model"""
        prediction_store.get_predictions_by_metadata = Mock(
            return_value=[
                {"fileId": "file1", "name": "test_model"},
                {"fileId": "file2", "name": "test_model"},
            ]
        )
        
        result = prediction_store.list_all_predictions()
        
        assert len(result) == 2
        prediction_store.get_predictions_by_metadata.assert_called_once_with(
            filters={"name": "test_model"}
        )

    def test_list_all_predictions_unfiltered(self, prediction_store, mock_appwrite_manager):
        """Test listing all predictions without filters"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=True,
            data={
                "documents": [
                    {"fileId": "file1", "$createdAt": "2025-10-22T13:00:00.000Z"},
                    {"fileId": "file2", "$createdAt": "2025-10-22T12:00:00.000Z"},
                    {"fileId": "file3", "$createdAt": "2025-10-22T11:00:00.000Z"},
                ],
                "total": 3
            }
        )
        
        result = prediction_store.list_all_predictions_unfiltered()
        
        assert len(result) == 3
        # Should be sorted by creation date, newest first
        assert result[0]["fileId"] == "file1"
        assert result[1]["fileId"] == "file2"
        assert result[2]["fileId"] == "file3"
        
        # Verify no filters were passed
        call_args = mock_appwrite_manager.metadata_manager.search_files_by_metadata.call_args
        assert call_args[1]["filters"] is None

    def test_list_all_predictions_unfiltered_search_fails(self, prediction_store, mock_appwrite_manager):
        """Test unfiltered list when search fails"""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
            success=False,
            error="Search failed",
            code="ERROR"
        )
        
        result = prediction_store.list_all_predictions_unfiltered()
        
        assert result == []


# Integration-style tests
class TestPredictionStoreManagerIntegration:
    def test_full_upload_workflow(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test complete upload workflow including bucket creation"""
        test_file = tmp_path / "prediction.parquet"
        test_file.write_text("prediction data")
        
        # First upload fails, then bucket is created, then upload succeeds
        mock_appwrite_manager.upload_file_with_metadata.side_effect = [
            OperationResult(success=False, code="storage_bucket_not_found"),
            OperationResult(success=True, data={"$id": "file123"}, code="CREATED")
        ]
        mock_appwrite_manager.create_bucket.return_value = OperationResult(
            success=True, data={"$id": "test_bucket"}
        )
        
        result = prediction_store.upload_predictions(
            file=test_file,
            filename="prediction.parquet",
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1", "target2"],
            description="Full workflow test"
        )
        
        assert result.success
        assert mock_appwrite_manager.create_bucket.call_count == 1
        assert mock_appwrite_manager.upload_file_with_metadata.call_count == 2

    def test_search_and_download_workflow(self, prediction_store, mock_appwrite_manager, tmp_path):
        """Test searching for predictions and downloading the latest"""
        # Mock search results
        prediction_store.get_predictions_by_metadata = Mock(
            return_value=[
                {"fileId": "file123", "$createdAt": "2025-10-22T12:00:00.000Z"},
            ]
        )
        
        # Mock download
        save_path = tmp_path / "downloaded.parquet"
        mock_appwrite_manager.download_file.return_value = OperationResult(
            success=True,
            data={"save_path": str(save_path)},
            code="SAVED"
        )
        
        # Search for files
        files = prediction_store.get_predictions_by_metadata(filters={"type": "fatalities"})
        assert len(files) == 1
        
        # Download latest
        result = prediction_store.download_prediction(
            file_id=files[0]["fileId"],
            save_path=str(save_path)
        )
        
        assert result.success
        assert result.data["save_path"] == str(save_path)