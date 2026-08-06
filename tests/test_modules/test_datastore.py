import logging
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import pandas as pd

from views_pipeline_core.modules.datastore import (
    FileMetadata,
    DatastoreModule,
)
from views_pipeline_core.modules.datastore.datastore import MetadataSearchIncomplete
from views_pipeline_core.modules.appwrite.file import (
    AppwriteConfig,
    OperationResult,
    AuthMethod,
)


# Test FileMetadata
class TestPredictionMetadata:
    def test_valid_initialization(self):
        """Test creating FileMetadata with valid parameters"""
        metadata = FileMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1", "target2"],
            category="forecast",
            description="Test description"
        )
        
        assert metadata.loa == "country"
        assert metadata.name == "test_model"
        assert metadata.type == "fatalities"
        assert metadata.targets == ["target1", "target2"]
        assert metadata.category == "forecast"
        assert metadata.description == "Test description"

    def test_initialization_without_description(self):
        """Test creating FileMetadata without optional description"""
        metadata = FileMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"],
            category="historical"
        )
        
        assert metadata.description is None
        assert metadata.category == "historical"

    def test_invalid_loa_type(self):
        """Test that non-string loa raises TypeError"""
        with pytest.raises(TypeError, match="loa must be a string"):
            FileMetadata(
                loa=123,
                name="test_model",
                type="fatalities",
                targets=["target1"],
                category="forecast"
            )

    def test_invalid_name_type(self):
        """Test that non-string name raises TypeError"""
        with pytest.raises(TypeError, match="name must be a string"):
            FileMetadata(
                loa="country",
                name=["invalid"],
                type="fatalities",
                targets=["target1"],
                category="forecast"
            )

    def test_invalid_type_type(self):
        """Test that non-string type raises TypeError"""
        with pytest.raises(TypeError, match="type must be a string"):
            FileMetadata(
                loa="country",
                name="test_model",
                type=123,
                targets=["target1"],
                category="forecast"
            )

    def test_invalid_targets_not_list(self):
        """Test that non-list targets raises TypeError"""
        with pytest.raises(TypeError, match="targets must be a list of strings"):
            FileMetadata(
                loa="country",
                name="test_model",
                type="fatalities",
                targets="target1",
                category="forecast"
            )

    def test_invalid_targets_non_string_elements(self):
        """Test that list with non-string elements raises TypeError"""
        with pytest.raises(TypeError, match="targets must be a list of strings"):
            FileMetadata(
                loa="country",
                name="test_model",
                type="fatalities",
                targets=[1, 2, 3],
                category="forecast"
            )

    def test_invalid_description_type(self):
        """Test that non-string, non-None description raises TypeError"""
        with pytest.raises(TypeError, match="description must be a string or None"):
            FileMetadata(
                loa="country",
                name="test_model",
                type="fatalities",
                targets=["target1"],
                category="forecast",
                description=123
            )

    def test_invalid_category_value(self):
        """Test that invalid category value raises ValueError"""
        with pytest.raises(ValueError, match="category must be either 'forecast' or 'historical'"):
            FileMetadata(
                loa="country",
                name="test_model",
                type="fatalities",
                targets=["target1"],
                category="invalid_category"
            )

    def test_to_dict_with_description(self):
        """Test to_dict method with description"""
        metadata = FileMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1", "target2"],
            category="forecast",
            description="Test description"
        )
        
        result = metadata.to_dict()
        
        assert result == {
            "loa": "country",
            "name": "test_model",
            "type": "fatalities",
            "targets": ["target1", "target2"],
            "category": "forecast",
            "description": "Test description"
        }

    def test_to_dict_without_description(self):
        """Test to_dict method without description"""
        metadata = FileMetadata(
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"],
            category="historical"
        )
        
        result = metadata.to_dict()
        
        assert result == {
            "loa": "country",
            "name": "test_model",
            "type": "fatalities",
            "targets": ["target1"],
            "category": "historical"
        }
        assert "description" not in result


# Fixtures for DatastoreModule tests
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
def mock_appwrite_manager():
    """Mock AppWriteFileModule"""
    manager = MagicMock()
    
    # Mock metadata_manager
    manager.metadata_manager = MagicMock()
    manager.metadata_manager.create_metadata_collection_if_not_exists.return_value = OperationResult(
        success=True,
        data={"collection_id": "test_collection", "database_id": "test_database"},
        code="EXISTS"
    )
    manager.metadata_manager.search_files_by_metadata.return_value = OperationResult(
        success=True,
        data={"documents": [], "total": 0}
    )
    
    # Mock upload methods
    manager.upload_file_with_metadata.return_value = OperationResult(
        success=True,
        data={"$id": "file123"},
        code="CREATED"
    )
    
    # Mock download methods
    manager.download_file.return_value = OperationResult(
        success=True,
        data={"file_bytes": b"test content"},
        code="RETURNED_FROM_REMOTE"
    )
    
    # Mock bucket methods
    manager.create_bucket.return_value = OperationResult(
        success=True,
        data={"$id": "test_bucket"},
        code="CREATED"
    )
    
    return manager


@pytest.fixture
def prediction_store(mock_config, mock_appwrite_manager):
    """Create DatastoreModule instance with mocked dependencies"""
    # Patch at the point where DatastoreModule imports AppWriteFileModule
    with patch("views_pipeline_core.modules.datastore.datastore.AppWriteFileModule", return_value=mock_appwrite_manager):
        store = DatastoreModule(mock_config)
        yield store


# Test DatastoreModule
class TestPredictionStoreManager:
    def test_initialization(self, mock_config, mock_appwrite_manager):
        """Test DatastoreModule initialization"""
        with patch("views_pipeline_core.modules.datastore.datastore.AppWriteFileModule", return_value=mock_appwrite_manager):
            store = DatastoreModule(mock_config)
            
            assert store.model_path == mock_config.path_manager
            assert store._DatastoreModule__appwrite_file_manager_config == mock_config
            assert store._DatastoreModule__appwrite_file_manager == mock_appwrite_manager

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
            category="forecast",
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
        assert call_args[1]["metadata"]["category"] == "forecast"

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
            targets=["target1"],
            category="historical"
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
                targets=["target1"],
                category="forecast"
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
                targets=["target1"],
                category="forecast"
            )

    def test_missing_bucket_is_reported_not_created(self, prediction_store, mock_appwrite_manager, tmp_path):
        """#331 — the delivery path no longer provisions its own destination.

        Previously a `storage_bucket_not_found` made `upload_data` CREATE the bucket
        and retry into it, so a mistyped or renamed coordinate silently published the
        forecast to a brand-new bucket nobody reads (register C-228). The failure must
        now surface, and the bucket must not be created.
        """
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test content")

        mock_appwrite_manager.upload_file_with_metadata.return_value = OperationResult(
            success=False,
            error="Bucket not found",
            code="storage_bucket_not_found",
        )

        result = prediction_store.upload_predictions(
            file=test_file,
            filename="test.parquet",
            loa="country",
            name="test_model",
            type="fatalities",
            targets=["target1"],
            category="forecast",
        )

        assert not result.success
        assert result.code == "storage_bucket_not_found"
        mock_appwrite_manager.create_bucket.assert_not_called()
        # And it is not retried into a bucket that was never made.
        assert mock_appwrite_manager.upload_file_with_metadata.call_count == 1

    def test_missing_bucket_logs_the_remediation_command(
        self, prediction_store, mock_appwrite_manager, tmp_path, caplog
    ):
        """The operator must be told which coordinate is wrong and how to fix it."""
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test content")

        mock_appwrite_manager.upload_file_with_metadata.return_value = OperationResult(
            success=False,
            error="Bucket not found",
            code="storage_bucket_not_found",
        )

        with caplog.at_level(logging.ERROR):
            prediction_store.upload_predictions(
                file=test_file,
                filename="test.parquet",
                loa="country",
                name="test_model",
                type="fatalities",
                targets=["target1"],
                category="forecast",
            )

        assert "provisioning ensure-bucket" in caplog.text
        assert "APPWRITE_PROD_FORECASTS_BUCKET_ID" in caplog.text


class TestSearchFailureIsNotAbsence:
    """C-241, consumer half — Cluster J at the FAO delivery's first lookup.

    `search_files_by_metadata` can now return `SEARCH_INCOMPLETE`, because #341 made it
    refuse to certify a walk it could not complete. The chain below used to convert that
    into an empty list and then into `None`, so the delivery path would read "there is no
    such forecast" when the truth was "I could not tell". Swapping a false-stale answer
    for a false-absent one is not a fix.
    """

    def test_a_failed_search_raises_rather_than_reporting_no_predictions(
        self, prediction_store, mock_appwrite_manager
    ):
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = (
            OperationResult(
                success=False,
                error="Search incomplete: enumerated 25 of a reported 461 documents",
                code="SEARCH_INCOMPLETE",
            )
        )

        with pytest.raises(MetadataSearchIncomplete) as excinfo:
            prediction_store.get_predictions_by_metadata(filters={"loa": "pgm"})

        assert "SEARCH_INCOMPLETE" in str(excinfo.value)

    def test_get_latest_file_id_does_not_answer_none_over_a_failed_search(
        self, prediction_store, mock_appwrite_manager
    ):
        """`None` means "no such file". It must never mean "the lookup broke"."""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = (
            OperationResult(success=False, error="boom", code="SEARCH_INCOMPLETE")
        )

        with pytest.raises(MetadataSearchIncomplete):
            prediction_store.get_latest_file_id(filters={"loa": "pgm"})

    def test_a_genuinely_empty_match_still_returns_none(
        self, prediction_store, mock_appwrite_manager
    ):
        """The other side of the distinction: a real 'no' must stay cheap and quiet."""
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = (
            OperationResult(success=True, data={"documents": [], "total": 0})
        )

        assert prediction_store.get_latest_file_id(filters={"loa": "pgm"}) is None

    def test_list_all_predictions_unfiltered_also_refuses_to_swallow(
        self, prediction_store, mock_appwrite_manager
    ):
        mock_appwrite_manager.metadata_manager.search_files_by_metadata.return_value = (
            OperationResult(success=False, error="boom", code="SEARCH_INCOMPLETE")
        )

        with pytest.raises(MetadataSearchIncomplete):
            prediction_store.list_all_predictions_unfiltered()
