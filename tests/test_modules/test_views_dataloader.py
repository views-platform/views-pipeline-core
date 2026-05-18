import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np

from views_pipeline_core.modules.dataloaders import ViewsDataLoader, UpdateViewser
from views_pipeline_core.managers.model import ModelPathManager
from viewser import Queryset


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model_path():
    """Mock ModelPathManager"""
    mock = MagicMock(spec=ModelPathManager)
    mock.model_name = "test_model"
    mock.data_raw = Path("/tmp/test_model/data/raw")
    mock.data_processed = Path("/tmp/test_model/data/processed")
    mock.find_project_root.return_value = Path("/tmp/test_project")
    
    # Mock queryset
    mock_queryset = MagicMock(spec=Queryset)
    mock_queryset.model_dump.return_value = {
        "loa": "priogrid_month",
        "operations": [
            [
                {"namespace": "base", "name": "priogrid_month.ged_sb_best_sum_nokgi"},
                {"namespace": "trf", "name": "util.rename", "arguments": ["raw_ged_sb"]},
            ],
            [
                {"namespace": "base", "name": "priogrid_month.ged_sb_best_sum_nokgi"},
                {"namespace": "trf", "name": "ops.ln", "arguments": []},
                {"namespace": "trf", "name": "util.rename", "arguments": ["ln_ged_sb"]},
            ]
        ]
    }
    mock.get_queryset.return_value = mock_queryset
    
    return mock


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    month_ids = list(range(121, 445))  # 1990-2015
    pg_ids = [1000, 1001, 1002]
    
    index = pd.MultiIndex.from_product(
        [month_ids, pg_ids],
        names=["month_id", "priogrid_gid"]
    )
    
    data = {
        "raw_ged_sb": np.random.randn(len(index)),
        "ln_ged_sb": np.random.randn(len(index)),
        "feature_1": np.random.randn(len(index)),
        "feature_2": np.random.randn(len(index)),
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_queryset(sample_dataframe):
    """Create sample Queryset for testing"""
    mock_qs = MagicMock(spec=Queryset)
    mock_qs.model_dump.return_value = {
        "loa": "priogrid_month",
        "operations": [
            [
                {"namespace": "base", "name": "priogrid_month.ged_sb_best_sum_nokgi"},
                {"namespace": "trf", "name": "util.rename", "arguments": ["raw_ged_sb"]},
            ],
            [
                {"namespace": "base", "name": "priogrid_month.ged_sb_best_sum_nokgi"},
                {"namespace": "trf", "name": "ops.ln", "arguments": []},
                {"namespace": "trf", "name": "temporal.tlag", "arguments": ["1"]},
                {"namespace": "trf", "name": "util.rename", "arguments": ["ln_ged_sb_tlag_1"]},
            ]
        ]
    }
    
    # Mock publish chain - return the dataframe, not call it
    mock_publish = MagicMock()
    mock_publish.fetch_with_drift_detection.return_value = (sample_dataframe, [])
    mock_publish.fetch.return_value = sample_dataframe
    mock_qs.publish.return_value = mock_publish
    
    return mock_qs


@pytest.fixture
def data_loader(mock_model_path):
    """Create ViewsDataLoader instance"""
    return ViewsDataLoader(
        model_path=mock_model_path,
        steps=36
    )


# ============================================================================
# Test ViewsDataLoader Initialization
# ============================================================================

class TestViewsDataLoaderInit:
    def test_initialization_basic(self, mock_model_path):
        """Test basic initialization"""
        loader = ViewsDataLoader(mock_model_path, steps=36)
        
        assert loader._model_path == mock_model_path
        assert loader._model_name == "test_model"
        assert loader.steps == 36
        assert loader.partition is None
        assert loader.partition_dict is None
        
    def test_initialization_with_custom_partition(self, mock_model_path):
        """Test initialization with custom partition dict"""
        custom_part = {
            "calibration": {
                "train": (121, 396),
                "test": (397, 444)
            }
        }
        
        loader = ViewsDataLoader(
            mock_model_path,
            partition_dict=custom_part,
            steps=48
        )
        
        assert loader.partition_dict == custom_part
        assert loader.steps == 48
        
    def test_initialization_with_kwargs(self, mock_model_path):
        """Test initialization with additional kwargs"""
        loader = ViewsDataLoader(
            mock_model_path,
            steps=36,
            partition="calibration",
            override_month=530
        )
        
        assert loader.partition == "calibration"
        assert loader.override_month == 530


# ============================================================================
# Test _get_partition_dict
# ============================================================================

class TestGetPartitionDict:
    def test_calibration_partition(self, data_loader):
        """Test calibration partition dict"""
        data_loader.partition = "calibration"
        part_dict = data_loader._get_partition_dict(steps=36)
        
        assert part_dict["train"] == (121, 396)
        assert part_dict["test"] == (397, 444)
        
    def test_validation_partition(self, data_loader):
        """Test validation partition dict"""
        data_loader.partition = "validation"
        part_dict = data_loader._get_partition_dict(steps=36)
        
        assert part_dict["train"] == (121, 444)
        assert part_dict["test"] == (445, 492)
        
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ViewsMonth")
    def test_forecasting_partition(self, mock_views_month, data_loader):
        """Test forecasting partition dict"""
        # Mock current month
        mock_month = MagicMock()
        mock_month.id = 530
        mock_views_month.now.return_value = mock_month
        
        data_loader.partition = "forecasting"
        part_dict = data_loader._get_partition_dict(steps=36)
        
        assert part_dict["train"] == (121, 529)  # current - 1
        assert part_dict["test"] == (530, 566)  # current to current + 1 + steps (FIXED)
        
    def test_invalid_partition(self, data_loader):
        """Test invalid partition raises error"""
        data_loader.partition = "invalid"
        
        with pytest.raises(ValueError, match="partition should be either"):
            data_loader._get_partition_dict(steps=36)


# ============================================================================
# Test _get_month_range
# ============================================================================

class TestGetMonthRange:
    def test_calibration_range(self, data_loader):
        """Test month range for calibration"""
        data_loader.partition = "calibration"
        data_loader.partition_dict = {
            "train": (121, 396),
            "test": (397, 444)
        }
        
        first, last = data_loader._get_month_range()
        
        assert first == 121
        assert last == 444  # End of test range
        
    def test_validation_range(self, data_loader):
        """Test month range for validation"""
        data_loader.partition = "validation"
        data_loader.partition_dict = {
            "train": (121, 444),
            "test": (445, 492)
        }
        
        first, last = data_loader._get_month_range()
        
        assert first == 121
        assert last == 492
        
    def test_forecasting_range(self, data_loader):
        """Test month range for forecasting"""
        data_loader.partition = "forecasting"
        data_loader.partition_dict = {
            "train": (121, 529),
            "test": (530, 565)
        }
        
        first, last = data_loader._get_month_range()
        
        assert first == 121
        assert last == 529  # End of train range only
        
    def test_forecasting_with_override(self, data_loader):
        """Test month range with override"""
        data_loader.partition = "forecasting"
        data_loader.partition_dict = {
            "train": (121, 529),
            "test": (530, 565)
        }
        data_loader.override_month = 520
        
        first, last = data_loader._get_month_range()
        
        assert first == 121
        assert last == 520



# ============================================================================
# Test _fetch_data_from_viewser
# ============================================================================

class TestFetchDataFromViewser:
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64")
    def test_successful_fetch(self, mock_ensure_float, data_loader, sample_dataframe, sample_queryset):
        """Test successful data fetch from viewser"""
        # Setup mocks
        mock_ensure_float.return_value = sample_dataframe
        
        data_loader._model_path.get_queryset.return_value = sample_queryset
        data_loader.month_first = 121
        data_loader.month_last = 444
        
        # Execute
        df, alerts = data_loader._fetch_data_from_viewser(self_test=False)
        
        # Verify
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        mock_ensure_float.assert_called_once()
        
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64")
    def test_fetch_with_drift_detection_failure(self, mock_ensure_float, data_loader, sample_dataframe, sample_queryset):
        """Test fetch falls back when drift detection fails"""
        # Setup mocks
        mock_ensure_float.return_value = sample_dataframe
        
        data_loader._model_path.get_queryset.return_value = sample_queryset
        data_loader.month_first = 121
        data_loader.month_last = 444
        
        # Make drift detection fail
        mock_publish = sample_queryset.publish.return_value
        mock_publish.fetch_with_drift_detection.side_effect = KeyError("drift_key")
        mock_publish.fetch.return_value = sample_dataframe
        
        # Execute - should fallback to regular fetch
        df, alerts = data_loader._fetch_data_from_viewser(self_test=False)
        
        # Verify fallback was used
        assert df is not None
        mock_publish.fetch.assert_called_once()
        
    def test_fetch_queryset_not_found(self, data_loader):
        """Test error when queryset not found"""
        data_loader._model_path.get_queryset.return_value = None
        
        with pytest.raises(RuntimeError, match="Could not find queryset"):
            data_loader._fetch_data_from_viewser(self_test=False)


# ============================================================================
# Test UpdateViewser
# ============================================================================

class TestUpdateViewser:
    @pytest.fixture
    def update_viewser_setup(self, sample_queryset, sample_dataframe, tmp_path):
        """Setup for UpdateViewser tests"""
        # Create update dataframe
        update_index = pd.MultiIndex.from_product(
            [[528, 529, 530], [1000, 1001, 1002]],
            names=["month_id", "priogrid_id"]
        )
        update_df = pd.DataFrame({
            "ged_sb_best_sum_nokgi": np.random.randn(9),
        }, index=update_index)
        
        # Save update file
        update_file = tmp_path / "update.parquet"
        update_df.to_parquet(update_file)
        
        return {
            "queryset": sample_queryset,
            "viewser_df": sample_dataframe,
            "update_path": update_file,
            "months": [528, 529, 530]
        }
        
    def test_updater_initialization(self, update_viewser_setup):
        """Test UpdateViewser initialization"""
        setup = update_viewser_setup
        
        updater = UpdateViewser(
            queryset=setup["queryset"],
            viewser_df=setup["viewser_df"],
            data_path=setup["update_path"],
            months_to_update=setup["months"]
        )
        
        assert updater.queryset == setup["queryset"]
        assert len(updater.base_variables) > 0
        assert len(updater.var_names) > 0
        
    def test_extract_from_queryset(self, update_viewser_setup):
        """Test queryset parsing"""
        setup = update_viewser_setup
        
        updater = UpdateViewser(
            queryset=setup["queryset"],
            viewser_df=setup["viewser_df"],
            data_path=setup["update_path"],
            months_to_update=setup["months"]
        )
        
        base_vars, var_names, transforms = updater._extract_from_queryset()
        
        assert len(base_vars) == len(var_names)
        assert len(var_names) == len(transforms)
        assert any("raw_" in name for name in var_names)
        
    def test_preprocess_update_df(self, update_viewser_setup):
        """Test update DataFrame preprocessing"""
        setup = update_viewser_setup
        
        updater = UpdateViewser(
            queryset=setup["queryset"],
            viewser_df=setup["viewser_df"],
            data_path=setup["update_path"],
            months_to_update=setup["months"]
        )
        
        df_preprocessed = updater._preprocess_update_df()
        
        # Check only specified months
        assert df_preprocessed.index.get_level_values("month_id").unique().tolist() == setup["months"]
        # Check renamed columns
        assert any(col.startswith("raw_") for col in df_preprocessed.columns)
        
    def test_smart_cast(self, update_viewser_setup):
        """Test argument type conversion"""
        setup = update_viewser_setup
        
        updater = UpdateViewser(
            queryset=setup["queryset"],
            viewser_df=setup["viewser_df"],
            data_path=setup["update_path"],
            months_to_update=setup["months"]
        )
        
        assert updater._smart_cast("123") == 123
        assert updater._smart_cast("[1, 2, 3]") == [1, 2, 3]
        assert updater._smart_cast("{'key': 'val'}") == {'key': 'val'}
        assert updater._smart_cast("not_a_literal") == "not_a_literal"


# ============================================================================
# Integration Tests
# ============================================================================

class TestDataLoaderIntegration:
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64")
    def test_full_workflow_calibration(
        self,
        mock_ensure_float,
        mock_create_log,
        mock_save_df,
        mock_model_path,
        sample_dataframe,
        sample_queryset,
        tmp_path
    ):
        """Test complete workflow for calibration partition"""
        # Setup
        mock_model_path.data_raw = tmp_path
        mock_model_path.get_queryset.return_value = sample_queryset
        
        mock_ensure_float.return_value = sample_dataframe
        
        loader = ViewsDataLoader(mock_model_path, steps=36)
        
        # Execute
        df, alerts = loader.get_data(
            self_test=False,
            partition="calibration",
            use_saved=False,
            validate=False
        )
        
        # Verify
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert loader.partition == "calibration"
        assert loader.month_first == 121
        assert loader.month_last == 444