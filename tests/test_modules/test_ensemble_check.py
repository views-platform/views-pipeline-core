import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
from views_pipeline_core.modules.validation.ensemble.check import (
    validate_model_conditions,
    validate_ensemble_model_deployment_status,
    validate_partition_config,
    validate_ensemble_model
)


class TestValidateModelConditions:
    """Test suite for validate_model_conditions function."""

    @pytest.fixture
    def mock_log_data(self):
        """Create mock log data."""
        return {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20241106_100000",
            "Data Fetch Timestamp": "20241106_090000",
            "Deployment Status": "production"
        }

    @pytest.fixture
    def current_november_2024(self):
        """Mock current time as November 2024."""
        return datetime(2024, 11, 6, 10, 0, 0)

    @pytest.fixture
    def current_august_2024(self):
        """Mock current time as August 2024."""
        return datetime(2024, 8, 15, 10, 0, 0)

    @pytest.fixture
    def current_march_2024(self):
        """Mock current time as March 2024."""
        return datetime(2024, 3, 15, 10, 0, 0)

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_validate_model_conditions_success_after_july(
        self, mock_datetime, mock_read_log, mock_log_data, current_november_2024
    ):
        """Test successful validation when current month is after July."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime
        mock_read_log.return_value = mock_log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is True
        mock_read_log.assert_called_once()

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_validate_model_conditions_success_before_july(
        self, mock_datetime, mock_read_log, current_march_2024
    ):
        """Test successful validation when current month is before July."""
        mock_datetime.now.return_value = current_march_2024
        mock_datetime.strptime = datetime.strptime
        
        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20230801_120000",  # Trained after July 2023
            "Data Generation Timestamp": "20240315_100000",
            "Data Fetch Timestamp": "20240315_090000",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "calibration"
        )

        assert result is True

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_validate_model_conditions_old_model_after_july(
        self, mock_datetime, mock_read_log, current_august_2024
    ):
        """Test failure when model is too old (current month after July)."""
        mock_datetime.now.return_value = current_august_2024
        mock_datetime.strptime = datetime.strptime
        
        log_data = {
            "Single Model Name": "old_model",
            "Single Model Timestamp": "20230601_120000",  # Trained before July
            "Data Generation Timestamp": "20240815_100000",
            "Data Fetch Timestamp": "20240815_090000",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_validate_model_conditions_old_model_before_july(
        self, mock_datetime, mock_read_log, current_march_2024
    ):
        """Test failure when model is too old (current month before July)."""
        mock_datetime.now.return_value = current_march_2024
        mock_datetime.strptime = datetime.strptime
        
        log_data = {
            "Single Model Name": "old_model",
            "Single Model Timestamp": "20220601_120000",  # Too old
            "Data Generation Timestamp": "20240315_100000",
            "Data Fetch Timestamp": "20240315_090000",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_validate_model_conditions_old_data_generation(
        self, mock_datetime, mock_read_log, current_november_2024
    ):
        """Test failure when data was not generated in current month."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime
        
        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20241015_100000",  # October, not November
            "Data Fetch Timestamp": "20241106_090000",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_validate_model_conditions_old_data_fetch(
        self, mock_datetime, mock_read_log, current_november_2024
    ):
        """Test failure when data was not fetched in current month."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime
        
        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20241106_100000",
            "Data Fetch Timestamp": "20241015_090000",  # October, not November
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_validate_model_conditions_none_timestamps(
        self, mock_datetime, mock_read_log, current_november_2024
    ):
        """Test validation with None timestamps (should pass if model is valid)."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime
        
        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "None",
            "Data Fetch Timestamp": "None",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is True

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_model_conditions_log_file_error(self, mock_read_log):
        """Test handling of log file read errors."""
        mock_read_log.side_effect = Exception("File not found")

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is False


class TestValidateEnsembleModelDeploymentStatus:
    """Test suite for validate_ensemble_model_deployment_status function."""

    @pytest.fixture
    def mock_log_data(self):
        """Create mock log data."""
        return {
            "Single Model Name": "test_model",
            "Deployment Status": "production"
        }

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_deployment_status_success(self, mock_read_log, mock_log_data):
        """Test successful deployment status validation."""
        mock_read_log.return_value = mock_log_data

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "production"
        )

        assert result is True

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_deployment_status_deprecated_ensemble(self, mock_read_log, mock_log_data):
        """Test failure when ensemble is deprecated."""
        mock_read_log.return_value = mock_log_data

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "Deprecated"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_deployment_status_deprecated_model(self, mock_read_log):
        """Test failure when constituent model is deprecated."""
        log_data = {
            "Single Model Name": "deprecated_model",
            "Deployment Status": "Deprecated"
        }
        mock_read_log.return_value = log_data

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "production"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_deployment_status_mismatch(self, mock_read_log):
        """Test failure when production model is in non-production ensemble."""
        log_data = {
            "Single Model Name": "prod_model",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "shadow"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_deployment_status_shadow_in_shadow(self, mock_read_log):
        """Test success when shadow model is in shadow ensemble."""
        log_data = {
            "Single Model Name": "shadow_model",
            "Deployment Status": "shadow"
        }
        mock_read_log.return_value = log_data

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "shadow"
        )

        assert result is True

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_deployment_status_log_file_error(self, mock_read_log):
        """Test handling of log file read errors."""
        mock_read_log.side_effect = Exception("File not found")

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "production"
        )

        assert result is False


class TestValidatePartitionConfig:
    """Test suite for validate_partition_config function."""

    @pytest.fixture
    def mock_ensemble_manager(self):
        """Create mock ensemble manager."""
        manager = Mock()
        manager._partition_dict = {
            "calibration": {"train": [1, 2, 3], "test": [4, 5]},
            "forecasting": {"train": [1, 2, 3, 4, 5], "test": []},
        }
        return manager

    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        manager = Mock()
        manager._partition_dict = {
            "calibration": {"train": [1, 2, 3], "test": [4, 5]},
            "forecasting": {"train": [1, 2, 3, 4, 5], "test": []},
        }
        return manager

    def test_validate_partition_config_success(
        self, mock_ensemble_manager, mock_model_manager
    ):
        """Test successful partition config validation."""
        result = validate_partition_config(
            mock_ensemble_manager,
            mock_model_manager,
            "calibration"
        )

        assert result is True

    def test_validate_partition_config_mismatch(self, mock_ensemble_manager):
        """Test failure when partition configs don't match."""
        mismatched_model_manager = Mock()
        mismatched_model_manager._partition_dict = {
            "calibration": {"train": [1, 2], "test": [3, 4, 5]},
        }

        result = validate_partition_config(
            mock_ensemble_manager,
            mismatched_model_manager,
            "calibration"
        )

        assert result is False

    def test_validate_partition_config_different_run_type(
        self, mock_ensemble_manager, mock_model_manager
    ):
        """Test validation for different run types."""
        result = validate_partition_config(
            mock_ensemble_manager,
            mock_model_manager,
            "forecasting"
        )

        assert result is True


class TestValidateEnsembleModel:
    """Test suite for validate_ensemble_model function."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ensemble configuration."""
        return {
            "name": "test_ensemble",
            "models": ["model1", "model2"],
            "run_type": "forecasting",
            "deployment_status": "production"
        }

    @pytest.fixture
    def mock_log_data(self):
        """Create mock log data for models."""
        return {
            "Single Model Name": "model1",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20241106_100000",
            "Data Fetch Timestamp": "20241106_090000",
            "Deployment Status": "production"
        }

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.managers.model.ModelPathManager')
    @patch('views_pipeline_core.managers.ensemble.EnsembleManager')
    @patch('views_pipeline_core.managers.ensemble.EnsemblePathManager')
    def test_validate_ensemble_model_success(
        self,
        mock_ensemble_path_manager,
        mock_ensemble_manager_class,
        mock_model_path_manager_class,
        mock_model_manager_class,
        mock_validate_conditions,
        mock_validate_deployment,
        mock_validate_partition,
        mock_config
    ):
        """Test successful ensemble model validation."""
        # Setup mocks
        mock_validate_conditions.return_value = True
        mock_validate_deployment.return_value = True
        mock_validate_partition.return_value = True

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        # Call without expecting exit
        validate_ensemble_model(mock_config)

        # Verify all validations were called
        assert mock_validate_conditions.call_count == 2  # Once per model
        assert mock_validate_deployment.call_count == 2
        assert mock_validate_partition.call_count == 2

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.managers.model.ModelPathManager')
    @patch('views_pipeline_core.managers.ensemble.EnsembleManager')
    @patch('views_pipeline_core.managers.ensemble.EnsemblePathManager')
    def test_validate_ensemble_model_conditions_fail(
        self,
        mock_ensemble_path_manager,
        mock_ensemble_manager_class,
        mock_model_path_manager_class,
        mock_model_manager_class,
        mock_validate_conditions,
        mock_validate_deployment,
        mock_validate_partition,
        mock_config
    ):
        """Test ensemble validation exits when model conditions fail."""
        mock_validate_conditions.return_value = False
        mock_validate_deployment.return_value = True
        mock_validate_partition.return_value = True

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        with pytest.raises(SystemExit) as exc_info:
            validate_ensemble_model(mock_config)
        
        assert exc_info.value.code == 1

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.managers.model.ModelPathManager')
    @patch('views_pipeline_core.managers.ensemble.EnsembleManager')
    @patch('views_pipeline_core.managers.ensemble.EnsemblePathManager')
    def test_validate_ensemble_model_deployment_fail(
        self,
        mock_ensemble_path_manager,
        mock_ensemble_manager_class,
        mock_model_path_manager_class,
        mock_model_manager_class,
        mock_validate_conditions,
        mock_validate_deployment,
        mock_validate_partition,
        mock_config
    ):
        """Test ensemble validation exits when deployment status fails."""
        mock_validate_conditions.return_value = True
        mock_validate_deployment.return_value = False
        mock_validate_partition.return_value = True

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        with pytest.raises(SystemExit) as exc_info:
            validate_ensemble_model(mock_config)
        
        assert exc_info.value.code == 1

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.managers.model.ModelPathManager')
    @patch('views_pipeline_core.managers.ensemble.EnsembleManager')
    @patch('views_pipeline_core.managers.ensemble.EnsemblePathManager')
    def test_validate_ensemble_model_partition_fail(
        self,
        mock_ensemble_path_manager,
        mock_ensemble_manager_class,
        mock_model_path_manager_class,
        mock_model_manager_class,
        mock_validate_conditions,
        mock_validate_deployment,
        mock_validate_partition,
        mock_config
    ):
        """Test ensemble validation exits when partition config fails."""
        mock_validate_conditions.return_value = True
        mock_validate_deployment.return_value = True
        mock_validate_partition.return_value = False

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        with pytest.raises(SystemExit) as exc_info:
            validate_ensemble_model(mock_config)
        
        assert exc_info.value.code == 1