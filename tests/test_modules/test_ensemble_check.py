import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
from views_pipeline_core.modules.validation.ensemble.check import (
    validate_model_conditions,
    validate_ensemble_model_deployment_status,
    validate_partition_config,
    validate_ensemble_model,
    validate_ensemble_raw_data_alignment,
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
        mock_read_log.side_effect = FileNotFoundError("File not found")

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_calibration_skips_freshness_checks(
        self, mock_datetime, mock_read_log, current_november_2024
    ):
        """Calibration runs skip Conditions 2+3 (issue #150)."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime

        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20241015_100000",  # October — stale
            "Data Fetch Timestamp": "20241015_090000",       # October — stale
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
    def test_validation_skips_freshness_checks(
        self, mock_datetime, mock_read_log, current_november_2024
    ):
        """Validation runs skip Conditions 2+3 (issue #150)."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime

        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20240901_100000",  # September — stale
            "Data Fetch Timestamp": "20240901_090000",       # September — stale
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "validation"
        )
        assert result is True

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_saved_skips_freshness_checks_for_forecasting(
        self, mock_datetime, mock_read_log, current_november_2024
    ):
        """saved=True skips Conditions 2+3 even for forecasting (issue #150)."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime

        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20241015_100000",  # October — stale
            "Data Fetch Timestamp": "20241015_090000",       # October — stale
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting",
            saved=True
        )
        assert result is True

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_forecasting_not_saved_still_enforces_freshness(
        self, mock_datetime, mock_read_log, current_november_2024
    ):
        """Forecasting with saved=False still enforces Conditions 2+3."""
        mock_datetime.now.return_value = current_november_2024
        mock_datetime.strptime = datetime.strptime

        log_data = {
            "Single Model Name": "test_model",
            "Single Model Timestamp": "20241101_120000",
            "Data Generation Timestamp": "20241015_100000",  # October — stale
            "Data Fetch Timestamp": "20241106_090000",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting",
            saved=False
        )
        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_calibration_still_enforces_training_cycle(
        self, mock_datetime, mock_read_log, current_august_2024
    ):
        """Calibration skips freshness but still enforces Condition 1 (training cycle)."""
        mock_datetime.now.return_value = current_august_2024
        mock_datetime.strptime = datetime.strptime

        log_data = {
            "Single Model Name": "old_model",
            "Single Model Timestamp": "20230601_120000",  # Too old
            "Data Generation Timestamp": "20240815_100000",
            "Data Fetch Timestamp": "20240815_090000",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "calibration"
        )
        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    @patch('views_pipeline_core.modules.validation.ensemble.check.datetime')
    def test_saved_still_enforces_training_cycle(
        self, mock_datetime, mock_read_log, current_august_2024
    ):
        """saved=True skips freshness but still enforces Condition 1 (training cycle)."""
        mock_datetime.now.return_value = current_august_2024
        mock_datetime.strptime = datetime.strptime

        log_data = {
            "Single Model Name": "old_model",
            "Single Model Timestamp": "20230601_120000",  # Too old
            "Data Generation Timestamp": "20240815_100000",
            "Data Fetch Timestamp": "20240815_090000",
            "Deployment Status": "production"
        }
        mock_read_log.return_value = log_data

        result = validate_model_conditions(
            Path("/test/path/generated"),
            "forecasting",
            saved=True
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
            "deprecated"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_validate_deployment_status_deprecated_model(self, mock_read_log):
        """Test failure when constituent model is deprecated."""
        log_data = {
            "Single Model Name": "deprecated_model",
            "Deployment Status": "deprecated"
        }
        mock_read_log.return_value = log_data

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "production"
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_a_non_graduate_member_in_a_graduate_ensemble_is_rejected(self, mock_read_log):
        """R2, and the rule the deleted `== "production"` branch was reaching for (#400).

        The test this replaces asserted that a member with status ``"production"`` in a
        ``"shadow"`` ensemble was rejected. It passed — but only because both the branch
        and the test invented a value views-models has never written. It writes ``shadow``,
        ``deployed``, ``baseline``, ``deprecated``. So the rule was real, the enforcement
        was real, and neither had ever met real data. C-218's shape exactly: a test that
        can only fail when the code disagrees with our mock.

        Restated against maturities that exist.
        """
        mock_read_log.return_value = {
            "Single Model Name": "candidate_model",
            "Deployment Status": "candidate",
        }

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "graduate",
        )

        assert result is False

    @patch('views_pipeline_core.modules.validation.ensemble.check.read_log_file')
    def test_a_graduate_member_in_a_graduate_ensemble_is_accepted(self, mock_read_log):
        """The negative control for the rule above. A rule that always fires is not a rule."""
        mock_read_log.return_value = {
            "Single Model Name": "graduate_model",
            "Deployment Status": "graduate",
        }

        result = validate_ensemble_model_deployment_status(
            Path("/test/path/generated"),
            "forecasting",
            "graduate",
        )

        assert result is True

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
        mock_read_log.side_effect = FileNotFoundError("File not found")

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

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_output_scale_consistency')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.data.model_path.ModelPathManager')
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
        mock_validate_scale,
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

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_output_scale_consistency')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.data.model_path.ModelPathManager')
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
        mock_validate_scale,
        mock_config
    ):
        """Test ensemble validation raises when model conditions fail."""
        mock_validate_conditions.return_value = False
        mock_validate_deployment.return_value = True
        mock_validate_partition.return_value = True

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        with pytest.raises(ValueError, match="failed validation"):
            validate_ensemble_model(mock_config)

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_output_scale_consistency')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.data.model_path.ModelPathManager')
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
        mock_validate_scale,
        mock_config
    ):
        """Test ensemble validation raises when deployment status fails."""
        mock_validate_conditions.return_value = True
        mock_validate_deployment.return_value = False
        mock_validate_partition.return_value = True

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        with pytest.raises(ValueError, match="failed validation"):
            validate_ensemble_model(mock_config)

    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_output_scale_consistency')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.data.model_path.ModelPathManager')
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
        mock_validate_scale,
        mock_config
    ):
        """Test ensemble validation raises when partition config fails."""
        mock_validate_conditions.return_value = True
        mock_validate_deployment.return_value = True
        mock_validate_partition.return_value = False

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        with pytest.raises(ValueError, match="failed validation"):
            validate_ensemble_model(mock_config)


    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_output_scale_consistency')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_partition_config')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_ensemble_model_deployment_status')
    @patch('views_pipeline_core.modules.validation.ensemble.check.validate_model_conditions')
    @patch('views_pipeline_core.managers.model.ModelManager')
    @patch('views_pipeline_core.data.model_path.ModelPathManager')
    @patch('views_pipeline_core.managers.ensemble.EnsembleManager')
    @patch('views_pipeline_core.managers.ensemble.EnsemblePathManager')
    def test_validate_ensemble_model_threads_saved_flag(
        self,
        mock_ensemble_path_manager,
        mock_ensemble_manager_class,
        mock_model_path_manager_class,
        mock_model_manager_class,
        mock_validate_conditions,
        mock_validate_deployment,
        mock_validate_partition,
        mock_validate_scale,
        mock_config
    ):
        """saved=True is threaded to validate_model_conditions (issue #150)."""
        mock_validate_conditions.return_value = True
        mock_validate_deployment.return_value = True
        mock_validate_partition.return_value = True

        mock_model_path = Mock()
        mock_model_path.data_generated = Path("/test/model/generated")
        mock_model_path_manager_class.return_value = mock_model_path

        validate_ensemble_model(mock_config, saved=True)

        for call in mock_validate_conditions.call_args_list:
            assert call.kwargs.get("saved") is True


class TestValidateEnsembleRawDataAlignment:
    """C-03: Validate that all ensemble models share consistent raw data files."""

    def test_empty_model_list_passes(self):
        """Empty model list has nothing to compare — should pass."""
        assert validate_ensemble_raw_data_alignment([], "calibration") is True

    def test_single_model_passes(self):
        """Single model has nothing to compare against — should pass."""
        with patch(
            "views_pipeline_core.data.model_path.ModelPathManager"
        ) as MockMPM:
            mock_path = Mock()
            mock_path.data_raw = Path("/project/models/model_a/data/raw")
            MockMPM.return_value = mock_path

            assert validate_ensemble_raw_data_alignment(["model_a"], "calibration") is True

    def test_models_with_different_file_sizes_warn(self, tmp_path):
        """Models with different raw data file sizes should return False."""
        dir_a = tmp_path / "model_a"
        dir_a.mkdir()
        file_a = dir_a / "calibration_viewser_df.parquet"
        file_a.write_bytes(b"x" * 1000)

        dir_b = tmp_path / "model_b"
        dir_b.mkdir()
        file_b = dir_b / "calibration_viewser_df.parquet"
        file_b.write_bytes(b"x" * 2000)

        with patch(
            "views_pipeline_core.data.model_path.ModelPathManager"
        ) as MockMPM:
            mock_a = Mock()
            mock_a._get_raw_data_file_paths.return_value = [file_a]
            mock_b = Mock()
            mock_b._get_raw_data_file_paths.return_value = [file_b]
            MockMPM.side_effect = [mock_a, mock_b]

            result = validate_ensemble_raw_data_alignment(
                ["model_a", "model_b"], "calibration"
            )

        assert result is False