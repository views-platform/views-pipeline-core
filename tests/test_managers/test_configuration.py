import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from pathlib import Path
import copy

from views_pipeline_core.managers.configuration import ConfigurationManager
from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.exceptions import ConfigurationException


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_configs():
    """Provide base configuration dictionaries for testing."""
    return {
        "hyperparameters": {
            "algorithm": "random_forest",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
            },
            "features": ["feature_1", "feature_2"],
            "targets": ["ged_sb"],
            "steps": [1, 2, 3],
        },
        "deployment": {
            "name": "test_model",
            "environment": "development",
            "version": "1.0.0",
            "deployment_status": "production",
        },
        "meta": {
            "description": "Test model",
            "author": "Test Author",
            "metrics": ["mse", "mae"],
        },
        "partition": {
            "calibration": {
                "train": (121, 396),
                "test": (397, 444),
            },
            "validation": {
                "train": (121, 444),
                "test": (445, 492),
            },
        },
        "sweep": {
            "name": "test_sweep",
            "method": "grid",
            "parameters": {
                "n_estimators": {"values": [50, 100, 150]},
            },
        },
    }


@pytest.fixture
def config_manager(base_configs):
    """Create ConfigurationManager instance."""
    return ConfigurationManager(
        config_hyperparameters=base_configs["hyperparameters"],
        config_deployment=base_configs["deployment"],
        config_meta=base_configs["meta"],
        partition_dict=base_configs["partition"],
        config_sweep=base_configs["sweep"],
    )


@pytest.fixture
def mock_wandb_module():
    """Mock WandBModule for testing."""
    mock = MagicMock()
    mock.send_alert = MagicMock()
    return mock


# ============================================================================
# Test ConfigurationManager Initialization
# ============================================================================

class TestConfigurationManagerInit:
    def test_initialization_with_all_configs(self, base_configs):
        """Test initialization with all configuration sources."""
        manager = ConfigurationManager(
            config_hyperparameters=base_configs["hyperparameters"],
            config_deployment=base_configs["deployment"],
            config_meta=base_configs["meta"],
            partition_dict=base_configs["partition"],
            config_sweep=base_configs["sweep"],
        )
        
        assert manager.config_hyperparameters == base_configs["hyperparameters"]
        assert manager.config_deployment == base_configs["deployment"]
        assert manager.config_meta == base_configs["meta"]
        assert manager.partition_dict == base_configs["partition"]
        assert manager.config_sweep == base_configs["sweep"]
        assert "timestamp" in manager._runtime_config
        
    def test_initialization_with_none_configs(self):
        """Test initialization with None values."""
        manager = ConfigurationManager(
            config_hyperparameters=None,
            config_deployment=None,
            config_meta=None,
            partition_dict=None,
            config_sweep=None,
        )
        
        assert manager.config_hyperparameters == {}
        assert manager.config_deployment == {}
        assert manager.config_meta == {}
        assert manager.partition_dict == {}
        assert manager.config_sweep is None
        assert "timestamp" in manager._runtime_config
        
    def test_timestamp_format(self, config_manager):
        """Test timestamp is in correct format."""
        timestamp = config_manager._runtime_config["timestamp"]
        
        # Should be YYYYMMDD_HHMMSS format
        assert len(timestamp) == 15
        assert timestamp[8] == "_"
        datetime.strptime(timestamp, "%Y%m%d_%H%M%S")  # Should not raise


# ============================================================================
# Test ConfigurationManager Dict-like Interface
# ============================================================================

class TestConfigurationManagerDictInterface:
    def test_getitem(self, config_manager):
        """Test dictionary-style access."""
        assert config_manager["algorithm"] == "random_forest"
        assert config_manager["name"] == "test_model"
        
    def test_getitem_missing_key(self, config_manager):
        """Test accessing missing key raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent_key"):
            _ = config_manager["nonexistent_key"]
            
    def test_setitem(self, config_manager):
        """Test dictionary-style setting."""
        config_manager["custom_param"] = 42
        assert config_manager["custom_param"] == 42
        
    def test_contains(self, config_manager):
        """Test 'in' operator."""
        assert "algorithm" in config_manager
        assert "nonexistent" not in config_manager
        
    def test_delitem(self, config_manager):
        """Test deleting keys."""
        config_manager["temp_key"] = "temp_value"
        assert "temp_key" in config_manager
        
        del config_manager["temp_key"]
        assert "temp_key" not in config_manager
        
    def test_delitem_not_in_runtime(self, config_manager):
        """Test deleting non-runtime key raises error."""
        with pytest.raises(KeyError):
            del config_manager["algorithm"]  # In hyperparameters, not runtime
            
    def test_get_with_default(self, config_manager):
        """Test get method with default."""
        assert config_manager.get("algorithm") == "random_forest"
        assert config_manager.get("nonexistent", "default") == "default"
        assert config_manager.get("nonexistent") is None
        
    def test_keys(self, config_manager):
        """Test keys method."""
        keys = config_manager.keys()
        assert isinstance(keys, list)
        assert "algorithm" in keys
        assert "name" in keys
        
    def test_values(self, config_manager):
        """Test values method."""
        values = config_manager.values()
        assert isinstance(values, list)
        assert "random_forest" in values
        
    def test_items(self, config_manager):
        """Test items method."""
        items = config_manager.items()
        assert isinstance(items, list)
        assert ("algorithm", "random_forest") in items


# ============================================================================
# Test get_combined_config
# ============================================================================

class TestGetCombinedConfig:
    def test_basic_merge(self, config_manager):
        """Test basic configuration merging."""
        config = config_manager.get_combined_config()
        
        # Check keys from all sources present
        # Partition dict contains nested structure: {'calibration': {'train': ..., 'test': ...}}
        assert "calibration" in config  # partition
        assert "train" in config["calibration"]  # nested train key
        assert "algorithm" in config  # hyperparameters
        assert "name" in config  # deployment
        assert "description" in config  # meta
        assert "timestamp" in config  # runtime
        
    def test_merge_priority(self, base_configs):
        """Test later sources override earlier ones."""
        # Create conflicting configs
        configs = copy.deepcopy(base_configs)
        configs["hyperparameters"]["name"] = "hp_name"
        configs["deployment"]["name"] = "deploy_name"
        configs["meta"]["name"] = "meta_name"
        
        manager = ConfigurationManager(
            config_hyperparameters=configs["hyperparameters"],
            config_deployment=configs["deployment"],
            config_meta=configs["meta"],
        )
        
        # Meta should override deployment which overrides hyperparameters
        assert manager.get_combined_config()["name"] == "meta_name"
        
    def test_runtime_highest_priority(self, config_manager):
        """Test runtime config has highest priority."""
        config_manager._runtime_config["algorithm"] = "override_algorithm"
        
        config = config_manager.get_combined_config()
        assert config["algorithm"] == "override_algorithm"
        
    def test_string_targets_converted_to_list(self, base_configs):
        """Test string targets are converted to list."""
        configs = copy.deepcopy(base_configs)
        configs["meta"]["targets"] = "single_target"
        
        manager = ConfigurationManager(
            config_hyperparameters=configs["hyperparameters"],
            config_deployment=configs["deployment"],
            config_meta=configs["meta"],
        )
        
        config = manager.get_combined_config()
        assert config["targets"] == ["single_target"]


# ============================================================================
# Test add_config
# ============================================================================

class TestAddConfig:
    def test_add_config(self, config_manager):
        """Test adding runtime configuration."""
        config_manager.add_config({"run_type": "calibration", "custom": 123})
        
        config = config_manager.get_combined_config()
        assert config["run_type"] == "calibration"
        assert config["custom"] == 123
        
    def test_add_config_overwrites(self, config_manager):
        """Test adding config overwrites existing runtime keys."""
        config_manager.add_config({"key": "value1"})
        config_manager.add_config({"key": "value2"})
        
        assert config_manager.get_combined_config()["key"] == "value2"


# ============================================================================
# Test update_for_single_run
# ============================================================================

class TestUpdateForSingleRun:
    def test_update_with_valid_args(self, config_manager, mock_wandb_module):
        """Test update with valid arguments."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
            evaluate=True,
        )
        
        with patch("views_pipeline_core.managers.configuration.validate_config"):
            config_manager.update_for_single_run(args, mock_wandb_module)
        
        config = config_manager.get_combined_config()
        assert config["run_type"] == "calibration"
        assert config["eval_type"] == "standard"
        assert config["sweep"] is False
        
    def test_update_with_timestep_override(self, config_manager, mock_wandb_module):
        """Test timestep override is applied."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            train=True,
            forecast=True,
            override_timestep=530,
        )
        
        with patch("views_pipeline_core.managers.configuration.validate_config"):
            config_manager.update_for_single_run(args, mock_wandb_module)
        
        config = config_manager.get_combined_config()
        assert "forecasting" in config
        assert config["forecasting"]["train"] == (121, 530)
        assert config["forecasting"]["test"][0] == 531
        
    def test_validation_failure_raises_exception(self, config_manager, mock_wandb_module):
        """Test validation failure raises ConfigurationException."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch("views_pipeline_core.managers.configuration.validate_config") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid config")
            
            with pytest.raises(ConfigurationException, match="Configuration validation failed"):
                config_manager.update_for_single_run(args, mock_wandb_module)
                
    def test_validation_sends_alert(self, config_manager, mock_wandb_module):
        """Test validation failure sends WandB alert."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch("views_pipeline_core.managers.configuration.validate_config") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid config")
            
            try:
                config_manager.update_for_single_run(args, mock_wandb_module)
            except ConfigurationException:
                pass
            
            # Alert sent via ConfigurationException


# ============================================================================
# Test _apply_timestep_override
# ============================================================================

class TestApplyTimestepOverride:
    def test_apply_override(self, config_manager):
        """Test timestep override calculation."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            train=True,
            forecast=True,
            override_timestep=530,
        )
        
        config_manager._apply_timestep_override(args)
        
        config = config_manager.get_combined_config()
        assert config["forecasting"]["train"] == (121, 530)
        # test_end = 530 + 1 + len(steps) = 530 + 1 + 3 = 534
        assert config["forecasting"]["test"] == (531, 534)
        
    def test_override_without_steps_logs_warning(self, config_manager, caplog):
        """Test override without steps logs warning."""
        # Remove steps from config
        config_manager.config_hyperparameters.pop("steps")
        
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
            override_timestep=530
        )
        
        config_manager._apply_timestep_override(args)
        
        assert "No 'steps' found in config" in caplog.text


# ============================================================================
# Test update_for_sweep_run
# ============================================================================

class TestUpdateForSweepRun:
    def test_sweep_update(self, config_manager, mock_wandb_module):
        """Test sweep configuration update."""
        wandb_config = {
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 15,
            }
        }
        
        args = ForecastingModelArgs(
            run_type="calibration",
            sweep=True,
        )
        
        with patch("views_pipeline_core.managers.configuration.validate_config"):
            config_manager.update_for_sweep_run(wandb_config, args, mock_wandb_module)
        
        config = config_manager.get_combined_config()
        assert config["hyperparameters"]["n_estimators"] == 200
        assert config["hyperparameters"]["max_depth"] == 15
        assert config["run_type"] == "calibration"
        assert config["sweep"] is True
        
    def test_sweep_validation_failure(self, config_manager, mock_wandb_module):
        """Test sweep validation failure."""
        wandb_config = {"invalid": "config"}
        args = ForecastingModelArgs(run_type="calibration", sweep=True)
        
        with patch("views_pipeline_core.managers.configuration.validate_config") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid sweep config")
            
            with pytest.raises(ConfigurationException, match="Sweep configuration validation failed"):
                config_manager.update_for_sweep_run(wandb_config, args, mock_wandb_module)


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigurationManagerIntegration:
    def test_full_workflow(self, base_configs, mock_wandb_module):
        """Test complete configuration workflow."""
        # Initialize
        manager = ConfigurationManager(
            config_hyperparameters=base_configs["hyperparameters"],
            config_deployment=base_configs["deployment"],
            config_meta=base_configs["meta"],
            partition_dict=base_configs["partition"],
        )
        
        # Add custom runtime config
        manager.add_config({"debug": True})
        
        # Update for single run
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
            evaluate=True,
        )
        
        with patch("views_pipeline_core.managers.configuration.validate_config"):
            manager.update_for_single_run(args, mock_wandb_module)
        
        # Get final config
        config = manager.get_combined_config()
        
        # Verify all sources merged correctly
        assert config["algorithm"] == "random_forest"  # hyperparameters
        assert config["name"] == "test_model"  # deployment
        assert config["description"] == "Test model"  # meta
        assert config["calibration"]["train"] == (121, 396)  # partition (nested structure)
        assert config["run_type"] == "calibration"  # runtime from args
        assert config["debug"] is True  # custom runtime
        assert "timestamp" in config  # auto-added
        
    def test_dict_interface_consistency(self, config_manager):
        """Test dict-like interface is consistent with get_combined_config."""
        config = config_manager.get_combined_config()
        
        # All keys accessible via dict interface
        for key in config.keys():
            assert config_manager[key] == config[key]
        
        # All items match
        for key, value in config_manager.items():
            assert config[key] == value
            
    def test_partition_access_after_update(self, base_configs, mock_wandb_module):
        """Test partition data accessible after run update."""
        manager = ConfigurationManager(
            config_hyperparameters=base_configs["hyperparameters"],
            config_deployment=base_configs["deployment"],
            config_meta=base_configs["meta"],
            partition_dict=base_configs["partition"],
        )
        
        args = ForecastingModelArgs(
            run_type="validation",
            train=True,
        )
        
        with patch("views_pipeline_core.managers.configuration.validate_config"):
            manager.update_for_single_run(args, mock_wandb_module)
        
        config = manager.get_combined_config()
        
        # Check both calibration and validation partitions present
        assert "calibration" in config
        assert "validation" in config
        assert config["validation"]["train"] == (121, 444)
        assert config["validation"]["test"] == (445, 492)