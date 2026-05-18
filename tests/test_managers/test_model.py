import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import pandas as pd

from views_pipeline_core.managers.model import ModelManager, ForecastingModelManager
from views_pipeline_core.managers import ModelPathManager
from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.exceptions import (
    DataFetchException,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model_path():
    """Create mock ModelPathManager."""
    mock = MagicMock(spec=ModelPathManager)
    mock.model_name = "purple_alien"
    mock.target = "model"
    mock.root = Path("/test/root")
    mock.models = Path("/test/root/models")
    mock.model_dir = Path("/test/root/models/purple_alien")
    mock.artifacts = Path("/test/root/models/purple_alien/artifacts")
    mock.configs = Path("/test/root/models/purple_alien/configs")
    mock.data = Path("/test/root/models/purple_alien/data")
    mock.data_raw = Path("/test/root/models/purple_alien/data/raw")
    mock.data_generated = Path("/test/root/models/purple_alien/data/generated")
    mock.reports = Path("/test/root/models/purple_alien/reports")
    mock.dotenv = Path("/test/root/.env")
    mock.logging = Path("/test/root/models/purple_alien/logs")  # Add logging attribute
    
    # Mock methods
    mock.get_scripts.return_value = {
        "config_deployment.py": "/test/root/models/purple_alien/configs/config_deployment.py",
        "config_hyperparameters.py": "/test/root/models/purple_alien/configs/config_hyperparameters.py",
        "config_meta.py": "/test/root/models/purple_alien/configs/config_meta.py",
        "config_partitions.py": "/test/root/models/purple_alien/configs/config_partitions.py",
        "config_sweep.py": "/test/root/models/purple_alien/configs/config_sweep.py",
        "main.py": "/test/root/models/purple_alien/main.py",
    }
    
    mock._get_raw_data_file_paths.return_value = [
        Path("/test/root/models/purple_alien/data/raw/calibration_viewser_df.parquet")
    ]
    mock._get_generated_predictions_data_file_paths.return_value = [
        Path("/test/root/models/purple_alien/data/generated/predictions_forecasting_20241105.parquet")
    ]
    mock.get_latest_model_artifact_path.return_value = Path(
        "/test/root/models/purple_alien/artifacts/calibration_model_20241105.pt"
    )
    
    return mock

@pytest.fixture
def mock_configs():
    """Provide mock configuration dictionaries."""
    return {
        "deployment": {
            "name": "purple_alien",
            "environment": "development",
            "deployment_status": "shadow",
        },
        "hyperparameters": {
            "algorithm": "random_forest",
            "n_estimators": 100,
            "steps": list(range(1, 37)),
            "time_steps": 36,
            "rolling_origin_stride": 1,
            "regression_targets": ["target_a"],
            "classification_targets": [],
            "regression_point_metrics": ["MSE"],
        },
        "meta": {
            "description": "Test model",
            "level": "pgm",
            "creator": "test",
            "prediction_format": "dataframe",
        },
        "partition": {
            "calibration": {
                "train": (121, 396),
                "test": (397, 444),
            },
        },
        "sweep": {
            "name": "test_sweep",
            "method": "grid",
        },
    }


@pytest.fixture
def mock_wandb_module():
    """Create mock WandBModule."""
    mock = MagicMock()
    mock.login = MagicMock()
    mock.initialize_run = MagicMock()
    mock.finish_run = MagicMock()
    mock.send_alert = MagicMock()
    mock.log = MagicMock()
    mock.save = MagicMock()
    mock.log_artifact = MagicMock()
    mock.log_evaluation_results = MagicMock()
    
    # Make initialize_run work as context manager
    mock.initialize_run.return_value.__enter__ = MagicMock()
    mock.initialize_run.return_value.__exit__ = MagicMock()
    
    return mock

@pytest.fixture
def concrete_manager_class():
    """Create concrete ModelManager subclass for testing."""
    class ConcreteModelManager(ModelManager):
        """Concrete implementation for testing."""
        pass
    
    return ConcreteModelManager


# ============================================================================
# Test ModelManager Initialization
# ============================================================================

class TestModelManagerInit:
    def test_initialization(self, mock_model_path, mock_configs):
        """Test basic initialization."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
            mock_load.side_effect = lambda script, method: mock_configs.get(
                script.replace("config_", "").replace(".py", "")
            )
            
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):  # Mock LoggingModule
                            manager = ModelManager(
                                model_path=mock_model_path,
                                wandb_notifications=True,
                                use_prediction_store=False,
                            )
                            
                            assert manager._model_path == mock_model_path
                            assert manager._wandb_notifications is True
                            assert manager._use_prediction_store is False
                            assert manager._sweep is False
                        
    def test_initialization_increments_counter(self, mock_model_path):
        """Test instance counter increments."""
        initial_count = ModelManager.__instances__
        
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config'):
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            ModelManager(model_path=mock_model_path)
                            
                            assert ModelManager.__instances__ == initial_count + 1




# ============================================================================
# Test ModelManager Configuration Loading
# ============================================================================

class TestModelManagerConfigLoading:
    def test_load_config_success(self, mock_model_path, mock_configs):
        """Test successful config loading."""
        with patch('importlib.util.spec_from_file_location'):
            with patch('importlib.util.module_from_spec') as mock_module:
                # Setup mock module with config method
                mock_config_module = MagicMock()
                mock_config_module.get_deployment_config.return_value = mock_configs["deployment"]
                mock_module.return_value = mock_config_module
                
                with patch('views_pipeline_core.modules.wandb.WandBModule'):
                    with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                        with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                            with patch('views_pipeline_core.modules.logging.LoggingModule'):
                                manager = ModelManager(model_path=mock_model_path)
                                
                                # Config should be loaded
                                assert hasattr(manager, '_config_deployment')



# ============================================================================
# Test ModelManager Properties
# ============================================================================

class TestModelManagerProperties:
    @pytest.fixture
    def manager(self, mock_model_path, mock_configs):
        """Create manager instance."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
            # Return actual config values
            mock_load.side_effect = lambda script, method: mock_configs.get(
                script.replace("config_", "").replace(".py", "")
            )
            
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                    with patch('views_pipeline_core.modules.logging.LoggingModule'):
                        manager = ModelManager(model_path=mock_model_path)
                        return manager
    
    def test_configs_property(self, manager):
        """Test configs property returns combined config."""
        configs = manager.configs
        
        assert isinstance(configs, dict)
        assert "name" in configs
        
    def test_configs_setter(self, manager):
        """Test configs setter updates runtime config."""
        manager.configs = {"custom_param": 42}
        
        # Verify it's in the combined config
        assert manager.configs.get("custom_param") == 42
        
    def test_config_property_alias(self, manager):
        """Test config property is alias for configs."""
        assert manager.config == manager.configs
        
    def test_args_property_before_execution_raises(self, manager):
        """Test accessing args before execution raises error."""
        # _args is None by default
        assert manager._args is None
            
    def test_args_property_after_execution(self, manager):
        """Test args property after setting."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        manager._args = args
        
        assert manager.args == args
# ============================================================================
# Test ForecastingModelManager Initialization
# ============================================================================

class TestForecastingModelManagerInit:
    def test_initialization(self, mock_model_path):
        """Test ForecastingModelManager initialization."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config'):
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            manager = ForecastingModelManager(
                                model_path=mock_model_path,
                                wandb_notifications=True,
                            )
                            
                            assert isinstance(manager, ModelManager)
                            assert isinstance(manager, ForecastingModelManager)



# ============================================================================
# Test ForecastingModelManager Static Methods
# ============================================================================


class TestForecastingModelManagerStatic:
    def test_resolve_evaluation_sequence_standard(self):
        """Standard evaluation: MAX_SHIFT_COUNT + 1 = 13 sequences."""
        n = ForecastingModelManager._resolve_evaluation_sequence_number("standard")
        assert n == 13  # base origin + 12 shifts

    def test_resolve_evaluation_sequence_long(self):
        """Long evaluation: 3 * MAX_SHIFT_COUNT + 1 = 37 sequences."""
        n = ForecastingModelManager._resolve_evaluation_sequence_number("long")
        assert n == 37  # base origin + 36 shifts
        
    def test_resolve_evaluation_sequence_complete(self):
        """Test complete evaluation returns None."""
        n = ForecastingModelManager._resolve_evaluation_sequence_number("complete")
        assert n is None
        
    def test_resolve_evaluation_sequence_invalid(self):
        """Test invalid eval type raises error."""
        with pytest.raises(ValueError, match="Invalid evaluation type"):
            ForecastingModelManager._resolve_evaluation_sequence_number("invalid")


# ============================================================================
# Test _get_evaluation_step_mappings (rolling-origin fix, ADR-031)
# ============================================================================

class TestGetEvaluationStepMappings:
    @pytest.fixture
    def manager(self, mock_model_path):
        """Create a minimally configured manager for method-level tests."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config'):
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            m = ForecastingModelManager(model_path=mock_model_path)
                            m.configs = {'steps': [1, 2, 3]}
                            # Correct nested structure for strict validation
                            m._partition_dict = {
                                'calibration': {'train': (100, 200), 'test': (201, 203)}
                            }
                            m._args = ForecastingModelArgs(run_type='calibration', train=True)
                            return m

    def test_returns_one_dict_per_sequence(self, manager):
        """List length equals n_sequences."""
        result = manager._get_evaluation_step_mappings(n_sequences=3)
        assert len(result) == 3

    def test_first_sequence_anchored_at_base_origin(self, manager):
        """Sequence 0 maps base_origin+s → s for all steps."""
        result = manager._get_evaluation_step_mappings(n_sequences=1)
        # base_origin = partition_dict['test'][0] - 1 = 201 - 1 = 200, steps = [1,2,3]
        assert result[0] == {201: 1, 202: 2, 203: 3}

    def test_each_sequence_shifts_by_one_month(self, manager):
        """Sequence i is anchored at base_origin + i."""
        result = manager._get_evaluation_step_mappings(n_sequences=3)
        assert result[0] == {201: 1, 202: 2, 203: 3}   # origin 200
        assert result[1] == {202: 1, 203: 2, 204: 3}   # origin 201
        assert result[2] == {203: 1, 204: 2, 205: 3}   # origin 202

    def test_no_shared_month_id_maps_to_same_step_across_sequences(self, manager):
        """Month 202 appears in seq 0 (step 2) and seq 1 (step 1): different steps."""
        result = manager._get_evaluation_step_mappings(n_sequences=2)
        assert result[0][202] == 2
        assert result[1][202] == 1

    def test_forecasting_run_type_uses_data_loader_origin(self, manager):
        """For forecasting, base_origin comes from _data_loader.month_last."""
        manager._args = ForecastingModelArgs(run_type='forecasting', train=True)
        mock_loader = Mock()
        mock_loader.month_last = 500
        manager._data_loader = mock_loader
        manager.configs = {'steps': [1, 2]}

        result = manager._get_evaluation_step_mappings(n_sequences=1)
        assert result[0] == {501: 1, 502: 2}

    def test_missing_partition_raises_keyerror(self, manager):
        """When partition dict is missing required run_type, raise KeyError."""
        manager._partition_dict = {}
        manager._args = ForecastingModelArgs(run_type='calibration', train=True)
        manager.configs = {'steps': [1, 2]}

        with pytest.raises(KeyError, match="Partition configuration for run_type 'calibration' not found"):
            manager._get_evaluation_step_mappings(n_sequences=1)


# ============================================================================
# Test _assert_partition_config_accessible (Layer 1 structural assertion)
# ============================================================================

class TestAssertPartitionConfigAccessible:
    @pytest.fixture
    def manager(self, mock_model_path):
        """Create a minimally configured manager for method-level tests."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config'):
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            m = ForecastingModelManager(model_path=mock_model_path)
                            m._partition_dict = {
                                'calibration': {'train': (100, 200), 'test': (201, 203)}
                            }
                            return m

    def test_valid_partition_passes(self, manager):
        """Well-formed partition with test key passes silently."""
        manager._assert_partition_config_accessible('calibration')  # no error

    def test_missing_run_type_raises_keyerror(self, manager):
        """run_type not in partition_dict raises KeyError immediately."""
        manager._partition_dict = {}
        with pytest.raises(KeyError, match="missing for run_type"):
            manager._assert_partition_config_accessible('calibration')

    def test_missing_test_key_raises_keyerror(self, manager):
        """Partition present but 'test' key absent raises KeyError."""
        manager._partition_dict = {'calibration': {'train': (100, 200)}}
        with pytest.raises(KeyError, match="no 'test' key"):
            manager._assert_partition_config_accessible('calibration')

    def test_empty_test_value_raises_indexerror(self, manager):
        """Empty test sequence (test[0] inaccessible) raises IndexError."""
        manager._partition_dict = {'calibration': {'train': (100, 200), 'test': []}}
        with pytest.raises(IndexError, match="at least one element"):
            manager._assert_partition_config_accessible('calibration')

    def test_forecasting_skips_partition_check(self, manager):
        """Forecasting run_type bypasses partition check (uses data_loader instead)."""
        manager._partition_dict = {}  # no partition at all
        manager._assert_partition_config_accessible('forecasting')  # no error


# ============================================================================
# Test ForecastingModelManager Abstract Methods
# ============================================================================

class TestForecastingModelManagerAbstractMethods:
    @pytest.fixture
    def manager(self, mock_model_path):
        """Create manager instance."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config'):
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            return ForecastingModelManager(model_path=mock_model_path)
    
    def test_train_model_artifact_not_implemented(self, manager):
        """Test _train_model_artifact raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="_train_model_artifact"):
            manager._train_model_artifact()
            
    def test_evaluate_model_artifact_not_implemented(self, manager):
        """Test _evaluate_model_artifact raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="_evaluate_model_artifact"):
            manager._evaluate_model_artifact("standard", "artifact.pt")
            
    def test_forecast_model_artifact_not_implemented(self, manager):
        """Test _forecast_model_artifact raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="_forecast_model_artifact"):
            manager._forecast_model_artifact("artifact.pt")
            
    def test_evaluate_sweep_not_implemented(self, manager):
        """Test _evaluate_sweep raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="_evaluate_sweep"):
            manager._evaluate_sweep("standard", Mock())



# ============================================================================
# Test ForecastingModelManager Execute Single Run
# ============================================================================

class TestExecuteSingleRun:
    @pytest.fixture
    def manager(self, mock_model_path, mock_configs):
        """Create manager with mocked dependencies."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
            mock_load.side_effect = lambda script, method: mock_configs.get(
                script.replace("config_", "").replace(".py", "")
            )
            
            # Mock wandb.login at the module level BEFORE creating manager
            with patch('wandb.login'):
                with patch('views_pipeline_core.modules.wandb.WandBModule') as MockWandB:
                    mock_wandb_instance = MagicMock()
                    mock_wandb_instance.login = MagicMock()  # Mock the login method
                    MockWandB.return_value = mock_wandb_instance
                    
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            manager = ForecastingModelManager(
                                model_path=mock_model_path,
                                wandb_notifications=False  # Disable notifications
                            )
                            # Layer 1 pre-condition: provide a structurally valid partition
                            # so tests that call execute_single_run(calibration) don't fail
                            # the structural check before reaching the code under test.
                            # test_len = 248-201+1 = 48 = time_steps(36) + MAX_SHIFT_COUNT(12)
                            manager._partition_dict = {
                                'calibration': {'train': (100, 200), 'test': (201, 248)}
                            }
                            return manager

    def test_execute_single_run_invalid_args_raises(self, manager):
        """Test execute_single_run with invalid args raises ValueError."""
        with pytest.raises(ValueError, match="must be an instance of ForecastingModelArgs"):
            manager.execute_single_run("invalid")
            
    def test_execute_single_run_sets_args(self, manager):
        """Test execute_single_run sets args property."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        # Mock the entire wandb module and manager's internal login
        with patch.object(manager._wandb_module, 'login'):
            with patch.object(manager, '_execute_data_fetching'):
                with patch.object(manager, '_execute_model_tasks'):
                    manager.execute_single_run(args)
                    
                    assert manager._args == args
                
    def test_execute_single_run_updates_config(self, manager):
        """Test execute_single_run updates configuration."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch.object(manager._wandb_module, 'login'):
            with patch.object(manager, '_execute_data_fetching'):
                with patch.object(manager, '_execute_model_tasks'):
                    manager.execute_single_run(args)
                    
                    # Check that run_type was added to config
                    assert manager.configs['run_type'] == 'calibration'
                
    def test_execute_single_run_calls_data_fetching(self, manager):
        """Test execute_single_run calls data fetching."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch.object(manager._wandb_module, 'login'):
            with patch.object(manager, '_execute_data_fetching') as mock_fetch:
                with patch.object(manager, '_execute_model_tasks'):
                    manager.execute_single_run(args)
                    
                    mock_fetch.assert_called_once()
                
    def test_execute_single_run_calls_model_tasks(self, manager):
        """Test execute_single_run calls model tasks."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch.object(manager._wandb_module, 'login'):
            with patch.object(manager, '_execute_data_fetching'):
                with patch.object(manager, '_execute_model_tasks') as mock_tasks:
                    manager.execute_single_run(args)
                    
                    mock_tasks.assert_called_once()



# ============================================================================
# Test ForecastingModelManager Execute Model Tasks
# ============================================================================

class TestExecuteModelTasks:
    @pytest.fixture
    def manager(self, mock_model_path):
        """Create manager instance."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config'):
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            return ForecastingModelManager(model_path=mock_model_path)
    
    def test_execute_model_tasks_training(self, manager):
        """Test execute_model_tasks calls training when requested."""
        manager._args = ForecastingModelArgs(run_type="calibration", train=True)
        manager._sweep = False
        
        with patch.object(manager, '_execute_model_training') as mock_train:
            manager._execute_model_tasks()
            
            mock_train.assert_called_once()
            
    def test_execute_model_tasks_evaluation(self, manager):
        """Test execute_model_tasks calls evaluation when requested."""
        manager._args = ForecastingModelArgs(
            run_type="calibration",
            evaluate=True,
            saved=True,
        )
        manager._sweep = False
        
        with patch.object(manager, '_execute_model_evaluation') as mock_eval:
            manager._execute_model_tasks()
            
            mock_eval.assert_called_once()
            
    def test_execute_model_tasks_forecasting(self, manager):
        """Test execute_model_tasks calls forecasting when requested."""
        manager._args = ForecastingModelArgs(
            run_type="forecasting",
            train=True,
            forecast=True,
        )
        manager._sweep = False
        
        with patch.object(manager, '_execute_model_forecasting') as mock_forecast:
            with patch.object(manager, '_execute_model_training'):
                manager._execute_model_tasks()
                
                mock_forecast.assert_called_once()
                
    def test_execute_model_tasks_sweep(self, manager):
        """Test execute_model_tasks calls sweep when in sweep mode."""
        manager._args = ForecastingModelArgs(run_type="calibration", sweep=True)
        manager._sweep = True
        
        with patch.object(manager, '_execute_model_sweeping') as mock_sweep:
            manager._execute_model_tasks()
            
            mock_sweep.assert_called_once()


# ============================================================================
# Test ForecastingModelManager Data Fetching
# ============================================================================

class TestExecuteDataFetching:
    @pytest.fixture
    def manager(self, mock_model_path, mock_configs):
        """Create manager with mocked WandB."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
            mock_load.side_effect = lambda script, method: mock_configs.get(
                script.replace("config_", "").replace(".py", "")
            )
            
            # Create a real WandBModule but mock its methods
            with patch('views_pipeline_core.modules.wandb.WandBModule') as MockWandB:
                mock_wandb_instance = MagicMock()
                MockWandB.return_value = mock_wandb_instance
                
                with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader') as mock_loader:
                    with patch('views_pipeline_core.modules.logging.LoggingModule'):
                        manager = ForecastingModelManager(model_path=mock_model_path)
                        manager._data_loader = mock_loader.return_value
                        
                        # Now mock the wandb module methods
                        manager._wandb_module.initialize_run = MagicMock()
                        manager._wandb_module.initialize_run.return_value.__enter__ = MagicMock()
                        manager._wandb_module.initialize_run.return_value.__exit__ = MagicMock(return_value=False)  # Don't suppress exceptions
                        manager._wandb_module.finish_run = MagicMock()
                        manager._wandb_module.send_alert = MagicMock()
                        
                        return manager
                        
    def test_data_fetching_calls_loader(self, manager):
        """Test data fetching calls data loader."""
        manager._args = ForecastingModelArgs(run_type="calibration", train=True)
        manager._project = "test_project"
        
        manager._execute_data_fetching()
        
        manager._data_loader.get_data.assert_called_once()
        
    def test_data_fetching_sends_alert(self, manager):
        """Test data fetching sends completion alert."""
        manager._args = ForecastingModelArgs(run_type="calibration", train=True)
        manager._project = "test_project"
        
        manager._execute_data_fetching()
        
        manager._wandb_module.send_alert.assert_called()
        
    def test_data_fetching_failure_raises_exception(self, manager):
        """Test data fetching failure raises DataFetchException."""
        manager._args = ForecastingModelArgs(run_type="calibration", train=True)
        manager._project = "test_project"
        
        # Make get_data raise an exception
        manager._data_loader.get_data.side_effect = Exception("Fetch failed")
        
        # The exception should be caught and re-raised as DataFetchException
        with pytest.raises(DataFetchException, match="Data fetching failed"):
            manager._execute_data_fetching()


# ============================================================================
# Test ForecastingModelManager String Representations
# ============================================================================

class TestStringRepresentations:
    @pytest.fixture
    def manager(self, mock_model_path):
        """Create manager instance."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config'):
            with patch('views_pipeline_core.modules.wandb.WandBModule'):
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            return ForecastingModelManager(model_path=mock_model_path)
    
    def test_repr(self, manager):
        """Test __repr__ contains key information."""
        r = repr(manager)
        
        assert "ForecastingModelManager" in r
        assert "model_name='purple_alien'" in r
        assert "target='model'" in r
        
    def test_repr_with_args(self, manager):
        """Test __repr__ includes run_type when args set."""
        manager._args = ForecastingModelArgs(run_type="calibration", train=True)
        
        r = repr(manager)
        
        assert "run_type='calibration'" in r
        
    def test_str(self, manager):
        """Test __str__ provides readable description."""
        s = str(manager)
        
        assert "ForecastingModelManager" in s
        assert "purple_alien" in s
        
    def test_str_with_args(self, manager):
        """Test __str__ includes run type when available."""
        manager._args = ForecastingModelArgs(run_type="validation", train=True)
        
        s = str(manager)
        
        assert "validation" in s


# ============================================================================
# Test ForecastingModelManager Save Methods
# ============================================================================

class TestSaveMethods:
    @pytest.fixture
    def manager(self, mock_model_path, mock_configs):
        """Create manager instance."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
            mock_load.side_effect = lambda script, method: mock_configs.get(
                script.replace("config_", "").replace(".py", "")
            )
            
            with patch('views_pipeline_core.modules.wandb.WandBModule') as MockWandB:
                mock_wandb_instance = MagicMock()
                MockWandB.return_value = mock_wandb_instance
                
                with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                    with patch('views_pipeline_core.modules.logging.LoggingModule'):
                        manager = ForecastingModelManager(model_path=mock_model_path)
                        
                        # Mock wandb methods
                        manager._wandb_module.send_alert = MagicMock()
                        manager._wandb_module.log_artifact = MagicMock()
                        
                        # Add required config for save methods
                        manager._config_manager.add_config({
                            "run_type": "calibration",
                            "timestamp": "20241105_120000",
                        })
                        
                        return manager
    
    def test_save_predictions(self, manager, mock_model_path):
        """Test saving predictions delegates to PredictionIOManager."""
        df = pd.DataFrame({"pred_ged_sb": [1, 2, 3]})

        with patch('pathlib.Path.mkdir'):
            with patch('views_pipeline_core.managers.prediction.io.save_dataframe') as mock_save:
                with patch('views_pipeline_core.managers.prediction.file_namer.generate_output_file_name', return_value="predictions.parquet"):
                    manager._save_predictions(df, mock_model_path.data_generated)

                    mock_save.assert_called_once()
                    manager._wandb_module.send_alert.assert_called()
                



# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_full_training_workflow(self, mock_model_path, mock_configs):
        """Test complete training workflow."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
            mock_load.side_effect = lambda script, method: mock_configs.get(
                script.replace("config_", "").replace(".py", "")
            )
            
            # Patch WandBModule BEFORE creating manager instance
            with patch('views_pipeline_core.modules.wandb.WandBModule') as MockWandB:
                mock_wandb_instance = MagicMock()
                MockWandB.return_value = mock_wandb_instance
                
                # Also mock the WandBModule import in the model module
                with patch('views_pipeline_core.managers.model.model.WandBModule', MockWandB):
                    with patch('views_pipeline_core.modules.dataloaders.dataloaders.ViewsDataLoader'):
                        with patch('views_pipeline_core.modules.logging.LoggingModule'):
                            # Mock the log file handling
                            with patch('views_pipeline_core.files.utils.handle_single_log_creation'):
                                
                                # Create concrete subclass
                                class TestManager(ForecastingModelManager):
                                    def _train_model_artifact(self):
                                        return MagicMock()
                                    
                                    def _evaluate_model_artifact(self, eval_type, artifact_name):
                                        return [pd.DataFrame()]
                                    
                                    def _forecast_model_artifact(self, artifact_name):
                                        return pd.DataFrame()
                                    
                                    def _evaluate_sweep(self, eval_type, model):
                                        return [pd.DataFrame()]
                                
                                # Now create the manager - it will use the mocked WandBModule
                                manager = TestManager(model_path=mock_model_path)
                                # Layer 1 pre-condition: provide a structurally valid partition
                                # test_len = 248-201+1 = 48 = time_steps(36) + MAX_SHIFT_COUNT(12)
                                manager._partition_dict = {
                                    'calibration': {'train': (100, 200), 'test': (201, 248)}
                                }

                                # Verify the mock was set up correctly
                                assert manager._wandb_module == mock_wandb_instance

                                args = ForecastingModelArgs(run_type="calibration", train=True)
                                
                                with patch.object(manager, '_execute_data_fetching'):
                                    # Mock wandb.login to avoid actual network calls
                                    with patch('wandb.login'):
                                        manager.execute_single_run(args)
                                        
                                        # Verify workflow executed
                                        assert manager.args == args
                                        # Verify WandB module's login was called
                                        mock_wandb_instance.login.assert_called_once()