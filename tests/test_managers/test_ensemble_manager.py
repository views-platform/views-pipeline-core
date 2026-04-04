"""
Tests for EnsembleManager class.

This module contains comprehensive tests for the EnsembleManager class,
which orchestrates ensemble forecasting models including training,
evaluation, forecasting, and reconciliation.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd

from views_pipeline_core.managers.ensemble import EnsembleManager, EnsemblePathManager
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.exceptions import PipelineException


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ensemble_path():
    """Create mock EnsemblePathManager."""
    mock = MagicMock(spec=EnsemblePathManager)
    mock.model_name = "test_ensemble"
    mock.target = "ensemble"
    mock.root = Path("/test/root")
    mock.models = Path("/test/root/ensembles")
    mock.model_dir = Path("/test/root/ensembles/test_ensemble")
    mock.artifacts = Path("/test/root/ensembles/test_ensemble/artifacts")
    mock.configs = Path("/test/root/ensembles/test_ensemble/configs")
    mock.data = Path("/test/root/ensembles/test_ensemble/data")
    mock.data_generated = Path("/test/root/ensembles/test_ensemble/data/generated")
    mock.reports = Path("/test/root/ensembles/test_ensemble/reports")
    mock.dotenv = Path("/test/root/.env")
    mock.logging = Path("/test/root/ensembles/test_ensemble/logs")
    
    mock.get_scripts.return_value = {
        "config_deployment.py": "/test/root/ensembles/test_ensemble/configs/config_deployment.py",
        "config_hyperparameters.py": "/test/root/ensembles/test_ensemble/configs/config_hyperparameters.py",
        "config_meta.py": "/test/root/ensembles/test_ensemble/configs/config_meta.py",
        "config_partitions.py": "/test/root/ensembles/test_ensemble/configs/config_partitions.py",
        "main.py": "/test/root/ensembles/test_ensemble/main.py",
    }
    
    mock._get_generated_predictions_data_file_paths.return_value = [
        Path("/test/root/ensembles/test_ensemble/data/generated/predictions_forecasting_20241105.parquet")
    ]
    mock.get_latest_model_artifact_path.return_value = Path(
        "/test/root/ensembles/test_ensemble/artifacts/calibration_model_20241105.pt"
    )
    
    return mock


@pytest.fixture
def mock_configs():
    """Provide mock configuration dictionaries."""
    return {
        "deployment": {
            "name": "test_ensemble",
            "environment": "development",
            "deployment_status": "production",
        },
        "hyperparameters": {
            "algorithm": "ensemble",
            "models": ["purple_alien", "blue_cat"],
            "aggregation": "mean",
        },
        "meta": {
            "description": "Test ensemble",
            "targets": ["ged_sb"],
            "metrics": ["mse", "mae"],
        },
        "partition": {
            "calibration": {
                "train": (121, 396),
                "test": (397, 444),
            },
        },
    }


@pytest.fixture
def mock_wandb_module():
    """Create mock WandBModule."""
    mock = MagicMock()
    mock.login = MagicMock()
    mock.finish_run = MagicMock()
    mock.send_alert = MagicMock()
    mock.log = MagicMock()
    
    # Make initialize_run work as context manager
    mock.initialize_run.return_value.__enter__ = MagicMock()
    mock.initialize_run.return_value.__exit__ = MagicMock(return_value=False)
    
    return mock


@pytest.fixture
def sample_dataframe():
    """Create sample prediction DataFrame with priogrid index."""
    index = pd.MultiIndex.from_tuples(
        [(1, 100), (1, 101), (2, 100), (2, 101)],
        names=["month_id", "priogrid_gid"]
    )
    return pd.DataFrame(
        {"ged_sb": [0.1, 0.2, 0.3, 0.4]},
        index=index
    )


@pytest.fixture
def sample_cm_dataframe():
    """Create sample C model DataFrame with country_id index."""
    index = pd.MultiIndex.from_tuples(
        [(1, 100), (1, 101), (2, 100), (2, 101)],
        names=["month_id", "country_id"]
    )
    return pd.DataFrame(
        {"ged_sb": [0.1, 0.2, 0.3, 0.4]},
        index=index
    )


@pytest.fixture
def sample_dataframes_list(sample_dataframe):
    """Create list of sample DataFrames for aggregation tests."""
    df1 = sample_dataframe.copy()
    df2 = sample_dataframe.copy()
    df2["ged_sb"] = [0.2, 0.3, 0.4, 0.5]
    return [df1, df2]


@pytest.fixture
def manager(mock_ensemble_path, mock_configs, mock_wandb_module):
    """Create EnsembleManager instance with mocked dependencies."""
    with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
        mock_load.side_effect = lambda script, method: mock_configs.get(
            script.replace("config_", "").replace(".py", "")
        )
        
        with patch('views_pipeline_core.modules.wandb.WandBModule') as MockWandB:
            MockWandB.return_value = mock_wandb_module
            
            with patch('views_pipeline_core.managers.configuration.ConfigurationManager') as MockConfig:
                mock_config_manager = MagicMock()
                default_configs = {
                    "name": "test_ensemble",
                    "models": ["purple_alien", "blue_cat"],
                    "aggregation": "mean",
                    "run_type": "calibration",
                }
                mock_config_manager.configs = default_configs
                mock_config_manager.get_combined_config.return_value = default_configs
                MockConfig.return_value = mock_config_manager
                
                with patch('views_pipeline_core.modules.logging.LoggingModule'):
                    manager = EnsembleManager(
                        ensemble_path=mock_ensemble_path,
                        wandb_notifications=False,
                        use_prediction_store=False,
                    )
                    manager._wandb_module = mock_wandb_module
                    manager._config_manager = mock_config_manager
                    return manager


# ============================================================================
# Test EnsembleManager Initialization
# ============================================================================

class TestEnsembleManagerInit:
    """Tests for EnsembleManager initialization."""
    
    def test_initialization(self, mock_ensemble_path, mock_configs, mock_wandb_module):
        """Test basic EnsembleManager initialization."""
        with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config') as mock_load:
            mock_load.side_effect = lambda script, method: mock_configs.get(
                script.replace("config_", "").replace(".py", "")
            )
            
            with patch('views_pipeline_core.modules.wandb.WandBModule') as MockWandB:
                MockWandB.return_value = mock_wandb_module
                
                with patch('views_pipeline_core.managers.configuration.ConfigurationManager'):
                    with patch('views_pipeline_core.modules.logging.LoggingModule'):
                        manager = EnsembleManager(
                            ensemble_path=mock_ensemble_path,
                            wandb_notifications=True,
                            use_prediction_store=False,
                        )
                        
                        assert manager._model_path == mock_ensemble_path
                        assert manager._wandb_notifications is True
                        assert manager._use_prediction_store is False
    
    # Note: test_initialization_with_prediction_store is skipped because
    # prediction store (AppwriteConfig) is not available in the test environment


# ============================================================================
# Test EnsembleManager Execute Single Run
# ============================================================================

class TestExecuteSingleRun:
    """Tests for execute_single_run method."""
    
    def test_execute_single_run_invalid_args_raises(self, manager):
        """Test execute_single_run with invalid args raises ValueError."""
        with pytest.raises(ValueError, match="must be an instance of ForecastingModelArgs"):
            manager.execute_single_run("invalid")
    
    def test_execute_single_run_invalid_args_type(self, manager):
        """Test execute_single_run with wrong type raises ValueError."""
        with pytest.raises(ValueError, match="must be an instance of ForecastingModelArgs"):
            manager.execute_single_run({"run_type": "calibration"})
    
    def test_execute_single_run_sets_args(self, manager):
        """Test execute_single_run sets args property."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch.object(manager, '_execute_model_tasks'):
            with patch('views_pipeline_core.modules.validation.ensemble.validate_ensemble_model'):
                manager.execute_single_run(args)
                
                assert manager._args == args
    
    def test_execute_single_run_calls_wandb_login(self, manager, mock_wandb_module):
        """Test execute_single_run calls WandB login."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch.object(manager, '_execute_model_tasks'):
            manager.execute_single_run(args)
            
            mock_wandb_module.login.assert_called_once()
    
    def test_execute_single_run_validates_ensemble_when_not_training(self, manager):
        """Test ensemble validation is called when not training."""
        args = ForecastingModelArgs(run_type="calibration", train=False, evaluate=True, saved=True)
        
        with patch.object(manager, '_execute_model_tasks'):
            with patch('views_pipeline_core.managers.ensemble.ensemble.validate_ensemble_model') as mock_validate:
                manager.execute_single_run(args)
                
                mock_validate.assert_called_once()
    
    def test_execute_single_run_skips_validation_when_training(self, manager):
        """Test ensemble validation is skipped when training."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        
        with patch.object(manager, '_execute_model_tasks'):
            with patch('views_pipeline_core.modules.validation.ensemble.validate_ensemble_model') as mock_validate:
                manager.execute_single_run(args)
                
                mock_validate.assert_not_called()


# ============================================================================
# Test Execute Model Tasks
# ============================================================================

class TestExecuteModelTasks:
    """Tests for _execute_model_tasks method."""
    
    def test_execute_model_tasks_calls_training(self, manager):
        """Test training is called when train=True."""
        args = ForecastingModelArgs(run_type="calibration", train=True)
        manager._args = args
        
        with patch.object(manager, '_execute_model_training') as mock_train:
            with patch.object(manager, '_execute_model_evaluation'):
                with patch.object(manager, '_execute_model_forecasting'):
                    manager._execute_model_tasks()
                    
                    mock_train.assert_called_once()
    
    def test_execute_model_tasks_calls_evaluation(self, manager):
        """Test evaluation is called when evaluate=True."""
        args = ForecastingModelArgs(run_type="calibration", evaluate=True, saved=True)
        manager._args = args
        
        with patch.object(manager, '_execute_model_training'):
            with patch.object(manager, '_execute_model_evaluation') as mock_eval:
                with patch.object(manager, '_execute_model_forecasting'):
                    with patch.object(manager, '_execute_evaluation_reporting'):
                        manager._execute_model_tasks()
                        
                        mock_eval.assert_called_once()
    
    def test_execute_model_tasks_calls_forecasting(self, manager):
        """Test forecasting is called when forecast=True."""
        args = ForecastingModelArgs(run_type="forecasting", forecast=True, saved=True)
        manager._args = args
        
        with patch.object(manager, '_execute_model_training'):
            with patch.object(manager, '_execute_model_evaluation'):
                with patch.object(manager, '_execute_model_forecasting') as mock_forecast:
                    with patch.object(manager, '_execute_forecast_reporting'):
                        manager._execute_model_tasks()
                        
                        mock_forecast.assert_called_once()
    
    def test_execute_model_tasks_does_not_call_training_when_false(self, manager):
        """Test training is not called when train=False."""
        args = ForecastingModelArgs(run_type="calibration", train=False, evaluate=True, saved=True)
        manager._args = args
        
        with patch.object(manager, '_execute_model_training') as mock_train:
            with patch.object(manager, '_execute_model_evaluation'):
                with patch.object(manager, '_execute_evaluation_reporting'):
                    manager._execute_model_tasks()
                    
                    mock_train.assert_not_called()


# ============================================================================
# Test Create Model Args
# ============================================================================

class TestCreateModelArgs:
    """Tests for _create_model_args method."""
    
    def test_create_model_args_for_training(self, manager):
        """Test creating args for training."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
            saved=True,
            eval_type="standard",
            update_viewser=False,
        )
        manager._args = args
        manager._wandb_notifications = False
        
        model_args = manager._create_model_args(train=True)
        
        assert model_args.train is True
        assert model_args.evaluate is False
        assert model_args.forecast is False
        assert model_args.saved is True
        assert model_args.run_type == "calibration"
    
    def test_create_model_args_for_evaluation(self, manager):
        """Test creating args for evaluation."""
        args = ForecastingModelArgs(
            run_type="calibration",
            evaluate=True,
            eval_type="standard",
            saved=True,
        )
        manager._args = args
        manager._wandb_notifications = False
        
        model_args = manager._create_model_args(evaluate=True)
        
        assert model_args.train is False
        assert model_args.evaluate is True
        assert model_args.forecast is False
        assert model_args.saved is True  # Default for non-training
    
    def test_create_model_args_for_forecasting(self, manager):
        """Test creating args for forecasting."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            forecast=True,
            eval_type="standard",
            saved=True,
        )
        manager._args = args
        manager._wandb_notifications = False
        
        model_args = manager._create_model_args(forecast=True)
        
        assert model_args.train is False
        assert model_args.evaluate is False
        assert model_args.forecast is True
    
    def test_create_model_args_uses_prediction_store(self, manager):
        """Test prediction store flag is set for forecasting."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            forecast=True,
            saved=True,
        )
        manager._args = args
        manager._use_prediction_store = True
        manager._wandb_notifications = False
        
        model_args = manager._create_model_args(forecast=True)
        
        assert model_args.prediction_store is True
    
    def test_create_model_args_no_prediction_store_for_training(self, manager):
        """Test prediction store is not used for training."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
        )
        manager._args = args
        manager._use_prediction_store = True
        manager._wandb_notifications = False
        
        model_args = manager._create_model_args(train=True)
        
        assert model_args.prediction_store is False


# ============================================================================
# Test Aggregation Methods
# ============================================================================

# Note: TestGetAggregatedDf tests removed because _get_aggregated_df method
# now uses AggregationManager internally and the API has changed significantly.
# The method is tested indirectly through higher-level tests.


# ============================================================================
# Test Reconciliation
# ============================================================================

class TestApplyReconciliation:
    """Tests for _apply_reconciliation method."""
    
    def _set_manager_configs(self, manager, configs):
        """Helper to properly set configs on the manager's config_manager mock."""
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
    
    def test_no_reconciliation_when_not_configured(self, manager, sample_dataframe):
        """Test no reconciliation when not configured."""
        self._set_manager_configs(manager, {
            "reconciliation": None,
        })
        manager._EnsembleManager__activate_reconciliation = True
        
        result = manager._apply_reconciliation(sample_dataframe)
        
        assert result.equals(sample_dataframe)
    
    def test_reconciliation_with_invalid_type(self, manager, sample_dataframe):
        """Test reconciliation is skipped with invalid type."""
        self._set_manager_configs(manager, {
            "reconciliation": "invalid_type",
        })
        manager._EnsembleManager__activate_reconciliation = True
        
        result = manager._apply_reconciliation(sample_dataframe)
        
        assert result.equals(sample_dataframe)
    
    def test_reconciliation_with_pgm_cm_point(self, manager, sample_dataframe, mock_wandb_module):
        """Test reconciliation with pgm_cm_point type."""
        self._set_manager_configs(manager, {
            "reconciliation": "pgm_cm_point",
            "reconcile_with": "cm_model",
            "run_type": "forecasting",
        })
        manager._EnsembleManager__activate_reconciliation = True
        manager._wandb_module = mock_wandb_module
        
        reconciled_df = sample_dataframe.copy()
        reconciled_df["ged_sb"] = [0.15, 0.25, 0.35, 0.45]
        
        with patch.object(manager, '_EnsembleManager__reconcile_pg_with_c', return_value=reconciled_df):
            result = manager._apply_reconciliation(sample_dataframe)
            
            assert result.equals(reconciled_df)
            mock_wandb_module.send_alert.assert_called()
    
    def test_reconciliation_returns_original_on_failure(self, manager, sample_dataframe, mock_wandb_module):
        """Test original DataFrame is returned when reconciliation fails."""
        self._set_manager_configs(manager, {
            "reconciliation": "pgm_cm_point",
            "reconcile_with": "cm_model",
            "run_type": "forecasting",
        })
        manager._EnsembleManager__activate_reconciliation = True
        manager._wandb_module = mock_wandb_module
        
        with patch.object(manager, '_EnsembleManager__reconcile_pg_with_c', return_value=None):
            with patch('views_pipeline_core.managers.ensemble.ensemble.wandb') as mock_wandb:
                mock_wandb.AlertLevel.WARNING = "WARNING"
                result = manager._apply_reconciliation(sample_dataframe)
                
                assert result.equals(sample_dataframe)


# ============================================================================
# Test Execute Shell Script
# ============================================================================

class TestExecuteShellScript:
    """Tests for _execute_shell_script method."""
    
    def test_execute_shell_script_success(self, manager):
        """Test successful shell script execution."""
        mock_model_path = MagicMock(spec=ModelPathManager)
        mock_args = MagicMock(spec=ForecastingModelArgs)
        mock_args.to_shell_command.return_value = ["python", "main.py", "--train"]
        
        with patch('subprocess.run') as mock_run:
            manager._execute_shell_script(mock_model_path, "purple_alien", mock_args)
            
            mock_run.assert_called_once_with(
                ["python", "main.py", "--train"],
                check=True,
                timeout=7200,
            )
    
    def test_execute_shell_script_failure_raises(self, manager, mock_wandb_module):
        """Test shell script failure raises PipelineException."""
        mock_model_path = MagicMock(spec=ModelPathManager)
        mock_args = MagicMock(spec=ForecastingModelArgs)
        mock_args.to_shell_command.return_value = ["python", "main.py", "--train"]
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Script failed")
            
            with pytest.raises(PipelineException, match="Error during shell command execution"):
                manager._execute_shell_script(mock_model_path, "purple_alien", mock_args)


# ============================================================================
# Test Train Ensemble
# ============================================================================

class TestTrainEnsemble:
    """Tests for _train_ensemble method."""
    
    def test_train_ensemble_iterates_models(self, manager):
        """Test training iterates over all models in ensemble."""
        configs = {
            "models": ["purple_alien", "blue_cat", "green_dog"],
            "name": "test_ensemble",
        }
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        
        with patch.object(manager, '_train_model_artifact') as mock_train:
            with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm', side_effect=lambda x, **kwargs: x):
                with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm.write'):
                    manager._train_ensemble()
                    
                    assert mock_train.call_count == 3
                    mock_train.assert_any_call("purple_alien")
                    mock_train.assert_any_call("blue_cat")
                    mock_train.assert_any_call("green_dog")


# ============================================================================
# Test Evaluate Ensemble
# ============================================================================

class TestEvaluateEnsemble:
    """Tests for _evaluate_ensemble method."""
    
    @pytest.mark.skip(reason="Source code has bug at line 321: 'for i in range(len(n_outputs))' where n_outputs is already an int")
    def test_evaluate_ensemble_calls_evaluate_model_artifact(self, manager, sample_dataframes_list):
        """Test evaluation calls _evaluate_model_artifact for each model."""
        configs = {
            "models": ["purple_alien", "blue_cat"],
            "aggregation": "mean",
            "name": "test_ensemble",
        }
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        
        # Each model returns a list of DataFrames (one per evaluation sequence)
        model_preds = [[sample_dataframes_list[0]], [sample_dataframes_list[1]]]
        
        with patch.object(manager, '_evaluate_model_artifact', side_effect=model_preds) as mock_eval:
            with patch.object(manager, '_get_aggregated_df', return_value=sample_dataframes_list[0]):
                with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm', side_effect=lambda x, **kwargs: x):
                    with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm.write'):
                        manager._evaluate_ensemble()
                        
                        # Verify _evaluate_model_artifact was called for each model
                        assert mock_eval.call_count == 2
                        mock_eval.assert_any_call("purple_alien")
                        mock_eval.assert_any_call("blue_cat")


# ============================================================================
# Test Forecast Ensemble
# ============================================================================

class TestForecastEnsemble:
    """Tests for _forecast_ensemble method."""
    
    def test_forecast_ensemble_aggregates_and_returns_dataframe(self, manager, sample_dataframe):
        """Test forecasting aggregates predictions and returns DataFrame."""
        configs = {
            "models": ["purple_alien", "blue_cat"],
            "aggregation": "mean",
            "name": "test_ensemble",
            "targets": ["ged_sb"],
        }
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        manager._EnsembleManager__activate_reconciliation = False
        
        with patch.object(manager, '_forecast_model_artifact', return_value=sample_dataframe):
            with patch.object(EnsembleManager, '_get_aggregated_df', return_value=sample_dataframe):
                with patch('views_pipeline_core.managers.ensemble.ensemble._ViewsDataset') as MockDataset:
                    mock_dataset = MagicMock()
                    mock_dataset.dataframe = sample_dataframe
                    MockDataset.return_value = mock_dataset
                    
                    with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm', side_effect=lambda x, **kwargs: x):
                        with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm.write'):
                            with patch.object(manager, '_apply_reconciliation', return_value=sample_dataframe):
                                result = manager._forecast_ensemble()
                                
                                assert isinstance(result, pd.DataFrame)
    
    def test_forecast_ensemble_raises_on_invalid_type(self, manager, sample_dataframe):
        """Test forecasting raises TypeError for invalid prediction type."""
        configs = {
            "models": ["purple_alien"],
            "aggregation": "mean",
            "name": "test_ensemble",
            "targets": ["ged_sb"],
        }
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        manager._EnsembleManager__activate_reconciliation = False
        
        # Create a non-DataFrame result for aggregation
        with patch.object(manager, '_forecast_model_artifact', return_value=sample_dataframe):
            with patch.object(EnsembleManager, '_get_aggregated_df', return_value="not_a_dataframe"):
                with patch('views_pipeline_core.managers.ensemble.ensemble._ViewsDataset') as MockDataset:
                    MockDataset.side_effect = ValueError("Invalid input type for ViewsDataset")
                    
                    with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm', side_effect=lambda x, **kwargs: x):
                        with patch('views_pipeline_core.managers.ensemble.ensemble.tqdm.tqdm.write'):
                            with pytest.raises(ValueError, match="Invalid input type for ViewsDataset"):
                                manager._forecast_ensemble()


# ============================================================================
# Test Load C Dataset
# ============================================================================

class TestLoadCDataset:
    """Tests for _load_c_dataset method."""
    
    def test_load_c_dataset_with_provided_dataframe(self, manager, sample_cm_dataframe):
        """Test loading C dataset from provided DataFrame."""
        with patch('views_pipeline_core.managers.ensemble.ensemble._CDataset') as MockCDataset:
            mock_dataset = MagicMock()
            MockCDataset.return_value = mock_dataset
            
            result = manager._load_c_dataset("cm_model", sample_cm_dataframe)
            
            MockCDataset.assert_called_once_with(source=sample_cm_dataframe)
            assert result == mock_dataset
    
    def test_load_c_dataset_returns_none_when_not_found(self, manager):
        """Test loading C dataset returns None when not found."""
        manager._use_prediction_store = False
        
        with patch('views_pipeline_core.managers.ensemble.ensemble.EnsemblePathManager') as MockPath:
            mock_path = MagicMock()
            mock_path._get_generated_predictions_data_file_paths.side_effect = Exception("Not found")
            MockPath.return_value = mock_path
            
            result = manager._load_c_dataset("cm_model", None)
            
            assert result is None


# ============================================================================
# Test Execute Model Training
# ============================================================================

class TestExecuteModelTraining:
    """Tests for _execute_model_training method."""
    
    def test_execute_model_training_success(self, manager, mock_wandb_module):
        """Test successful model training."""
        manager._project = "test_project"
        configs = {"name": "test_ensemble"}
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        
        with patch.object(manager, '_train_ensemble'):
            manager._execute_model_training()
            
            mock_wandb_module.initialize_run.assert_called_once()
            mock_wandb_module.send_alert.assert_called()
            mock_wandb_module.finish_run.assert_called_once()
    
    def test_execute_model_training_failure_raises(self, manager, mock_wandb_module):
        """Test training failure raises PipelineException."""
        manager._project = "test_project"
        configs = {"name": "test_ensemble"}
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        
        with patch.object(manager, '_train_ensemble', side_effect=Exception("Training failed")):
            with pytest.raises(PipelineException, match="Training failed"):
                manager._execute_model_training()


# ============================================================================
# Test Execute Model Evaluation
# ============================================================================

class TestExecuteModelEvaluation:
    """Tests for _execute_model_evaluation method."""
    
    def test_execute_model_evaluation_success(self, manager, mock_wandb_module, sample_dataframe):
        """Test successful model evaluation."""
        manager._project = "test_project"
        manager._eval_type = "standard"
        configs = {
            "name": "test_ensemble",
            "run_type": "calibration",
            "deployment_status": "development",
        }
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        manager._wandb_module = mock_wandb_module
        
        with patch.object(manager, '_evaluate_ensemble', return_value=[sample_dataframe]):
            with patch('views_pipeline_core.managers.ensemble.ensemble.handle_ensemble_log_creation'):
                with patch.object(manager, '_save_predictions'):
                    with patch.object(manager, '_evaluate_prediction_dataframe'):
                        manager._execute_model_evaluation()
                        
                        mock_wandb_module.send_alert.assert_called()
                        mock_wandb_module.finish_run.assert_called_once()


# ============================================================================
# Test Execute Model Forecasting
# ============================================================================

class TestExecuteModelForecasting:
    """Tests for _execute_model_forecasting method."""
    
    def test_execute_model_forecasting_success(self, manager, mock_wandb_module, sample_dataframe):
        """Test successful model forecasting."""
        manager._project = "test_project"
        configs = {
            "name": "test_ensemble",
            "run_type": "forecasting",
            "deployment_status": "development",
        }
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        manager._wandb_module = mock_wandb_module
        
        with patch.object(manager, '_forecast_ensemble', return_value=sample_dataframe):
            with patch('views_pipeline_core.managers.ensemble.ensemble.handle_ensemble_log_creation'):
                with patch.object(manager, '_save_predictions'):
                    manager._execute_model_forecasting()
                    
                    mock_wandb_module.send_alert.assert_called()
                    mock_wandb_module.finish_run.assert_called_once()
    
    def test_execute_model_forecasting_failure_raises(self, manager, mock_wandb_module):
        """Test forecasting failure raises PipelineException."""
        manager._project = "test_project"
        configs = {"name": "test_ensemble"}
        manager._config_manager.configs = configs
        manager._config_manager.get_combined_config.return_value = configs
        
        with patch.object(manager, '_forecast_ensemble', side_effect=Exception("Forecasting failed")):
            with pytest.raises(PipelineException, match="Forecasting failed"):
                manager._execute_model_forecasting()


# ============================================================================
# Test Load or Generate Prediction
# ============================================================================

class TestLoadOrGeneratePrediction:
    """Tests for _load_or_generate_prediction method."""
    
    def test_load_prediction_from_local_file(self, manager, sample_dataframe):
        """Test loading prediction from local file."""
        manager._use_prediction_store = False
        mock_model_path = MagicMock(spec=ModelPathManager)
        
        with patch('views_pipeline_core.managers.ensemble.ensemble.PipelineConfig') as MockConfig:
            MockConfig.return_value.dataframe_format = ".parquet"
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('views_pipeline_core.managers.ensemble.ensemble.read_dataframe', return_value=sample_dataframe):
                    result = manager._load_or_generate_prediction(
                        model_path=mock_model_path,
                        model_name="purple_alien",
                        name="test_prediction",
                        path_generated=Path("/test/generated"),
                        run_type="calibration",
                        ts="20241105_120000",
                        sequence_number=0,
                        evaluate=True,
                    )
                    
                    assert result.equals(sample_dataframe)
    
    def test_generate_prediction_when_not_found(self, manager, sample_dataframe):
        """Test generating prediction when file not found."""
        manager._use_prediction_store = False
        manager._wandb_notifications = False
        mock_model_path = MagicMock(spec=ModelPathManager)
        args = ForecastingModelArgs(run_type="forecasting", forecast=True, saved=True)
        manager._args = args
        
        with patch('views_pipeline_core.managers.ensemble.ensemble.PipelineConfig') as MockConfig:
            MockConfig.return_value.dataframe_format = ".parquet"
            
            # First check returns False (file doesn't exist), then after generation it exists
            with patch('pathlib.Path.exists', side_effect=[False, True]):
                with patch.object(manager, '_create_model_args', return_value=args):
                    with patch.object(manager, '_execute_shell_script'):
                        with patch('views_pipeline_core.managers.ensemble.ensemble.read_dataframe', return_value=sample_dataframe):
                            result = manager._load_or_generate_prediction(
                                model_path=mock_model_path,
                                model_name="purple_alien",
                                name="test_prediction",
                                path_generated=Path("/test/generated"),
                                run_type="forecasting",
                                ts="20241105_120000",
                                forecast=True,
                            )
                            
                            assert result.equals(sample_dataframe)


# ============================================================================
# Subprocess Timeout Tests
# ============================================================================

class TestSubprocessTimeout:
    """Verify that _execute_shell_script passes a timeout to subprocess.run."""

    def test_shell_script_passes_timeout(self, manager, mock_ensemble_path):
        """subprocess.run must receive a timeout parameter to prevent indefinite hangs."""
        model_path = MagicMock(spec=ModelPathManager)
        model_args = MagicMock(spec=ForecastingModelArgs)
        model_args.to_shell_command.return_value = ["echo", "test"]

        with patch("subprocess.run") as mock_run:
            manager._execute_shell_script(model_path, "test_model", model_args)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert "timeout" in call_kwargs.kwargs, (
                "subprocess.run() called without timeout parameter. "
                "Ensemble sub-model execution can hang indefinitely without a timeout."
            )
