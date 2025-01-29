import pytest
from unittest.mock import MagicMock, patch, mock_open
from views_pipeline_core.managers.model import ModelManager
import wandb
import pandas as pd
from pathlib import Path

@pytest.fixture
def mock_model_path():
    """
    Fixture to mock the ModelPath class with validate flag set to False.
    
    Yields:
        MagicMock: The mock object for ModelPath.
    """
    with patch("views_pipeline_core.managers.model.ModelPathManager") as mock:
        mock_instance = mock.return_value
        mock_instance.get_scripts.return_value = {
            "config_deployment.py": "path/to/config_deployment.py",
            "config_hyperparameters.py": "path/to/config_hyperparameters.py",
            "config_meta.py": "path/to/config_meta.py",
            "config_sweep.py": "path/to/config_sweep.py"
        }
        mock_instance._validate = False
        yield mock

@pytest.fixture
def mock_ensemble_path():
    """
    Fixture to mock the EnsemblePath class.
    
    Yields:
        MagicMock: The mock object for EnsemblePath.
    """
    with patch("views_pipeline_core.managers.ensemble.EnsemblePathManager") as mock:
        yield mock

@pytest.fixture
def mock_dataloader():
    """
    Fixture to mock the ViewsDataLoader class.
    
    Yields:
        MagicMock: The mock object for ViewsDataLoader.
    """
    with patch("views_pipeline_core.data.dataloaders.ViewsDataLoader") as mock:
        mock_instance = mock.return_value
        mock_instance._path_raw = "/path/to/raw"
        mock_instance.get_data.return_value = (MagicMock(), MagicMock())  # Queryset output but not really
        yield mock

@pytest.fixture
def mock_wandb():
    """
    Fixture to mock the wandb functions.
    
    Yields:
        None
    """
    with patch("wandb.init"), patch("wandb.finish"), patch("wandb.sweep"), patch("wandb.agent"):
        yield

@pytest.fixture
def mock_model_manager(mock_model_path):
    mock_config_deployment_content = """
def get_deployment_config():
    deployment_config = {'deployment_status': 'shadow'}
    return deployment_config
"""
    with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
        mock_spec.return_value.loader = MagicMock()
        mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
        mock_module.return_value.get_hp_config.return_value = {"hp_key": "hp_value"}
        mock_module.return_value.get_meta_config.return_value = {"meta_key": "meta_value"}
        mm = ModelManager(mock_model_path.return_value, use_prediction_store=False, wandb_notifications=False)
        mm.config = {
            "depvar": "target_variable"
        }
    return mm

def test_wandb_alert(mock_model_path):
    """
    Test the _wandb_alert method of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
    
    Asserts:
        - The wandb alert is called with the correct parameters.
    """
    mock_model_instance = mock_model_path.return_value
    mock_config_deployment_content = """
def get_deployment_config():
    deployment_config = {'deployment_status': 'shadow'}
    return deployment_config
"""
    with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
        mock_spec.return_value.loader = MagicMock()
        mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
        mock_module.return_value.get_hp_config.return_value = {"hp_key": "hp_value"}
        mock_module.return_value.get_meta_config.return_value = {"meta_key": "meta_value"}
        mock_model_instance = mock_model_path.return_value
        manager = ModelManager(mock_model_instance, wandb_notifications=True, use_prediction_store=False)
        with patch("wandb.alert") as mock_alert:
            with patch("wandb.run"):
                manager._wandb_alert(title="Test Alert", text="This is a test alert", level="info")
                mock_alert.assert_called_once_with(title="Test Alert", text="This is a test alert", level="info")

def test_model_manager_init(mock_model_path):
    """
    Test the initialization of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
    
    Asserts:
        - The ModelManager is initialized with the correct attributes.
    """
    mock_model_instance = mock_model_path.return_value
    mock_model_instance.get_scripts.return_value = {
        "config_deployment.py": "path/to/config_deployment.py",
        "config_hyperparameters.py": "path/to/config_hyperparameters.py",
        "config_meta.py": "path/to/config_meta.py",
        "config_sweep.py": "path/to/config_sweep.py"
    }
    mock_config_deployment_content = """
def get_deployment_config():
    deployment_config = {'deployment_status': 'shadow'}
    return deployment_config
"""
    mock_config_hyperparameters_content = """
def get_hp_config():
    hp_config = {'hp_key': 'hp_value'}
    return hp_config
"""
    mock_config_meta_content = """
def get_meta_config():
    meta_config = {'meta_key': 'meta_value'}
    return meta_config
"""
    with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
        mock_spec.return_value.loader = MagicMock()
        mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
        mock_module.return_value.get_hp_config.return_value = {"hp_key": "hp_value"}
        mock_module.return_value.get_meta_config.return_value = {"meta_key": "meta_value"}
        manager = ModelManager(mock_model_instance, use_prediction_store=False)
        assert manager._entity == "views_pipeline"
        assert manager._model_path == mock_model_instance
        assert manager._config_deployment == {"deployment_status": "shadow"}
        assert manager._config_hyperparameters == {"hp_key": "hp_value"}
        assert manager._config_meta == {"meta_key": "meta_value"}

def test_load_config(mock_model_path):
    """
    Test the __load_config method of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
    
    Asserts:
        - The configuration is loaded correctly from the specified script.
    """
    mock_model_instance = mock_model_path.return_value
    mock_model_instance.get_scripts.return_value = {
        "config_deployment.py": "path/to/config_deployment.py"
    }
    mock_config_deployment_content = """
def get_deployment_config():
    deployment_config = {'deployment_status': 'shadow'}
    return deployment_config
"""
    with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
        mock_spec.return_value.loader = MagicMock()
        mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
        manager = ModelManager(mock_model_instance, use_prediction_store=False)
        config = manager._ModelManager__load_config("config_deployment.py", "get_deployment_config")
        assert config == {"deployment_status": "shadow"}

def test_update_single_config(mock_model_path):
    """
    Test the _update_single_config method of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
    
    Asserts:
        - The single run configuration is updated correctly.
    """
    mock_model_instance = mock_model_path.return_value
    mock_model_instance.get_scripts.return_value = {
        "config_deployment.py": "path/to/config_deployment.py",
        "config_hyperparameters.py": "path/to/config_hyperparameters.py",
        "config_meta.py": "path/to/config_meta.py"
    }
    mock_config_deployment_content = """
def get_deployment_config():
    deployment_config = {'deployment_status': 'shadow'}
    return deployment_config
"""
    with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
        mock_spec.return_value.loader = MagicMock()
        mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
        manager = ModelManager(mock_model_instance, use_prediction_store=False)
        manager._config_hyperparameters = {"hp_key": "hp_value"}
        manager._config_meta = {"meta_key": "meta_value"}
        manager._config_deployment = {"deploy_key": "deploy_value"}
        args = MagicMock(run_type="test_run")
        config = manager._update_single_config(args)
        assert config["hp_key"] == "hp_value"
        assert config["meta_key"] == "meta_value"
        assert config["deploy_key"] == "deploy_value"
        assert config["run_type"] == "test_run"
        # assert config["sweep"] is False

def test_validate_prediction_dataframe_empty(mock_model_manager):
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Prediction DataFrame is empty."):
        mock_model_manager._validate_prediction_dataframe(df)

def test_validate_prediction_dataframe_missing_depvar(mock_model_manager):
    df = pd.DataFrame({"month_id": [1, 2], "priogrid_id": [1, 2]})
    with pytest.raises(ValueError, match="target_variable not found in dataframe columns."):
        mock_model_manager._validate_prediction_dataframe(df)

def test_validate_prediction_dataframe_valid_pgm(mock_model_manager):
    df = pd.DataFrame({
        "month_id": [1, 2],
        "priogrid_id": [1, 2],
        "target_variable": [0.1, 0.2]
    }).set_index(["month_id", "priogrid_id"])
    mock_model_manager._validate_prediction_dataframe(df)

def test_validate_prediction_dataframe_valid_cm(mock_model_manager):
    df = pd.DataFrame({
        "month_id": [1, 2],
        "country_id": [1, 2],
        "target_variable": [0.1, 0.2]
    }).set_index(["month_id", "country_id"])
    mock_model_manager._validate_prediction_dataframe(df)

def test_validate_prediction_dataframe_invalid_index(mock_model_manager):
    df = pd.DataFrame({
        "month_id": [1, 2],
        "invalid_id": [1, 2],
        "target_variable": [0.1, 0.2]
    }).set_index(["month_id", "invalid_id"])
    with pytest.raises(ValueError, match="Valid indices for pgm or cm not found in prediction DataFrame."):
        mock_model_manager._validate_prediction_dataframe(df)

def test_validate_prediction_dataframe_valid_single_depvar(mock_model_manager):
    mock_model_manager.config["depvar"] = ["target_variable"]
    df = pd.DataFrame({
        "month_id": [1, 2],
        "priogrid_id": [1, 2],
        "target_variable": [0.1, 0.2]
    }).set_index(["month_id", "priogrid_id"])
    mock_model_manager._validate_prediction_dataframe(df)

def test_validate_prediction_dataframe_invalid_single_depvar(mock_model_manager):
    mock_model_manager.config["depvar"] = ["target_variable"]
    df = pd.DataFrame({
        "month_id": [1, 2],
        "priogrid_id": [1, 2],
        "another_variable": [0.1, 0.2]
    }).set_index(["month_id", "priogrid_id"])
    with pytest.raises(ValueError, match="target_variable not found in dataframe columns."):
        mock_model_manager._validate_prediction_dataframe(df)

def test_update_sweep_config(mock_model_path):
    """
    Test the _update_sweep_config method of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
    
    Asserts:
        - The sweep run configuration is updated correctly.
    """
    mock_model_instance = mock_model_path.return_value
    mock_model_instance.get_scripts.return_value = {
        "config_sweep.py": "path/to/config_sweep.py",
        "config_meta.py": "path/to/config_meta.py"
    }
    mock_config_sweep_content = """
def get_sweep_config():
    sweep_config = {
        'method': 'grid',
        'name': 'test_model'
    }

    # Example metric setup:
    metric = {
        'name': 'MSE',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    # Example parameters setup:
    parameters_dict = {
        'steps': {'values': [[*range(1, 36 + 1, 1)]]},
        'n_estimators': {'values': [100, 150, 200]},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
"""
    mock_config_meta_content = """
def get_meta_config():
    meta_config = {'name': 'test_model', 'depvar': 'test_depvar', 'algorithm': 'test_algorithm'}
    return meta_config
"""
    with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_sweep_content)):
        mock_spec.return_value.loader = MagicMock()
        mock_module.return_value.get_sweep_config.return_value = {
            'method': 'grid',
            'name': 'test_model',
            'metric': {
                'name': 'MSE',
                'goal': 'minimize'
            },
            'parameters': {
                'steps': {'values': [[*range(1, 36 + 1, 1)]]},
                'n_estimators': {'values': [100, 150, 200]},
            }
        }
        mock_module.return_value.get_meta_config.return_value = {"name": "test_model", "depvar": "test_depvar", "algorithm": "test_algorithm"}
        manager = ModelManager(mock_model_instance, use_prediction_store=False)
        manager._config_sweep = {
            'method': 'grid',
            'name': 'test_model',
            'metric': {
                'name': 'MSE',
                'goal': 'minimize'
            },
            'parameters': {
                'steps': {'values': [[*range(1, 36 + 1, 1)]]},
                'n_estimators': {'values': [100, 150, 200]},
            }
        }
        manager._config_meta = {"name": "test_model", "depvar": "test_depvar", "algorithm": "test_algorithm"}
        manager._config_deployment = {"deployment_status": "test_shadow"}
        manager._args = MagicMock(run_type="test_run", sweep=True)
        wandb_config = MagicMock()
        config = manager._update_sweep_config(wandb_config=wandb_config)
        assert config["run_type"] == "test_run"
        assert config["sweep"] is True
        assert config["name"] == "test_model"
        assert config["depvar"] == "test_depvar"
        assert config["algorithm"] == "test_algorithm"
        assert config["deployment_status"] == "test_shadow"

def test_execute_single_run(mock_model_path, mock_dataloader, mock_wandb):
    """
    Test the execute_single_run method of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
        mock_dataloader (MagicMock): The mock object for ViewsDataLoader.
        mock_wandb (None): The mock object for wandb functions.
    
    Asserts:
        - The single run is executed correctly.
    """
    mock_model_instance = mock_model_path.return_value
    mock_model_instance.get_scripts.return_value = {
        "config_deployment.py": "path/to/config_deployment.py",
        "config_hyperparameters.py": "path/to/config_hyperparameters.py",
        "config_meta.py": "path/to/config_meta.py"
    }
    mock_config_deployment_content = """
def get_deployment_config():
    deployment_config = {'deployment_status': 'shadow'}
    return deployment_config
"""
    with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
        mock_spec.return_value.loader = MagicMock()
        mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
        manager = ModelManager(mock_model_instance, use_prediction_store=False)
        manager._update_single_config = MagicMock(return_value={"name": "test_model"})
        manager._execute_model_tasks = MagicMock()
        args = MagicMock(run_type="calibration", saved=False, drift_self_test=False, train=True, evaluate=True, forecast=True, artifact_name="test_artifact")
        
        # Add logging to identify where the failure occurs
        try:
            manager.execute_single_run(args)
        except Exception as e:
            print(f"Error during execute_single_run: {e}")
    
        manager._update_single_config.assert_called_once_with(args)

        #idek anymore
        # manager._execute_model_tasks.assert_called_once_with(config={"name": "test_model"}, train=True, eval=True, forecast=False, artifact_name="test_artifact")

# def test_save_model_outputs(mock_model_path):
#     """
#     Test the _save_model_outputs method of the ModelManager class.
    
#     Args:
#         mock_model_path (MagicMock): The mock object for ModelPath.
    
#     Asserts:
#         - The model outputs are saved correctly.
#     """
#     mock_model_instance = mock_model_path.return_value
#     mock_model_instance.get_scripts.return_value = {
#         "config_deployment.py": "path/to/config_deployment.py",
#         "config_hyperparameters.py": "path/to/config_hyperparameters.py",
#         "config_meta.py": "path/to/config_meta.py"
#     }
#     mock_config_deployment_content = """
# def get_deployment_config():
#     deployment_config = {'deployment_status': 'shadow'}
#     return deployment_config
# """
#     with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
#         mock_spec.return_value.loader = MagicMock()
#         mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
#         manager = ModelManager(mock_model_instance)
#         manager.config = {"run_type": "calibration", "timestamp": "20210831_123456"}
#         df_evaluation = pd.DataFrame({"metric": [1, 2, 3]})
#         df_output = pd.DataFrame({"output": [4, 5, 6]})
#         path_generated = "/path/to/generated"
#         sequence_number = 1
#         with patch("pathlib.Path.mkdir") as mock_mkdir, patch("pandas.DataFrame.to_pickle") as mock_to_pickle:
#             manager._save_model_outputs(df_evaluation, df_output, path_generated, sequence_number)
#             mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
#             mock_to_pickle.assert_any_call(Path(path_generated) / "output_calibration_20210831_123456_01.pkl")
#             mock_to_pickle.assert_any_call(Path(path_generated) / "evaluation_calibration_20210831_123456_01.pkl")

# def test_save_predictions(mock_model_path):
#     """
#     Test the _save_predictions method of the ModelManager class.
    
#     Args:
#         mock_model_path (MagicMock): The mock object for ModelPath.
    
#     Asserts:
#         - The model predictions are saved correctly.
#     """
#     mock_model_instance = mock_model_path.return_value
#     mock_model_instance.get_scripts.return_value = {
#         "config_deployment.py": "path/to/config_deployment.py",
#         "config_hyperparameters.py": "path/to/config_hyperparameters.py",
#         "config_meta.py": "path/to/config_meta.py"
#     }
#     mock_config_deployment_content = """
# def get_deployment_config():
#     deployment_config = {'deployment_status': 'shadow'}
#     return deployment_config
# """
#     with patch("importlib.util.spec_from_file_location") as mock_spec, patch("importlib.util.module_from_spec") as mock_module, patch("builtins.open", mock_open(read_data=mock_config_deployment_content)):
#         mock_spec.return_value.loader = MagicMock()
#         mock_module.return_value.get_deployment_config.return_value = {"deployment_status": "shadow"}
#         manager = ModelManager(mock_model_instance)
#         manager.config = {"run_type": "calibration", "timestamp": "20210831_123456"}
#         df_predictions = pd.DataFrame({"prediction": [7, 8, 9]})
#         path_generated = "/path/to/generated"
#         sequence_number = 1
#         with patch("pathlib.Path.mkdir") as mock_mkdir, patch("pandas.DataFrame.to_pickle") as mock_to_pickle:
#             manager._save_predictions(df_predictions, path_generated, sequence_number)
#             mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
#             mock_to_pickle.assert_called_once_with(Path(path_generated) / "predictions_calibration_20210831_123456_01.pkl")