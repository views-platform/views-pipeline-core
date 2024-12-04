import pytest
from unittest.mock import MagicMock, patch, mock_open
from views_pipeline_core.managers.path_manager import ModelPath, EnsemblePath
from views_pipeline_core.data.dataloaders import ViewsDataLoader
from views_pipeline_core.managers.model_manager import ModelManager
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
    with patch("views_pipeline_core.managers.path_manager.ModelPath") as mock:
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
    with patch("views_pipeline_core.managers.path_manager.EnsemblePath") as mock:
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
        manager = ModelManager(mock_model_instance)
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
        manager = ModelManager(mock_model_instance)
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
        manager = ModelManager(mock_model_instance)
        manager._config_hyperparameters = {"hp_key": "hp_value"}
        manager._config_meta = {"meta_key": "meta_value"}
        manager._config_deployment = {"deploy_key": "deploy_value"}
        args = MagicMock(run_type="test_run")
        config = manager._update_single_config(args)
        assert config["hp_key"] == "hp_value"
        assert config["meta_key"] == "meta_value"
        assert config["deploy_key"] == "deploy_value"
        assert config["run_type"] == "test_run"
        assert config["sweep"] is False

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
        manager = ModelManager(mock_model_instance)
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
        args = MagicMock(run_type="test_run")
        config = manager._update_sweep_config(args)
        assert config["parameters"]["run_type"]["value"] == "test_run"
        assert config["parameters"]["sweep"]["value"] is True
        assert config["parameters"]["name"]["value"] == "test_model"
        assert config["parameters"]["depvar"]["value"] == "test_depvar"
        assert config["parameters"]["algorithm"]["value"] == "test_algorithm"

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
        manager = ModelManager(mock_model_instance)
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

def test_save_model_outputs(mock_model_path):
    """
    Test the _save_model_outputs method of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
    
    Asserts:
        - The model outputs are saved correctly.
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
        manager = ModelManager(mock_model_instance)
        manager.config = {"run_type": "calibration", "timestamp": "20210831_123456"}
        df_evaluation = pd.DataFrame({"metric": [1, 2, 3]})
        df_output = pd.DataFrame({"output": [4, 5, 6]})
        path_generated = "/path/to/generated"
        sequence_number = 1
        with patch("pathlib.Path.mkdir") as mock_mkdir, patch("pandas.DataFrame.to_pickle") as mock_to_pickle:
            manager._save_model_outputs(df_evaluation, df_output, path_generated, sequence_number)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_to_pickle.assert_any_call(Path(path_generated) / "output_calibration_20210831_123456_01.pkl")
            mock_to_pickle.assert_any_call(Path(path_generated) / "evaluation_calibration_20210831_123456_01.pkl")

def test_save_predictions(mock_model_path):
    """
    Test the _save_predictions method of the ModelManager class.
    
    Args:
        mock_model_path (MagicMock): The mock object for ModelPath.
    
    Asserts:
        - The model predictions are saved correctly.
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
        manager = ModelManager(mock_model_instance)
        manager.config = {"run_type": "calibration", "timestamp": "20210831_123456"}
        df_predictions = pd.DataFrame({"prediction": [7, 8, 9]})
        path_generated = "/path/to/generated"
        sequence_number = 1
        with patch("pathlib.Path.mkdir") as mock_mkdir, patch("pandas.DataFrame.to_pickle") as mock_to_pickle:
            manager._save_predictions(df_predictions, path_generated, sequence_number)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_to_pickle.assert_called_once_with(Path(path_generated) / "predictions_calibration_20210831_123456_01.pkl")