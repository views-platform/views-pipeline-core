import pytest
from unittest.mock import patch, MagicMock
from views_pipeline_core.managers.ensemble_manager import EnsembleManager
from views_pipeline_core.models.check import ensemble_model_check
from views_pipeline_core.managers.path_manager import EnsemblePath, ModelPath
from views_pipeline_core.managers.model_manager import ModelManager
import logging
import pandas as pd
import os

@pytest.fixture
def mock_model_path():
    mock_path = MagicMock()
    mock_path.model_dir = "/path/to/models/test_model"
    return mock_path

@pytest.fixture
def mock_get_scripts():
    mock_scripts = MagicMock()
    mock_scripts.get_scripts.return_value = {"config_deployment.py": "mock_script_path"}
    return mock_scripts


# Parameterized test cases for different scenarios
@pytest.mark.parametrize(
    "run_type, train, evaluate, forecast, use_saved, eval_type, expected_command",
    [
        (
            "test", True, False, False, True, "standard",
            [
                "/path/to/models/test_model/run.sh",
                "--run_type", "test",
                "--train",
                "--saved",
                "--eval_type", "standard"
            ]
        ),
        (
            "forecast", False, True, True, False, "detailed",
            [
                "/path/to/models/test_model/run.sh",
                "--run_type", "forecast",
                "--evaluate",
                "--forecast",
                "--eval_type", "detailed"
            ]
        ),
        (
            "calibration", False, False, False, False, "minimal",
            [
                "/path/to/models/test_model/run.sh",
                "--run_type", "calibration",
                "--eval_type", "minimal"
            ]
        )
    ]
)

def test_get_shell_command(mock_model_path, run_type, train, evaluate, forecast, use_saved, eval_type, expected_command):
    """
    Test the _get_shell_command method with various input combinations to ensure it generates the correct shell command.
    """
    command = EnsembleManager._get_shell_command(
        mock_model_path, run_type, train, evaluate, forecast, use_saved, eval_type
    )
    assert command == expected_command



@pytest.fixture
def sample_data():
    """
    Fixture to provide common sample data for the aggregation tests.
    """
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    return [df1, df2]

def test_get_aggregated_df_mean(sample_data):
    """
    Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using mean.
    """
    df_to_aggregate = sample_data

    result = EnsembleManager._get_aggregated_df(df_to_aggregate, "mean")
    expected = pd.DataFrame({"A": [3.0, 4.0], "B": [5.0, 6.0]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    
    pd.testing.assert_frame_equal(result, expected, check_like=True)

def test_get_aggregated_df_median(sample_data):
    """
    Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using median.
    """
    df_to_aggregate = sample_data

    result = EnsembleManager._get_aggregated_df(df_to_aggregate, "median")
    expected = pd.DataFrame({"A": [3.0, 4.0], "B": [5.0, 6.0]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    
    pd.testing.assert_frame_equal(result, expected, check_like=True)

def test_get_aggregated_df_invalid_aggregation(sample_data):
    """
    Test the _get_aggregated_df method for invalid aggregation method.
    """
   
    with pytest.raises(ValueError, match="Invalid aggregation method: invalid_aggregation"):
        EnsembleManager._get_aggregated_df(sample_data, "invalid_aggregation")

        
# # @patch.object(EnsembleManager, '_execute_model_tasks')
# # @patch.object(ModelManager, '_update_single_config')
# # def test_execute_single_run(mock_update_single_config, mock_execute_model_tasks, ensemble_manager):
# #     """
# #     Test the execute_single_run method to ensure it correctly executes a single run.
# #     """
# #     args = MagicMock()
# #     args.run_type = "test"
# #     args.eval_type = "standard"
# #     mock_update_single_config.return_value = {"name": "test_model"}

# #     ensemble_manager.execute_single_run(args)

# #     mock_update_single_config.assert_called_once_with(args)
# #     mock_execute_model_tasks.assert_called_once()




@pytest.fixture
def mock_args():
    """Fixture to provide mock arguments for testing."""
    class Args:
        def __init__(self, train=False, evaluate=False, forecast=False, run_type="test", eval_type="default", saved=False):
            self.train = train
            self.evaluate = evaluate
            self.forecast = forecast
            self.run_type = run_type
            self.eval_type = eval_type
            self.saved = saved

    return Args(train=True, evaluate=False, forecast=False, run_type="test_run", eval_type="test_eval", saved=False)

@patch('views_pipeline_core.managers.ensemble_manager.EnsembleManager._execute_model_tasks')
@patch('views_pipeline_core.managers.model_manager.ModelManager._update_single_config')
@patch('views_pipeline_core.models.check.ensemble_model_check')
@patch('views_pipeline_core.managers.ensemble_manager.logger')
def test_execute_single_run(mock_logger, mock_ensemble_model_check, mock_update_single_config, mock_execute_model_tasks, mock_model_path, mock_args):
    """Test the execute_single_run method."""
    
    # Setup mocks
    mock_model_path.get_scripts.return_value = {"config_deployment.py": "mock_script_path"}
    mock_update_single_config.return_value = {"name": "test_model"}
    mock_load_config = MagicMock(return_value={"name": "test_model"})

    # Instantiate EnsembleManager and execute the method
    manager = EnsembleManager(ensemble_path=mock_model_path)
    manager._script_paths = {"config_deployment.py": "mock_script_path"}  # Ensure it behaves like a dict
    manager._load_config = mock_load_config  # Mock _load_config to avoid real config loading


    manager.execute_single_run(mock_args)

    # Assertions for expected behavior
    mock_update_single_config.assert_called_once_with(mock_args)
    assert manager._project == "test_model_test_run"
    assert manager._eval_type == "test_eval"
    mock_ensemble_model_check.assert_not_called()
    mock_execute_model_tasks.assert_called_once_with(
        config={"name": "test_model"}, 
        train=True, 
        eval=False, 
        forecast=False, 
        use_saved=False
    )
    mock_logger.error.assert_not_called()

    # Test exception handling
    mock_execute_model_tasks.side_effect = Exception("Test error")

    # Run the method and check for exception logging (not for raising directly)
    manager.execute_single_run(mock_args)

    # Check that the error was logged
    mock_logger.error.assert_called_once_with("Error during single run execution: Test error")








# @patch.object(EnsembleManager, '_execute_model_tasks')
# @patch.object(ModelManager, '_update_single_config')
# def test_execute_single_run(mock_update_single_config, mock_execute_model_tasks, ensemble_manager):
#     """
#     Test the execute_single_run method to ensure it correctly updates the config and executes model tasks.
#     """
#     args = MagicMock()
#     args.run_type = "test"
#     args.eval_type = "standard"
#     args.train = True
#     args.evaluate = False
#     args.forecast = False
#     args.saved = True

#     mock_update_single_config.return_value = {"name": "test_model"}

#     ensemble_manager.execute_single_run(args)

#     mock_update_single_config.assert_called_once_with(args)
#     mock_execute_model_tasks.assert_called_once_with(
#         config={"name": "test_model", "run_type": "test"},
#         train=True,
#         eval=False,
#         forecast=False,
#         use_saved=True
#     )

# @patch('subprocess.run')
# def test_train_model_artifact(mock_subprocess_run, ensemble_manager):
#     """
#     Test the _train_model_artifact method to ensure it correctly generates the shell command and runs it.
#     """
#     model_name = "test_model"
#     run_type = "test"
#     use_saved = True

#     ensemble_manager._train_model_artifact(model_name, run_type, use_saved)

#     mock_subprocess_run.assert_called_once()

# @patch('subprocess.run')
# @patch('builtins.open', new_callable=MagicMock)
# @patch('pickle.load')
# def test_evaluate_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
#     """
#     Test the _evaluate_model_artifact method to ensure it correctly evaluates the model and loads predictions.
#     """
#     model_name = "test_model"
#     run_type = "test"
#     eval_type = "standard"

#     mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

#     preds = ensemble_manager._evaluate_model_artifact(model_name, run_type, eval_type)

#     assert len(preds) > 0
#     mock_subprocess_run.assert_called()
#     mock_open.assert_called()
#     mock_pickle_load.assert_called()

# @patch('subprocess.run')
# @patch('builtins.open', new_callable=MagicMock)
# @patch('pickle.load')
# def test_forecast_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
#     """
#     Test the _forecast_model_artifact method to ensure it correctly forecasts the model and loads predictions.
#     """
#     model_name = "test_model"
#     run_type = "test"

#     mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

#     df = ensemble_manager._forecast_model_artifact(model_name, run_type)

#     assert not df.empty
#     mock_subprocess_run.assert_called()
#     mock_open.assert_called()
#     mock_pickle_load.assert_called()

# @patch.object(EnsembleManager, '_train_model_artifact')
# def test_train_ensemble(mock_train_model_artifact, ensemble_manager):
#     """
#     Test the _train_ensemble method to ensure it correctly trains all models in the ensemble.
#     """
#     ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"]}

#     ensemble_manager._train_ensemble(use_saved=True)

#     assert mock_train_model_artifact.call_count == 2

# @patch.object(EnsembleManager, '_evaluate_model_artifact')
# @patch.object(EnsembleManager, '_get_aggregated_df')
# def test_evaluate_ensemble(mock_get_aggregated_df, mock_evaluate_model_artifact, ensemble_manager):
#     """
#     Test the _evaluate_ensemble method to ensure it correctly evaluates all models in the ensemble and aggregates results.
#     """
#     ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
#     mock_evaluate_model_artifact.return_value = [pd.DataFrame({"A": [1, 2], "B": [3, 4]})]

#     ensemble_manager._evaluate_ensemble(eval_type="standard")

#     assert mock_evaluate_model_artifact.call_count == 2
#     mock_get_aggregated_df.assert_called()

# @patch.object(EnsembleManager, '_forecast_model_artifact')
# @patch.object(EnsembleManager, '_get_aggregated_df')
# def test_forecast_ensemble(mock_get_aggregated_df, mock_forecast_model_artifact, ensemble_manager):
#     """
#     Test the _forecast_ensemble method to ensure it correctly forecasts all models in the ensemble and aggregates results.
#     """
#     ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
#     mock_forecast_model_artifact.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

#     ensemble_manager._forecast_ensemble()

#     assert mock_forecast_model_artifact.call_count == 2
#     mock_get_aggregated_df.assert_called()
#     df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
#     df_to_aggregate = [df1, df2]

#     result = EnsembleManager._get_aggregated_df(df_to_aggregate, "median")
#     expected = pd.DataFrame({"A": [3, 4], "B": [5, 6]})

#     pd.testing.assert_frame_equal(result, expected)

# @patch.object(EnsembleManager, '_execute_model_tasks')
# @patch.object(ModelManager, '_update_single_config')
# def test_execute_single_run(mock_update_single_config, mock_execute_model_tasks, ensemble_manager):
#     """
#     Test the execute_single_run method to ensure it correctly updates the config and executes model tasks.
#     """
#     args = MagicMock()
#     args.run_type = "test"
#     args.eval_type = "standard"
#     args.train = True
#     args.evaluate = False
#     args.forecast = False
#     args.saved = True

#     mock_update_single_config.return_value = {"name": "test_model"}

#     ensemble_manager.execute_single_run(args)

#     mock_update_single_config.assert_called_once_with(args)
#     mock_execute_model_tasks.assert_called_once_with(
#         config={"name": "test_model", "run_type": "test"},
#         train=True,
#         eval=False,
#         forecast=False,
#         use_saved=True
#     )

# @patch('subprocess.run')
# def test_train_model_artifact(mock_subprocess_run, ensemble_manager):
#     """
#     Test the _train_model_artifact method to ensure it correctly generates the shell command and runs it.
#     """
#     model_name = "test_model"
#     run_type = "test"
#     use_saved = True

#     ensemble_manager._train_model_artifact(model_name, run_type, use_saved)

#     mock_subprocess_run.assert_called_once()

# @patch('subprocess.run')
# @patch('builtins.open', new_callable=MagicMock)
# @patch('pickle.load')
# def test_evaluate_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
#     """
#     Test the _evaluate_model_artifact method to ensure it correctly evaluates the model and loads predictions.
#     """
#     model_name = "test_model"
#     run_type = "test"
#     eval_type = "standard"

#     mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

#     preds = ensemble_manager._evaluate_model_artifact(model_name, run_type, eval_type)

#     assert len(preds) > 0
#     mock_subprocess_run.assert_called()
#     mock_open.assert_called()
#     mock_pickle_load.assert_called()

# @patch('subprocess.run')
# @patch('builtins.open', new_callable=MagicMock)
# @patch('pickle.load')
# def test_forecast_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
#     """
#     Test the _forecast_model_artifact method to ensure it correctly forecasts the model and loads predictions.
#     """
#     model_name = "test_model"
#     run_type = "test"

#     mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

#     df = ensemble_manager._forecast_model_artifact(model_name, run_type)

#     assert not df.empty
#     mock_subprocess_run.assert_called()
#     mock_open.assert_called()
#     mock_pickle_load.assert_called()

# @patch.object(EnsembleManager, '_train_model_artifact')
# def test_train_ensemble(mock_train_model_artifact, ensemble_manager):
#     """
#     Test the _train_ensemble method to ensure it correctly trains all models in the ensemble.
#     """
#     ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"]}

#     ensemble_manager._train_ensemble(use_saved=True)

#     assert mock_train_model_artifact.call_count == 2

# @patch.object(EnsembleManager, '_evaluate_model_artifact')
# @patch.object(EnsembleManager, '_get_aggregated_df')
# def test_evaluate_ensemble(mock_get_aggregated_df, mock_evaluate_model_artifact, ensemble_manager):
#     """
#     Test the _evaluate_ensemble method to ensure it correctly evaluates all models in the ensemble and aggregates results.
#     """
#     ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
#     mock_evaluate_model_artifact.return_value = [pd.DataFrame({"A": [1, 2], "B": [3, 4]})]

#     ensemble_manager._evaluate_ensemble(eval_type="standard")

#     assert mock_evaluate_model_artifact.call_count == 2
#     mock_get_aggregated_df.assert_called()

# @patch.object(EnsembleManager, '_forecast_model_artifact')
# @patch.object(EnsembleManager, '_get_aggregated_df')
# def test_forecast_ensemble(mock_get_aggregated_df, mock_forecast_model_artifact, ensemble_manager):
#     """
#     Test the _forecast_ensemble method to ensure it correctly forecasts all models in the ensemble and aggregates results.
#     """
#     ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
#     mock_forecast_model_artifact.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

#     ensemble_manager._forecast_ensemble()

#     assert mock_forecast_model_artifact.call_count == 2
#     mock_get_aggregated_df.assert_called()