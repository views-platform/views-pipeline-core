import pytest
from unittest.mock import patch, MagicMock
from views_pipeline_core.managers.ensemble_manager import EnsembleManager
from views_pipeline_core.managers.path_manager import EnsemblePath, ModelPath
from views_pipeline_core.managers.model_manager import ModelManager
import pandas as pd
import os

@pytest.fixture
def ensemble_manager():
    ensemble_path = EnsemblePath("test_ensemble", validate=False)
    if not os.path.exists(ensemble_path.model_dir):
        os.makedirs(ensemble_path.model_dir)
    return EnsembleManager(ensemble_path)

def test_get_shell_command():
    """
    Test the _get_shell_command method to ensure it generates the correct shell command.
    """
    model_path = MagicMock()
    model_path.model_dir = "/path/to/models/test_model"
    run_type = "test"
    train = True
    evaluate = False
    forecast = False
    use_saved = True
    eval_type = "standard"

    expected_command = [
        "/path/to/models/test_model/run.sh",
        "--run_type", "test",
        "--train",
        "--saved",
        "--eval_type", "standard"
    ]

    command = EnsembleManager._get_shell_command(model_path, run_type, train, evaluate, forecast, use_saved, eval_type)
    assert command == expected_command

def test_get_aggregated_df_mean():
    """
    Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using mean.
    """
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    df_to_aggregate = [df1, df2]

    result = EnsembleManager._get_aggregated_df(df_to_aggregate, "mean")
    expected = pd.DataFrame({"A": [3, 4], "B": [5, 6]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))

    pd.testing.assert_frame_equal(result, expected, check_like=True)

def test_get_aggregated_df_median():
    """
    Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using median.
    """
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    df_to_aggregate = [df1, df2]

    result = EnsembleManager._get_aggregated_df(df_to_aggregate, "median")
    expected = pd.DataFrame({"A": [3, 4], "B": [5, 6]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))

    pd.testing.assert_frame_equal(result, expected, check_like=True)

# @patch.object(EnsembleManager, '_execute_model_tasks')
# @patch.object(ModelManager, '_update_single_config')
# def test_execute_single_run(mock_update_single_config, mock_execute_model_tasks, ensemble_manager):
#     """
#     Test the execute_single_run method to ensure it correctly executes a single run.
#     """
#     args = MagicMock()
#     args.run_type = "test"
#     args.eval_type = "standard"
#     mock_update_single_config.return_value = {"name": "test_model"}

#     ensemble_manager.execute_single_run(args)

#     mock_update_single_config.assert_called_once_with(args)
#     mock_execute_model_tasks.assert_called_once()

@patch.object(EnsembleManager, '_execute_model_tasks')
@patch.object(ModelManager, '_update_single_config')
def test_execute_single_run(mock_update_single_config, mock_execute_model_tasks, ensemble_manager):
    """
    Test the execute_single_run method to ensure it correctly updates the config and executes model tasks.
    """
    args = MagicMock()
    args.run_type = "test"
    args.eval_type = "standard"
    args.train = True
    args.evaluate = False
    args.forecast = False
    args.saved = True

    mock_update_single_config.return_value = {"name": "test_model"}

    ensemble_manager.execute_single_run(args)

    mock_update_single_config.assert_called_once_with(args)
    mock_execute_model_tasks.assert_called_once_with(
        config={"name": "test_model", "run_type": "test"},
        train=True,
        eval=False,
        forecast=False,
        use_saved=True
    )

@patch('subprocess.run')
def test_train_model_artifact(mock_subprocess_run, ensemble_manager):
    """
    Test the _train_model_artifact method to ensure it correctly generates the shell command and runs it.
    """
    model_name = "test_model"
    run_type = "test"
    use_saved = True

    ensemble_manager._train_model_artifact(model_name, run_type, use_saved)

    mock_subprocess_run.assert_called_once()

@patch('subprocess.run')
@patch('builtins.open', new_callable=MagicMock)
@patch('pickle.load')
def test_evaluate_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
    """
    Test the _evaluate_model_artifact method to ensure it correctly evaluates the model and loads predictions.
    """
    model_name = "test_model"
    run_type = "test"
    eval_type = "standard"

    mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    preds = ensemble_manager._evaluate_model_artifact(model_name, run_type, eval_type)

    assert len(preds) > 0
    mock_subprocess_run.assert_called()
    mock_open.assert_called()
    mock_pickle_load.assert_called()

@patch('subprocess.run')
@patch('builtins.open', new_callable=MagicMock)
@patch('pickle.load')
def test_forecast_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
    """
    Test the _forecast_model_artifact method to ensure it correctly forecasts the model and loads predictions.
    """
    model_name = "test_model"
    run_type = "test"

    mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    df = ensemble_manager._forecast_model_artifact(model_name, run_type)

    assert not df.empty
    mock_subprocess_run.assert_called()
    mock_open.assert_called()
    mock_pickle_load.assert_called()

@patch.object(EnsembleManager, '_train_model_artifact')
def test_train_ensemble(mock_train_model_artifact, ensemble_manager):
    """
    Test the _train_ensemble method to ensure it correctly trains all models in the ensemble.
    """
    ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"]}

    ensemble_manager._train_ensemble(use_saved=True)

    assert mock_train_model_artifact.call_count == 2

@patch.object(EnsembleManager, '_evaluate_model_artifact')
@patch.object(EnsembleManager, '_get_aggregated_df')
def test_evaluate_ensemble(mock_get_aggregated_df, mock_evaluate_model_artifact, ensemble_manager):
    """
    Test the _evaluate_ensemble method to ensure it correctly evaluates all models in the ensemble and aggregates results.
    """
    ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
    mock_evaluate_model_artifact.return_value = [pd.DataFrame({"A": [1, 2], "B": [3, 4]})]

    ensemble_manager._evaluate_ensemble(eval_type="standard")

    assert mock_evaluate_model_artifact.call_count == 2
    mock_get_aggregated_df.assert_called()

@patch.object(EnsembleManager, '_forecast_model_artifact')
@patch.object(EnsembleManager, '_get_aggregated_df')
def test_forecast_ensemble(mock_get_aggregated_df, mock_forecast_model_artifact, ensemble_manager):
    """
    Test the _forecast_ensemble method to ensure it correctly forecasts all models in the ensemble and aggregates results.
    """
    ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
    mock_forecast_model_artifact.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    ensemble_manager._forecast_ensemble()

    assert mock_forecast_model_artifact.call_count == 2
    mock_get_aggregated_df.assert_called()
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    df_to_aggregate = [df1, df2]

    result = EnsembleManager._get_aggregated_df(df_to_aggregate, "median")
    expected = pd.DataFrame({"A": [3, 4], "B": [5, 6]})

    pd.testing.assert_frame_equal(result, expected)

@patch.object(EnsembleManager, '_execute_model_tasks')
@patch.object(ModelManager, '_update_single_config')
def test_execute_single_run(mock_update_single_config, mock_execute_model_tasks, ensemble_manager):
    """
    Test the execute_single_run method to ensure it correctly updates the config and executes model tasks.
    """
    args = MagicMock()
    args.run_type = "test"
    args.eval_type = "standard"
    args.train = True
    args.evaluate = False
    args.forecast = False
    args.saved = True

    mock_update_single_config.return_value = {"name": "test_model"}

    ensemble_manager.execute_single_run(args)

    mock_update_single_config.assert_called_once_with(args)
    mock_execute_model_tasks.assert_called_once_with(
        config={"name": "test_model", "run_type": "test"},
        train=True,
        eval=False,
        forecast=False,
        use_saved=True
    )

@patch('subprocess.run')
def test_train_model_artifact(mock_subprocess_run, ensemble_manager):
    """
    Test the _train_model_artifact method to ensure it correctly generates the shell command and runs it.
    """
    model_name = "test_model"
    run_type = "test"
    use_saved = True

    ensemble_manager._train_model_artifact(model_name, run_type, use_saved)

    mock_subprocess_run.assert_called_once()

@patch('subprocess.run')
@patch('builtins.open', new_callable=MagicMock)
@patch('pickle.load')
def test_evaluate_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
    """
    Test the _evaluate_model_artifact method to ensure it correctly evaluates the model and loads predictions.
    """
    model_name = "test_model"
    run_type = "test"
    eval_type = "standard"

    mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    preds = ensemble_manager._evaluate_model_artifact(model_name, run_type, eval_type)

    assert len(preds) > 0
    mock_subprocess_run.assert_called()
    mock_open.assert_called()
    mock_pickle_load.assert_called()

@patch('subprocess.run')
@patch('builtins.open', new_callable=MagicMock)
@patch('pickle.load')
def test_forecast_model_artifact(mock_pickle_load, mock_open, mock_subprocess_run, ensemble_manager):
    """
    Test the _forecast_model_artifact method to ensure it correctly forecasts the model and loads predictions.
    """
    model_name = "test_model"
    run_type = "test"

    mock_pickle_load.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    df = ensemble_manager._forecast_model_artifact(model_name, run_type)

    assert not df.empty
    mock_subprocess_run.assert_called()
    mock_open.assert_called()
    mock_pickle_load.assert_called()

@patch.object(EnsembleManager, '_train_model_artifact')
def test_train_ensemble(mock_train_model_artifact, ensemble_manager):
    """
    Test the _train_ensemble method to ensure it correctly trains all models in the ensemble.
    """
    ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"]}

    ensemble_manager._train_ensemble(use_saved=True)

    assert mock_train_model_artifact.call_count == 2

@patch.object(EnsembleManager, '_evaluate_model_artifact')
@patch.object(EnsembleManager, '_get_aggregated_df')
def test_evaluate_ensemble(mock_get_aggregated_df, mock_evaluate_model_artifact, ensemble_manager):
    """
    Test the _evaluate_ensemble method to ensure it correctly evaluates all models in the ensemble and aggregates results.
    """
    ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
    mock_evaluate_model_artifact.return_value = [pd.DataFrame({"A": [1, 2], "B": [3, 4]})]

    ensemble_manager._evaluate_ensemble(eval_type="standard")

    assert mock_evaluate_model_artifact.call_count == 2
    mock_get_aggregated_df.assert_called()

@patch.object(EnsembleManager, '_forecast_model_artifact')
@patch.object(EnsembleManager, '_get_aggregated_df')
def test_forecast_ensemble(mock_get_aggregated_df, mock_forecast_model_artifact, ensemble_manager):
    """
    Test the _forecast_ensemble method to ensure it correctly forecasts all models in the ensemble and aggregates results.
    """
    ensemble_manager.config = {"run_type": "test", "models": ["model1", "model2"], "aggregation": "mean"}
    mock_forecast_model_artifact.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    ensemble_manager._forecast_ensemble()

    assert mock_forecast_model_artifact.call_count == 2
    mock_get_aggregated_df.assert_called()