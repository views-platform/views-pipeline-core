import pytest
from unittest.mock import patch, MagicMock
from views_pipeline_core.managers.ensemble import EnsemblePathManager, EnsembleManager
from views_pipeline_core.models.check import ensemble_model_check
from views_pipeline_core.managers.model import ModelPathManager, ModelManager

import logging
import pandas as pd
import os
from types import SimpleNamespace
import wandb


class MockArgs:
    def __init__(self, train, evaluate, forecast, saved, run_type, eval_type):
        self.train = train
        self.evaluate = evaluate
        self.forecast = forecast
        self.use_saved = saved
        self.run_type = run_type
        self.eval_type = eval_type


@pytest.fixture
def mock_model_path():
    mock_path = MagicMock()
    mock_path.model_dir = "/path/to/models/test_model"
    return mock_path


@pytest.mark.parametrize(
    "args, expected_command, expected_methods_called",
    [
        (
            MockArgs(
                train=True,  # Simulate training
                evaluate=False,  # Simulate no evaluation
                forecast=False,  # Simulate no forecasting
                saved=True,  # Simulate using used_saved data
                run_type="test",  # Example run type
                eval_type="standard",  # Example eval type
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type",
                "test",
                "--train",
                "--saved",
                "--eval_type",
                "standard",
            ],
            {"train": 1, "evaluate": 0, "forecast": 0},
        ),
        (
            MockArgs(
                train=False,  # Simulate no training
                evaluate=True,  # Simulate evaluation
                forecast=True,  # Simulate forecasting
                saved=False,  # Simulate not using used_saved data
                run_type="forecast",  # Example run type
                eval_type="detailed",  # Example eval type
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type",
                "forecast",
                "--evaluate",
                "--forecast",
                "--eval_type",
                "detailed",
            ],
            {"train": 0, "evaluate": 1, "forecast": 1},
        ),
        (
            MockArgs(
                train=False,  # Simulate no training
                evaluate=False,  # Simulate no evaluation
                forecast=False,  # Simulate no forecasting
                saved=False,  # Simulate not using saved data
                run_type="calibration",  # Example run type
                eval_type="minimal",  # Example eval type
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type",
                "calibration",
                "--eval_type",
                "minimal",
            ],
            {"train": 0, "evaluate": 0, "forecast": 0},
        ),
    ],
)
class TestParametrized:

    @pytest.fixture
    def mock_config(self, args):
        with patch(
            "views_pipeline_core.managers.model.ModelManager._update_single_config"
        ) as mock_update_single_config:
            print(mock_update_single_config(args))
            return {
                "name": "test_model",
                "parameter1": "value1",
            }

    @pytest.fixture
    def mock_ensemble_manager(self, mock_model_path, args):
        manager = EnsembleManager(ensemble_path=mock_model_path)
        manager._project = "test_project"
        manager._entity = "test_entity"
        manager._config_hyperparameters = {}
        manager._config_meta = {}
        manager._train_ensemble = MagicMock()
        manager._evaluate_ensemble = MagicMock()
        manager._forecast_ensemble = MagicMock()
        manager._eval_type = args.eval_type
        with patch(
            "views_pipeline_core.managers.model.ModelManager._update_single_config"
        ) as mock_update_single_config:
            manager.config = mock_update_single_config(args)
        manager.config = {"name": "test_model"}
        return manager

    def test_get_shell_command(
        self, mock_model_path, args, expected_command, expected_methods_called
    ):  # all arguments are necessary
        """
        Test the _get_shell_command method with various input combinations to ensure it generates the correct shell command.
        """
        # Directly use mock_args since it's already a SimpleNamespace object
        command = EnsembleManager._get_shell_command(
            mock_model_path,
            args.run_type,
            args.train,
            args.evaluate,
            args.forecast,
            args.use_saved,
            args.eval_type,
        )
        assert command == expected_command

    # @patch.object(EnsembleManager, '_train_model_artifact')
    # def test_train_ensemble(self, mock_train_model_artifact, args, expected_command):
    #     # Arrange
    #     manager = EnsembleManager(ensemble_path=mock_model_path)
    #     manager._script_paths = {"config_deployment.py": mock_script_path}

    #     manager.config = {
    #         "run_type": "test_run_type",
    #         "models": ["model1", "model2"]
    #     }
    #     use_saved = args.use_saved

    #     # Act
    #     manager._train_ensemble(use_saved)

    #     # Assert
    #     mock_train_model_artifact.assert_any_call("model1", "test_run_type", use_saved)
    #     mock_train_model_artifact.assert_any_call("model2", "test_run_type", use_saved)
    #     assert mock_train_model_artifact.call_count == 2

    @patch("views_pipeline_core.managers.ensemble.EnsembleManager._execute_model_tasks")
    @patch("views_pipeline_core.managers.model.ModelManager._update_single_config")
    @patch("views_pipeline_core.models.check.ensemble_model_check")
    def test_execute_single_run(
        self,
        mock_ensemble_model_check,
        mock_update_single_config,
        mock_execute_model_tasks,
        mock_model_path,
        args,
        expected_command,  # it is necessary to be here
        expected_methods_called,  # it is necessary to be here
    ):

        manager = EnsembleManager(ensemble_path=mock_model_path)
        manager.config = mock_update_single_config(args)
        mock_update_single_config.return_value = {
            "name": "test_model",
            "run_type": args.run_type,
        }

        manager.execute_single_run(args)

        assert manager._project == f"{manager.config['name']}_{args.run_type}"
        assert manager._eval_type == args.eval_type

        if not args.train:
            mock_ensemble_model_check(manager.config)
            mock_ensemble_model_check.assert_called_once_with(manager.config)
        else:
            mock_ensemble_model_check.assert_not_called()

        mock_execute_model_tasks(
            config=manager.config,
            train=args.train,
            eval=args.evaluate,
            forecast=args.forecast,
            use_saved=args.use_saved,
        )
        mock_execute_model_tasks.assert_called_once_with(
            config=manager.config,
            train=args.train,
            eval=args.evaluate,
            forecast=args.forecast,
            use_saved=args.use_saved,
        )

        mock_execute_model_tasks.reset_mock()
        mock_execute_model_tasks.side_effect = Exception("Test exception")

    def test_execute_model_tasks(
        self,
        mock_ensemble_manager,
        mock_config,
        args,
        expected_command,
        expected_methods_called,
    ):

        with patch("wandb.init") as mock_init, patch(
            "wandb.define_metric"
        ) as mock_define_metric, patch("wandb.config") as mock_config:

            mock_config.name = "test_model"
            mock_ensemble_manager._execute_model_tasks(
                config=mock_config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                use_saved=args.use_saved,
            )

            print(expected_methods_called["evaluate"])
            print(mock_ensemble_manager._evaluate_ensemble.call_count)

            assert (
                mock_ensemble_manager._train_ensemble.call_count
                == expected_methods_called["train"]
            )
            assert (
                mock_ensemble_manager._evaluate_ensemble.call_count
                == expected_methods_called["evaluate"]
            )
            assert (
                mock_ensemble_manager._forecast_ensemble.call_count
                == expected_methods_called["forecast"]
            )

            mock_init.assert_called_once()
            mock_define_metric.assert_called()

    # def test_train_model_artifact(
    #         self,
    #         args,
    #         expected_command,
    #         expected_methods_called,
    #         model_name,
    #         run_type,
    #         use_saved,
    #         expected_shell_command
    #     ):
    #         # Mocking necessary components
    #         with patch("views_pipeline_core.managers.ensemble_manager.ModelPathManager") as mock_model_path, \
    #             patch("views_pipeline_core.managers.ensemble_manager.ModelManager") as mock_model_manager, \
    #             patch("views_pipeline_core.managers.ensemble_manager.EnsembleManager._get_shell_command") as mock_get_shell_command, \
    #             patch("subprocess.run") as mock_subprocess_run:

    #             # Mocking the return value of ModelPathManager
    #             mock_model_path_instance = MagicMock()
    #             mock_model_path.return_value = mock_model_path_instance

    #             # Mocking the ModelManager and its configs
    #             mock_model_manager_instance = MagicMock()
    #             mock_model_manager_instance.configs = {"name": "test_model"}
    #             mock_model_manager.return_value = mock_model_manager_instance

    #             # Mocking the shell command returned by _get_shell_command
    #             mock_get_shell_command.return_value = expected_shell_command

    #             # Call the method
    #             manager = MagicMock()  # You can also replace this with your actual manager if needed
    #             manager._train_model_artifact(model_name, run_type, use_saved)

    #             # Check that _get_shell_command was called with the correct parameters
    #             mock_get_shell_command.assert_called_once_with(
    #                 mock_model_path_instance,
    #                 run_type,
    #                 train=True,
    #                 evaluate=False,
    #                 forecast=False,
    #                 use_saved=use_saved
    #             )

    #             # Check that subprocess.run was called with the expected shell command
    #             mock_subprocess_run.assert_called_once_with(expected_shell_command, check=True)

    def test_train_ensemble(
        self, mock_model_path, args, expected_command, expected_methods_called
    ):
        # Create a mock for the ensemble manager
        with patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._train_model_artifact"
        ) as mock_train_model_artifact:
            print("Mocking works:", mock_train_model_artifact)
            manager = EnsembleManager(ensemble_path=mock_model_path)
            manager.config = {
                "run_type": "test_run",
                "models": [
                    "/path/to/models/test_model1",
                    "/path/to/models/test_model2",
                ],
            }
            print(manager.config)
            print("args.use_saved in test:", args.use_saved)

            # Call the method
            manager._train_ensemble(args.use_saved)

            print("Call count:", mock_train_model_artifact.call_count)
            # Check that _train_model_artifact was called the expected number of times
            assert mock_train_model_artifact.call_count == len(manager.config["models"])

            # If there were models, assert that it was called with the expected parameters

            for model_name in manager.config["models"]:
                mock_train_model_artifact.assert_any_call(
                    model_name, "test_run", args.use_saved
                )


@pytest.fixture
def sample_data():
    """
    Fixture to provide common sample data for the aggregation tests.
    """
    df1 = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)])
    )
    df2 = pd.DataFrame(
        {"A": [5, 6], "B": [7, 8]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)])
    )
    return [df1, df2]


def test_get_aggregated_df_mean(sample_data):
    """
    Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using mean.
    """
    df_to_aggregate = sample_data

    result = EnsembleManager._get_aggregated_df(df_to_aggregate, "mean")
    expected = pd.DataFrame(
        {"A": [3.0, 4.0], "B": [5.0, 6.0]},
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]),
    )

    pd.testing.assert_frame_equal(result, expected, check_like=True)


def test_get_aggregated_df_median(sample_data):
    """
    Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using median.
    """
    df_to_aggregate = sample_data

    result = EnsembleManager._get_aggregated_df(df_to_aggregate, "median")
    expected = pd.DataFrame(
        {"A": [3.0, 4.0], "B": [5.0, 6.0]},
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]),
    )

    pd.testing.assert_frame_equal(result, expected, check_like=True)


def test_get_aggregated_df_invalid_aggregation(sample_data):
    """
    Test the _get_aggregated_df method for invalid aggregation method.
    """

    with pytest.raises(
        ValueError, match="Invalid aggregation method: invalid_aggregation"
    ):
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


# @patch('views_pipeline_core.managers.ensemble_manager.EnsembleManager._execute_model_tasks')
# @patch('views_pipeline_core.managers.model_manager.ModelManager._update_single_config')
# @patch('views_pipeline_core.models.check.ensemble_model_check')
# @patch('views_pipeline_core.managers.ensemble_manager.logger')
# def test_execute_single_run(
#     mock_logger,
#     mock_ensemble_model_check,
#     mock_update_single_config,
#     mock_execute_model_tasks,
#     mock_model_path,
#     args,
#     expected_command
# ):
#     """Test the execute_single_run method with multiple parameterized inputs."""

#     # Setup mocks
#     mock_model_path.get_scripts.return_value = {"config_deployment.py": "mock_script_path"}
#     mock_update_single_config.return_value = {"name": "test_model"}
#     mock_load_config = MagicMock(return_value={"name": "test_model"})

#     # Instantiate EnsembleManager
#     manager = EnsembleManager(ensemble_path=mock_model_path)
#     manager._script_paths = {"config_deployment.py": "mock_script_path"}
#     manager._load_config = mock_load_config

#     # Execute the method
#     manager.execute_single_run(args)

#     # Assertions
#     mock_update_single_config.assert_called_once_with(args)
#     assert manager._project == f"test_model_{args.run_type}"
#     assert manager._eval_type == args.eval_type

#     if not args.train:
#         mock_ensemble_model_check.assert_called_once_with({"name": "test_model"})
#     else:
#         mock_ensemble_model_check.assert_not_called()

#     mock_execute_model_tasks.assert_called_once_with(
#         config={"name": "test_model"},
#         train=args.train,
#         eval=args.evaluate,
#         forecast=args.forecast,
#         use_saved=args.saved
#     )
#     mock_logger.error.assert_not_called()

#     # Test exception handling
#     mock_execute_model_tasks.reset_mock()
#     mock_logger.reset_mock()
#     mock_execute_model_tasks.side_effect = Exception("Test error")

#     # Re-run to test exception handling
#     manager.execute_single_run(args)

#     # Verify error logging
#     mock_logger.error.assert_called_once_with("Error during single run execution: Test error")


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
