import pytest
import unittest
import pickle
from unittest.mock import patch, MagicMock, ANY, PropertyMock, mock_open, call
from views_pipeline_core.managers.ensemble import EnsembleManager
from views_pipeline_core.managers.model import (
    ModelPathManager,
    ModelManager,
    ForecastingModelManager,
)
import pandas as pd
import wandb
import subprocess


class MockArgs:
    def __init__(self, train, evaluate, forecast, saved, run_type, eval_type, report):
        self.train = train
        self.evaluate = evaluate
        self.forecast = forecast
        self.saved = saved
        self.run_type = run_type
        self.eval_type = eval_type
        self.report = report


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
                saved=False,  # Simulate using used_saved data
                run_type="test",  # Example run type
                eval_type="standard",  # Example eval type
                report=False,
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type",
                "test",
                "--train",
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
                report=False,
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
                forecast=True,  # Simulate forecasting
                saved=True,  # Simulate using saved data
                run_type="calibration",  # Example run type
                eval_type="minimal",  # Example eval type
                report=False,
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type",
                "calibration",
                "--forecast",
                "--saved",
                "--eval_type",
                "minimal",
            ],
            {"train": 0, "evaluate": 0, "forecast": 1},
        ),
    ],
)
class TestParametrized:

    @pytest.fixture
    def mock_config(self, args):
        with patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._update_single_config"
        ) as mock_update_single_config:
            print(mock_update_single_config(args))
            return {"name": "test_model", "parameter1": "value1"}

    @pytest.fixture
    def mock_ensemble_manager(self, mock_model_path, args):
        with patch.object(EnsembleManager, "_ModelManager__load_config"), patch(
            "views_pipeline_core.managers.package.PackageManager"
        ):

            manager = EnsembleManager(
                ensemble_path=mock_model_path, use_prediction_store=False
            )
            # manager.config = mock_update_single_config(args)
            manager._project = "test_project"
            manager._eval_type = args.eval_type
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
            args.saved,
            args.eval_type,
        )
        assert command == expected_command

    def test_execute_single_run(
        self,
        mock_ensemble_manager,
        args,
        expected_command,  # it is necessary to be here
        expected_methods_called,  # it is necessary to be here
    ):
        with patch(
            "views_pipeline_core.managers.ensemble.logger"
        ) as mock_logger, patch(
            "views_pipeline_core.managers.package.PackageManager"
        ), patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._execute_model_tasks"
        ) as mock_execute_model_tasks, patch(
            "views_pipeline_core.managers.model.ForecastingModelManager._update_single_config"
        ), patch(
            "views_pipeline_core.models.check.validate_ensemble_model"
        ) as mock_validate_ensemble_model:

            # Creating EnsembleManager object with the necessary configs
            manager = mock_ensemble_manager

            # Testing the function in the Try block
            manager.execute_single_run(args)

            # Asserting the attributes
            assert manager._project == f"{manager.config['name']}_{args.run_type}"
            assert manager._eval_type == args.eval_type

            # Asserting that validate_ensemble_model was called appropriately
            if not args.train:
                mock_validate_ensemble_model.assert_called_once_with(manager.config)
            else:
                mock_validate_ensemble_model.assert_not_called()

            # Asserting that _execute_model_tasks was called appropriately
            mock_execute_model_tasks.assert_called_once_with(
                config=manager.config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                use_saved=args.saved,
                report=args.report,
            )

            # Testing the function in the Except block
            mock_execute_model_tasks.side_effect = Exception("Test exception")
            manager = mock_ensemble_manager

            # Bypassing exit when exception is raised
            with pytest.raises(Exception) as exc_info:
                manager.execute_single_run(args)
            assert str(exc_info.value) == "Test exception"

            # Asserting that the error message was called appropriately
            mock_logger.error.assert_any_call(
                f"Error during single run execution: {mock_execute_model_tasks.side_effect}",
                exc_info=True,
            )

    def test_execute_model_tasks(
        self,
        mock_config,
        mock_model_path,
        mock_ensemble_manager,
        args,
        expected_command,  # it is necessary to be here
        expected_methods_called,  # it is necessary to be here
    ):

        with patch("wandb.init"), patch("wandb.AlertLevel") as mock_alert_level, patch(
            "views_pipeline_core.managers.ensemble.add_wandb_metrics"
        ) as mock_add_wandb_metrics, \
            patch("views_pipeline_core.managers.package.PackageManager"), \
            patch("views_pipeline_core.managers.ensemble.logger") as mock_logger, \
            patch("views_pipeline_core.managers.ensemble.EnsembleManager._train_ensemble") as mock_train_ensemble, patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._evaluate_ensemble"
        ) as mock_evaluate_ensemble, patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._forecast_ensemble"
        ) as mock_forecast_ensemble, patch(
            "views_pipeline_core.files.utils.handle_ensemble_log_creation"
        ) as mock_handle_ensemble_log_creation, patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._evaluate_prediction_dataframe"
        ) as mock_evaluate_prediction_dataframe, patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._save_predictions"
        ) as mock_save_predictions, patch(
            "traceback.format_exc"
        ) as mock_format_exc:

            manager = mock_ensemble_manager

            print(args.train, args.evaluate, args.forecast)

            manager._execute_model_tasks(
                config=mock_config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                use_saved=args.saved,
                report=args.report,
            )

            mock_add_wandb_metrics.assert_called_once

            if args.train:
                mock_logger.info.assert_any_call(
                    f"Training model {manager.config['name']}..."
                )
                mock_train_ensemble.assert_called_once_with(args.saved)
                # mock_wandb_alert.assert_has_calls(
                #     [
                #         call(
                #             title="Running Ensemble",
                #             text=f"Ensemble Name: {str(manager.config['name'])}\nConstituent Models: {str(manager.config['models'])}",
                #             level=mock_alert_level.INFO,
                #         ),
                #         call(
                #             title=f"Training for {manager._model_path.target} {manager.config['name']} completed successfully.",
                #             text=f"",
                #             level=mock_alert_level.INFO,
                #         ),
                #     ],
                #     any_order=False,
                # )

            if args.evaluate:
                mock_logger.info.assert_any_call(
                    f"Evaluating model {manager.config['name']}..."
                )
                mock_evaluate_ensemble.assert_called_once_with(manager._eval_type)
                mock_handle_ensemble_log_creation.assert_called_once

                mock_evaluate_prediction_dataframe.assert_called_once_with(
                    manager._evaluate_ensemble(manager._eval_type), ensemble=True
                )

            if args.forecast:
                mock_logger.info.assert_any_call(
                    f"Forecasting model {manager.config['name']}..."
                )
                mock_forecast_ensemble.assert_called_once
                # mock_wandb_alert.assert_has_calls(
                #     [
                #         call(
                #             title="Running Ensemble",
                #             text=f"Ensemble Name: {str(manager.config['name'])}\nConstituent Models: {str(manager.config['models'])}",
                #             level=mock_alert_level.INFO,
                #         ),
                #         call(
                #             title=f"Forecasting for ensemble {manager.config['name']} completed successfully.",
                #             level=mock_alert_level.INFO,
                #         ),
                #     ],
                #     any_order=False,
                # )
                mock_handle_ensemble_log_creation.assert_called_once
                mock_save_predictions.assert_called_once_with(
                    manager._forecast_ensemble(), manager._model_path.data_generated
                )

            minutes = 5.222956339518229e-05  # random number
            mock_logger.info.assert_any_call(f"Done. Runtime: {minutes:.3f} minutes.\n")

            # reset mock
            mock_add_wandb_metrics.reset_mock()
            mock_logger.reset_mock()
            mock_train_ensemble.reset_mock()
            mock_evaluate_ensemble.reset_mock()
            mock_forecast_ensemble.reset_mock()
            mock_handle_ensemble_log_creation.reset_mock()
            mock_evaluate_prediction_dataframe
            mock_save_predictions.reset_mock()
            # mock_wandb_alert.reset_mock()

            mock_train_ensemble.side_effect = Exception("Train ensemble failed")
            mock_evaluate_ensemble.side_effect = Exception("Evaluate ensemble failed")
            mock_forecast_ensemble.side_effect = Exception("Forecast ensemble failed")

            manager = mock_ensemble_manager

            with pytest.raises(Exception) as exc_info:
                manager._execute_model_tasks(
                    config=mock_config,
                    train=args.train,
                    eval=args.evaluate,
                    forecast=args.forecast,
                    use_saved=args.saved,
                    report=args.report,
                )
            assert str(exc_info.value) in [
                "Train ensemble failed",
                "Evaluate ensemble failed",
                "Forecast ensemble failed",
            ]

            if args.train:
                mock_logger.error.assert_has_calls(
                    [
                        call(
                            f"{manager._model_path.target.title()} training model: {mock_train_ensemble.side_effect}",
                            exc_info=True,
                        ),
                        call(
                            f"Error during model tasks execution: {mock_train_ensemble.side_effect}",
                            exc_info=True,
                        ),
                    ]
                )
                # mock_wandb_alert.assert_has_calls(
                #     [
                #         call(
                #             title="Running Ensemble",
                #             text=f"Ensemble Name: {str(manager.config['name'])}\nConstituent Models: {str(manager.config['models'])}",
                #             level=mock_alert_level.INFO,
                #         ),
                #         call(
                #             title=f"{manager._model_path.target.title()} Training Error",
                #             text=f"An error occurred during training of {manager._model_path.target} {manager.config['name']}: {mock_format_exc()}",
                #             level=mock_alert_level.ERROR,
                #         ),
                #         call(
                #             title=f"{manager._model_path.target.title()} Task Execution Error",
                #             text=f"An error occurred during the execution of {manager._model_path.target} tasks for {manager.config['name']}: {mock_train_ensemble.side_effect}",
                #             level=mock_alert_level.ERROR,
                #         ),
                #     ]
                # )

            elif args.evaluate:  # elif, since we can use the flags together
                mock_logger.error.assert_has_calls(
                    [
                        call(
                            f"Error evaluating model: {mock_evaluate_ensemble.side_effect}",
                            exc_info=True,
                        ),
                        call(
                            f"Error during model tasks execution: {mock_evaluate_ensemble.side_effect}",
                            exc_info=True,
                        ),
                    ]
                )
                # mock_wandb_alert.assert_has_calls(
                #     [
                #         call(
                #             title="Running Ensemble",
                #             text=f"Ensemble Name: {str(manager.config['name'])}\nConstituent Models: {str(manager.config['models'])}",
                #             level=mock_alert_level.INFO,
                #         ),
                #         call(
                #             title=f"{manager._model_path.target.title()} Evaluation Error",
                #             text=f"An error occurred during evaluation of {manager._model_path.target} {manager.config['name']}: {mock_format_exc()}",
                #             level=mock_alert_level.ERROR,
                #         ),
                #         call(
                #             title=f"{manager._model_path.target.title()} Task Execution Error",
                #             text=f"An error occurred during the execution of {manager._model_path.target} tasks for {manager.config['name']}: {mock_evaluate_ensemble.side_effect}",
                #             level=mock_alert_level.ERROR,
                #         ),
                #     ]
                # )

            elif args.forecast:
                mock_logger.error.assert_has_calls(
                    [
                        call(
                            f"Error forecasting {manager._model_path.target}: {mock_forecast_ensemble.side_effect}",
                            exc_info=True,
                        ),
                        call(
                            f"Error during model tasks execution: {mock_forecast_ensemble.side_effect}",
                            exc_info=True,
                        ),
                    ]
                )
                # mock_wandb_alert.assert_has_calls(
                #     [
                #         call(
                #             title="Running Ensemble",
                #             text=f"Ensemble Name: {str(manager.config['name'])}\nConstituent Models: {str(manager.config['models'])}",
                #             level=mock_alert_level.INFO,
                #         ),
                #         call(
                #             title="Model Forecasting Error",
                #             text=f"An error occurred during forecasting of {manager._model_path.target} {manager.config['name']}: {mock_format_exc()}",
                #             level=mock_alert_level.ERROR,
                #         ),
                #         call(
                #             title=f"{manager._model_path.target.title()} Task Execution Error",
                #             text=f"An error occurred during the execution of {manager._model_path.target} tasks for {manager.config['name']}: {mock_forecast_ensemble.side_effect}",
                #             level=mock_alert_level.ERROR,
                #         ),
                #     ],
                #     any_order=True,
                # )

                # TODO: assert call counts


#     def test_train_ensemble(self, mock_model_path, mock_ensemble_manager, args,
#         expected_command,
#         expected_methods_called):
#         # Create a mock for the ensemble manager
#         with patch("views_pipeline_core.managers.ensemble.EnsembleManager._train_model_artifact") as mock_train_model_artifact:
#             manager = EnsembleManager(ensemble_path=mock_model_path)

#             manager.config = {
#                 "run_type": "test_run",
#                 "models": ["/path/to/models/test_model1", "/path/to/models/test_model2"]
#             }

#             manager._train_ensemble(args.use_saved)

#             print("Call count:", mock_train_model_artifact.call_count)
#             # Check that _train_model_artifact was called the expected number of times
#             assert mock_train_model_artifact.call_count == len(manager.config["models"])

#             # If there were models, assert that it was called with the expected parameters

#             for model_name in manager.config["models"]:
#                     mock_train_model_artifact.assert_any_call(model_name, "test_run", args.use_saved)


#     def test_evaluate_ensemble(self, mock_model_path, args,
#         expected_command,
#         expected_methods_called):
#         with patch("views_pipeline_core.managers.ensemble.EnsembleManager._evaluate_model_artifact") as mock_evaluate_model_artifact, \
#          patch("views_pipeline_core.managers.ensemble.EnsembleManager._get_aggregated_df") as mock_get_aggregated_df, \
#          patch("views_pipeline_core.managers.model.ModelPathManager") as mock_model_path_class, \
#          patch("views_pipeline_core.managers.model.ModelPathManager._get_model_dir") as mock_get_model_dir, \
#          patch("views_pipeline_core.managers.model.ModelPathManager._build_absolute_directory") as mock_build_absolute_directory:


#             mock_model_path_instance = mock_model_path_class.return_value

#             mock_model_path_instance._initialize_directories()


#             mock_evaluate_model_artifact.side_effect = [
#                 [{"prediction": 1}, {"prediction": 2}],
#                 [{"prediction": 3}, {"prediction": 4}]
#             ]
#             mock_get_aggregated_df.side_effect = [
#                 {"ensemble_prediction": 1.5},
#                 {"ensemble_prediction": 3.0}
#             ]


#             manager = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager.config = {
#                 "run_type": "test_run",
#                 "models": ["test_model", "test_model"],
#                 "name": "test_ensemble",
#                 "deployment_status": "test_status",
#                 "aggregation": "mean",
#             }

#             manager._evaluate_ensemble(args.eval_type)

#             assert mock_evaluate_model_artifact.call_count == len(manager.config["models"])
#             mock_get_aggregated_df.assert_called()


#             # This is just not working:
#             # mock_create_log_file.assert_called_once_with(
#             #     Path("/mock/path/generated"),
#             #     manager.config,
#             #     ANY,  # Timestamp
#             #     ANY,  # Timestamp
#             #     ANY,  # Data fetch timestamp
#             #     model_type="ensemble",
#             #     models=manager.config["models"]
#             # )


#     def test_forecast_ensemble(self, mock_model_path, args,
#         expected_command,
#         expected_methods_called):
#         # Mock all required methods and classes
#         with patch("views_pipeline_core.managers.ensemble.EnsembleManager._forecast_model_artifact") as mock_forecast_model_artifact, \
#             patch("views_pipeline_core.managers.ensemble.EnsembleManager._get_aggregated_df") as mock_get_aggregated_df, \
#             patch("views_pipeline_core.managers.model.ModelPathManager") as mock_model_path_class, \
#             patch("views_pipeline_core.managers.model.ModelPathManager._get_model_dir") as mock_get_model_dir, \
#             patch("views_pipeline_core.managers.model.ModelPathManager._build_absolute_directory") as mock_build_absolute_directory:

#             mock_model_path_instance = mock_model_path.return_value
#             mock_model_path_instance._initialize_directories()

#             mock_forecast_model_artifact.side_effect = [
#                 {"model_name": "test_model", "prediction": 1},
#                 {"model_name": "test_model", "prediction": 2}
#             ]

#             mock_get_aggregated_df.return_value = {"ensemble_prediction": 1.5}

#             manager = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager.config = {
#                 "run_type": "test_run",
#                 "models": ["test_model", "test_model"],
#                 "name": "test_ensemble",
#                 "deployment_status": "test_status",
#                 "aggregation": "mean",
#             }

#             manager._forecast_ensemble()

#             assert mock_forecast_model_artifact.call_count == len(manager.config["models"])
#             assert mock_get_aggregated_df.call_count == 1

#             # This is not working for the same reason
#             # mock_create_log_file.assert_called_once_with(
#             #     Path("/mock/path/generated"),
#             #     manager.config,
#             #     ANY,  # model_timestamp
#             #     ANY,  # data_generation_timestamp
#             #     data_fetch_timestamp=None,
#             #     model_type="ensemble",
#             #     models=manager.config["models"]
#             # )


#     def test_train_model_artifact(self, mock_model_path, args,
#         expected_command,
#         expected_methods_called):
#         # Mocking required methods and classes
#         with patch("views_pipeline_core.managers.ensemble.ModelPathManager") as mock_model_path_class, \
#             patch("views_pipeline_core.managers.ensemble.ModelManager") as mock_model_manager_class, \
#             patch("views_pipeline_core.managers.ensemble.subprocess.run") as mock_subprocess_run, \
#             patch("views_pipeline_core.managers.ensemble.logger") as mock_logger:

#             mock_model_path_instance = mock_model_path_class.return_value

#             # Use PropertyMock to mock model_dir property
#             type(mock_model_path_instance).model_dir = PropertyMock(return_value="/mock/path/to/model")


#             # Mock the ModelManager instance and its configs
#             mock_model_manager_instance = MagicMock()
#             mock_model_manager_class.return_value = mock_model_manager_instance
#             mock_model_manager_instance.configs = {"model_name": "test_model", "run_type": "test_run"}

#             # Mock subprocess.run to simulate successful shell command execution
#             mock_subprocess_run.return_value = None  # Simulate success (no exception thrown)

#             # Instantiate the manager and set up the config
#             manager = EnsembleManager(ensemble_path=mock_model_path)
#             manager.config = {
#                 "run_type": "test_run",
#                 "models": ["test_model"],
#                 "name": "test_ensemble",
#                 "deployment_status": "test_status",
#                 "aggregation": "mean",
#             }

#             # Call the method under test
#             manager._train_model_artifact("test_model", "test_run", use_saved=args.use_saved)

#             # Assert that subprocess.run is called once
#             mock_subprocess_run.assert_called_once_with(
#                 ANY,  # Command should be flexible, so we use ANY
#                 check=True
#             )

#             # Assert that the logger's info method was called
#             mock_logger.info.assert_called_with("Training single model test_model...")

#             # Assert that the correct shell command was generated
#             shell_command = EnsembleManager._get_shell_command(
#                 mock_model_path_instance,
#                 "test_run",
#                 train=True,
#                 evaluate=False,
#                 forecast=False,
#                 use_saved=args.use_saved
#             )

#             mock_subprocess_run.assert_called_once_with(shell_command, check=True)

#             mock_logger.info.assert_called_with("Training single model test_model...")

#             # If an exception is thrown during subprocess.run, assert logger error
#             mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, 'command')
#             mock_exception = subprocess.CalledProcessError(1, 'command')
#             manager._train_model_artifact("test_model", "test_run", use_saved=False)
#             expected_error_message = "Error during shell command execution for model test_model: " + str(mock_exception)
#             mock_logger.error.assert_called_with(expected_error_message)


#     def test_evaluate_model_artifact(self, args, expected_command, expected_methods_called):

#         with patch("views_pipeline_core.managers.ensemble.ModelPathManager") as mock_model_path_manager, \
#             patch("subprocess.run") as mock_subprocess_run, \
#             patch("views_pipeline_core.managers.ensemble.logger") as mock_logger, \
#             patch("views_pipeline_core.managers.ensemble.read_log_file") as mock_read_log_file, \
#             patch("views_pipeline_core.managers.ensemble.create_log_file") as mock_create_log_file, \
#             patch("views_forecasts.extensions.ForecastAccessor.read_store") as mock_read_store, \
#             patch("views_pipeline_core.managers.ensemble.ModelManager._resolve_evaluation_sequence_number") as mock_resolve, \
#             patch("views_pipeline_core.managers.package.PackageManager") as mock_PackageManager, \
#             patch.object(ModelManager, '_ModelManager__load_config') as mock_load_config:

#             mock_resolve.return_value = 5

#             mock_read_store.return_value = pd.DataFrame({"a": [1, 2, 3]})


#             # Mock the ModelPath instance and its attributes
#             mock_model_path_instance = mock_model_path_manager.return_value
#             mock_artifact_path = MagicMock()
#             mock_artifact_path.stem = "predictions_test_run_202401011200000"
#             mock_model_path_instance.get_latest_model_artifact_path.return_value = mock_artifact_path

#             # mock_dataframe_format = ".parquet"
#             # mock_pipeline_config.dataframe_format = mock_dataframe_format

#             #mock_model_path_instance.data_raw = "/mock/path/raw"
#             mock_model_path_instance.data_generated = "/mock/path/generated"


#             # Instantiate the manager and set up the config
#             manager = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager._evaluate_model_artifact("test_model", "test_run", eval_type="standard")
#             mock_logger.info.assert_any_call("Evaluating single model test_model...")


#             for sequence_number in range(mock_resolve.return_value):
#                 mock_logger.info.assert_any_call(f"Loading existing prediction test_model_{mock_artifact_path.stem}_{sequence_number:02} from prediction store")


#             mock_read_store.side_effect = [item for _ in range(mock_resolve.return_value) for item in [
#                 Exception("Test exception"),
#                 pd.DataFrame({"a": [1, 2, 3]}),
#             ]]

#             manager_side = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager_side._evaluate_model_artifact("test_model", "test_run", eval_type="standard")

#             print("here")
#             # Generate the expected shell command
#             # shell_command = EnsembleManager._get_shell_command(
#             #     mock_model_path_instance,
#             #     "test_run",
#             #     train=False,
#             #     evaluate=True,
#             #     forecast=False,
#             #     use_saved=True,
#             #     eval_type="standard"
#             # )
#             # mock_subprocess_run.assert_called_once_with(
#             #     shell_command,
#             #     check=True
#             # )
#             mock_logger.info.assert_any_call("No existing test_run predictions found. Generating new test_run predictions...")
#             print("after first side")
#             print("mock_subprocess_run.call_args_list",mock_subprocess_run.call_args_list)
#             mock_read_store.side_effect = [item for _ in range(mock_resolve.return_value) for item in [
#                 Exception("Test exception"),
#                 pd.DataFrame({"a": [1, 2, 3]}),
#             ]]

#             mock_subprocess_run.side_effect = [Exception("Subprocess failed") for _ in range(mock_resolve.return_value)]

#             manager_side2 = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager_side2._evaluate_model_artifact("test_model", "test_run", eval_type="standard")
#             print("mock_logger.info.call_args_list",mock_logger.info.call_args_list)
#             print("mock_logger.error.call_args_list",mock_logger.error.call_args_list)
#             mock_logger.error.assert_any_call("Error during shell command execution for model test_model: Subprocess failed")


#             assert mock_create_log_file.call_count==10
#             assert mock_logger.error.call_count ==5
#             assert mock_logger.info.call_count ==18
#             assert mock_read_log_file.call_count==10


#     def test_forecast_model_artifact(self, mock_model_path, args, expected_command, expected_methods_called):

#         with patch("views_pipeline_core.managers.ensemble.ModelPathManager") as mock_model_path_manager, \
#             patch("subprocess.run") as mock_subprocess_run, \
#             patch("views_pipeline_core.managers.ensemble.logger") as mock_logger, \
#             patch("views_pipeline_core.managers.ensemble.read_log_file") as mock_read_log_file, \
#             patch("views_pipeline_core.managers.ensemble.create_log_file") as mock_create_log_file, \
#             patch("views_forecasts.extensions.ForecastAccessor.read_store") as mock_read_store, \
#             patch("views_pipeline_core.managers.package.PackageManager") as mock_PackageManager, \
#             patch.object(ModelManager, '_ModelManager__load_config') as mock_load_config:

#             # mock get_latest_model_artifact_path
#             mock_model_path_instance = mock_model_path_manager.return_value
#             mock_artifact_path = MagicMock()
#             mock_artifact_path.stem = "predictions_test_run_202401011200000"
#             mock_model_path_instance.get_latest_model_artifact_path.return_value = mock_artifact_path

#             # Try block
#             mock_read_store.return_value = pd.DataFrame({"a": [1, 2, 3]})
#             manager = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager._forecast_model_artifact("test_model", "test_run")

#             mock_logger.info.assert_any_call("Forecasting single model test_model...")
#             expected_name = (f"test_model_{mock_artifact_path.stem}")
#             mock_logger.info.assert_any_call(f"Loading existing prediction {expected_name} from prediction store")

#             # Except block for read_store, Try block for subprocess
#             mock_read_store.side_effect = [
#                 Exception("Test exception"),
#                 pd.DataFrame({"a": [1, 2, 3]}),
#             ]
#             manager_side = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager_side._forecast_model_artifact("test_model", "test_run")

#             mock_logger.info.assert_any_call("No existing test_run predictions found. Generating new test_run predictions...")

#             # Except block for read_store, Except block for subprocess
#             mock_read_store.side_effect = [
#                 Exception("Test exception"),
#                 pd.DataFrame({"a": [1, 2, 3]}),
#             ]
#             mock_subprocess_run.side_effect = Exception("Subprocess failed"),

#             manager_side2 = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager_side2._forecast_model_artifact("test_model", "test_run")

#             mock_logger.error.assert_any_call("Error during shell command execution for model test_model: Subprocess failed")

#             assert mock_create_log_file.call_count==2
#             assert mock_logger.error.call_count ==1
#             assert mock_logger.info.call_count ==6
#             assert mock_read_log_file.call_count==2


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
