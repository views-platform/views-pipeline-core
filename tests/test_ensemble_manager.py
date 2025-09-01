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
import os
from pathlib import Path



class MockArgs:
    def __init__(self, train, evaluate, forecast, saved, run_type, eval_type, report, sweep=False, update_viewser=False):
        self.train = train
        self.evaluate = evaluate
        self.forecast = forecast
        self.saved = saved
        self.run_type = run_type
        self.eval_type = eval_type
        self.report = report
        self.sweep = sweep
        self.update_viewser = update_viewser

@pytest.fixture
def mock_model_path():
    mock_path = MagicMock()
    mock_path.model_dir = "/path/to/models/test_model"
    mock_path._validate = False
    mock_path._check_if_dir_exists = MagicMock(return_value=True)
    mock_path.root = Path("/mock/project/root")
    mock_path.data_generated = Path("/path/to/models/test_model/data/generated")
    mock_path.data_raw = Path("/path/to/models/test_model/data/raw")
    return mock_path

@pytest.fixture
def mock_constituent_model_path():
    mock_path = MagicMock()
    mock_path.model_dir = "/path/to/models/constituent_model"
    mock_path._validate = False
    mock_path._check_if_dir_exists = MagicMock(return_value=True)
    mock_path.root = Path("/mock/project/root")
    mock_path.data_generated = Path("/path/to/models/constituent_model/data/generated")
    mock_path.data_raw = Path("/path/to/models/constituent_model/data/raw")
    mock_path.get_latest_model_artifact_path.return_value = Path("/path/to/artifact")
    return mock_path

@pytest.mark.parametrize(
    "args, expected_command, expected_methods_called",
    [
        (
            MockArgs(
                train=True,
                evaluate=False,
                forecast=False,
                saved=False,
                run_type="test",
                eval_type="standard",
                report=False,
                update_viewser=False
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
                train=False,
                evaluate=True,
                forecast=True,
                saved=False,
                run_type="forecast",
                eval_type="detailed",
                report=False,
                update_viewser=False
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
                train=False,
                evaluate=False,
                forecast=True,
                saved=True,
                run_type="calibration",
                eval_type="minimal",
                report=False,
                update_viewser=False
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
    def mock_ensemble_manager(self, mock_model_path, args):
        with patch.object(EnsembleManager, "_ModelManager__load_config"), patch(
            "views_pipeline_core.managers.package.PackageManager"
        ):
            manager = EnsembleManager(
                ensemble_path=mock_model_path, use_prediction_store=False
            )
            manager._project = "test_project"
            manager._eval_type = args.eval_type
            
            # Setup mock config with deployment_status
            manager.config = {
                "name": "test_model",
                "models": ["model_one", "model_two"],
                "run_type": args.run_type,
                "timestamp": "20230101_120000",
                "deployment_status": "test",  # Added to avoid KeyError
                "aggregation": "mean"
            }
        return manager

    def test_get_shell_command(
        self, mock_model_path, args, expected_command, expected_methods_called
    ):
        command = EnsembleManager._get_shell_command(
            mock_model_path,
            args.run_type,
            args.train,
            args.evaluate,
            args.forecast,
            args.saved,
            args.eval_type,
            args.update_viewser
        )
        assert command == expected_command

    def test_execute_single_run(
        self,
        mock_ensemble_manager,
        args,
        expected_command,
        expected_methods_called,
    ):
        with patch(
            "views_pipeline_core.managers.ensemble.logger"
        ) as mock_logger, patch(
            "views_pipeline_core.managers.package.PackageManager"
        ), patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._execute_model_tasks"
        ) as mock_execute_model_tasks, patch(
            "views_pipeline_core.managers.ensemble.validate_ensemble_model" 
        ) as mock_validate_ensemble_model, patch(
            "views_pipeline_core.managers.ensemble.EnsembleManager._update_single_config"
        ) as mock_update_single_config, patch(
            "views_pipeline_core.managers.model.ModelPathManager._check_if_dir_exists", return_value=True
        ), patch(
            "views_pipeline_core.files.utils.read_log_file"
        ) as mock_read_log_file:
            # Always return a config with deployment_status
            mock_update_single_config.return_value = {
                "name": "test_model",
                "models": ["model_one", "model_two"],
                "run_type": args.run_type,
                "timestamp": "20230101_120000",
                "deployment_status": "shadow",
                "aggregation": "mean"
            }
            # Patch read_log_file to always return a valid log dict
            mock_read_log_file.return_value = {
                "Deployment Status": "shadow",
                "Single Model Timestamp": "20230101_120000",
                "Data Generation Timestamp": "20230101_120000",
                "Data Fetch Timestamp": "20230101_120000"
            }

            manager = mock_ensemble_manager
            manager._model_path.target = "ensemble"

            # Test successful execution
            manager.execute_single_run(args)

            # Validate attributes
            assert manager._project == f"{manager.config['name']}_{args.run_type}"
            assert manager._eval_type == args.eval_type

            # Validate validation calls
            if not args.train:
                mock_validate_ensemble_model.assert_called_once_with(manager.config)
            else:
                mock_validate_ensemble_model.assert_not_called()

            # Validate task execution
            mock_execute_model_tasks.assert_called_once_with(
                config=manager.config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                use_saved=args.saved,
                report=args.report,
                update_viewser=args.update_viewser,
            )

            # Test exception handling
            mock_execute_model_tasks.side_effect = Exception("Test exception")
            manager = mock_ensemble_manager

            with pytest.raises(Exception) as exc_info:
                manager.execute_single_run(args)
            assert str(exc_info.value) == "Test exception"

            # Validate error logging
            mock_logger.error.assert_any_call(
                "Error during ensemble execution: Test exception",
                exc_info=True,
            )

    def test_execute_model_tasks(
        self,
        mock_ensemble_manager,
        mock_constituent_model_path,
        args,
        expected_command,
        expected_methods_called,
    ):
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.AlertLevel") as mock_alert_level, \
             patch("views_pipeline_core.managers.ensemble.add_wandb_metrics") as mock_add_wandb_metrics, \
             patch("wandb.config") as mock_wandb_config, \
             patch("views_pipeline_core.wandb.utils.wandb_alert") as mock_wandb_alert, \
             patch("views_pipeline_core.managers.package.PackageManager") as mock_package_manager, \
             patch("views_pipeline_core.managers.ensemble.logger") as mock_logger, \
             patch("views_pipeline_core.managers.ensemble.EnsembleManager._train_ensemble") as mock_train_ensemble, \
             patch("views_pipeline_core.managers.ensemble.EnsembleManager._evaluate_ensemble") as mock_evaluate_ensemble, \
             patch("views_pipeline_core.managers.ensemble.EnsembleManager._forecast_ensemble") as mock_forecast_ensemble, \
             patch("views_pipeline_core.managers.ensemble.handle_ensemble_log_creation") as mock_handle_ensemble_log_creation, \
             patch("views_pipeline_core.managers.ensemble.EnsembleManager._evaluate_prediction_dataframe") as mock_evaluate_prediction_dataframe, \
             patch("views_pipeline_core.managers.ensemble.EnsembleManager._save_predictions") as mock_save_predictions, \
             patch("traceback.format_exc") as mock_format_exc, \
             patch("builtins.open", mock_open()) as mock_file, \
             patch("views_pipeline_core.managers.model.ModelPathManager", return_value=mock_constituent_model_path), \
             patch("views_pipeline_core.files.utils.read_log_file") as mock_read_log_file:

            mock_read_log_file.return_value = {
                "Deployment Status": "shadow",
                "Single Model Timestamp": "20230101_120000",
                "Data Generation Timestamp": "20230101_120000",
                "Data Fetch Timestamp": "20230101_120000"
            }

            manager = mock_ensemble_manager
            manager.config = {
                "name": "test_model",
                "models": ["model_one", "model_two"],
                "run_type": args.run_type,
                "timestamp": "20230101_120000",
                "deployment_status": "shadow",
                "aggregation": "mean"
            }

            # Patch the target attribute for consistency
            manager._model_path.target = "ensemble"

            # Execute model tasks
            manager._execute_model_tasks(
                config=manager.config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                use_saved=args.saved,
                report=args.report,
                update_viewser=args.update_viewser
            )

            # Validate W&B metrics (allow multiple calls)
            assert mock_add_wandb_metrics.call_count >= 1

            # Validate log creation (allow multiple calls)
            if args.evaluate or args.forecast:
                assert mock_handle_ensemble_log_creation.call_count >= 1

            # Validate training
            if expected_methods_called["train"]:
                mock_logger.info.assert_any_call(
                    "Training model test_model..."
                )
                mock_train_ensemble.assert_called_once_with(use_saved=args.saved, update_viewser=args.update_viewser)
            else:
                mock_train_ensemble.assert_not_called()
                # Check for any call with these arguments, ignoring others
                # calls = [
                #     call(
                #         title="Running Ensemble",
                #         text="Ensemble Name: test_model\nConstituent Models: ['model_one', 'model_two']",
                #         level=ANY
                #     )
                #     for call in mock_wandb_alert.mock_calls
                #     if call[0] == "" and call[1] == () and 
                #        call[2]["title"] == "Running Ensemble" and
                #        call[2]["text"] == "Ensemble Name: test_model\nConstituent Models: ['model_one', 'model_two']"
                # ]
                # assert len(calls) > 0, "Expected wandb_alert call not found"

            # Validate evaluation
            if expected_methods_called["evaluate"]:
                mock_logger.info.assert_any_call(
                    "Evaluating model test_model..."
                )
                mock_evaluate_ensemble.assert_called_once_with(manager._eval_type, args.update_viewser)
                mock_handle_ensemble_log_creation.assert_called()
            else:
                mock_evaluate_ensemble.assert_not_called()

                # mock_evaluate_prediction_dataframe.assert_called_once_with(
                #     mock_evaluate_ensemble.return_value, 
                #     ensemble=True
                # )

            # Validate forecasting
            if expected_methods_called["forecast"]:
                mock_logger.info.assert_any_call(
                    "Forecasting model test_model..."
                )
                mock_forecast_ensemble.assert_called_once_with(update_viewser=args.update_viewser)
                mock_handle_ensemble_log_creation.assert_called()
                mock_save_predictions.assert_called_once_with(
                    mock_forecast_ensemble.return_value,
                    manager._model_path.data_generated
                )
            else:
                mock_forecast_ensemble.assert_not_called()
                # Check for any call with these arguments, ignoring others
                # calls = [
                #     call(
                #         title="Running Ensemble",
                #         text="Ensemble Name: test_model\nConstituent Models: ['model_one', 'model_two']",
                #         level=ANY
                #     )
                #     for call in mock_wandb_alert.mock_calls
                #     if call[0] == "" and call[1] == () and 
                #        call[2]["title"] == "Running Ensemble" and
                #        call[2]["text"] == "Ensemble Name: test_model\nConstituent Models: ['model_one', 'model_two']"
                # ]
                # assert len(calls) > 0, "Expected wandb_alert call not found"

            # Validate error handling
            mock_train_ensemble.reset_mock()
            mock_evaluate_ensemble.reset_mock()
            mock_forecast_ensemble.reset_mock()
            mock_logger.reset_mock()

            mock_train_ensemble.side_effect = Exception("Train ensemble failed")
            mock_evaluate_ensemble.side_effect = Exception("Evaluate ensemble failed")
            mock_forecast_ensemble.side_effect = Exception("Forecast ensemble failed")

            with pytest.raises(Exception) as exc_info:
                manager._execute_model_tasks(
                    config=manager.config,
                    train=args.train,
                    eval=args.evaluate,
                    forecast=args.forecast,
                    use_saved=args.saved,
                    report=args.report,
                    update_viewser=args.update_viewser
                )
            
            # Validate appropriate error is raised
            assert any(msg in str(exc_info.value) for msg in [
                "Train ensemble failed",
                "Evaluate ensemble failed",
                "Forecast ensemble failed"
            ])


@pytest.fixture
def sample_data():
    df1 = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4]}, 
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)])
    )
    df2 = pd.DataFrame(
        {"A": [5, 6], "B": [7, 8]}, 
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)])
    )
    return [df1, df2]


def test_get_aggregated_df_mean(sample_data):
    result = EnsembleManager._get_aggregated_df(sample_data, "mean")
    expected = pd.DataFrame(
        {"A": [3.0, 4.0], "B": [5.0, 6.0]},
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]),
    )
    pd.testing.assert_frame_equal(result, expected, check_like=True)


def test_get_aggregated_df_median(sample_data):
    result = EnsembleManager._get_aggregated_df(sample_data, "median")
    expected = pd.DataFrame(
        {"A": [3.0, 4.0], "B": [5.0, 6.0]},
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]),
    )
    pd.testing.assert_frame_equal(result, expected, check_like=True)


def test_get_aggregated_df_invalid_aggregation(sample_data):
    with pytest.raises(
        ValueError, match="Invalid aggregation method: invalid_aggregation"
    ):
        EnsembleManager._get_aggregated_df(sample_data, "invalid_aggregation")