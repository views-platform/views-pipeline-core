"""
Failure mode and pipeline ordering tests for ForecastingModelManager.

C-19: 5 of 10 CIC failure modes were untested. These tests cover:
  - DataFetchException propagation
  - ModelTrainingException propagation
  - ModelEvaluationException propagation
  - ModelForecastingException propagation
  - Wrong sequence count in evaluation

C-20: No pipeline ordering safety. These tests verify:
  - Data fetch is called before model tasks
  - Stage execution order matches args flags
  - Missing data loader produces a clear failure
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

sys.modules["wandb"] = MagicMock()
sys.modules["views_evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation"] = MagicMock()
sys.modules["art"] = MagicMock()

from views_pipeline_core.managers.model.model import ForecastingModelManager  # noqa: E402
from views_pipeline_core.cli.args import ForecastingModelArgs  # noqa: E402
from views_pipeline_core.exceptions import (  # noqa: E402
    DataFetchException,
    ModelTrainingException,
    ModelEvaluationException,
    ModelForecastingException,
)


# ── Factory ──────────────────────────────────────────────────────────────────


def _make_manager():
    """Create a ForecastingModelManager with mocked internals."""
    mock_path = MagicMock()
    mock_path.model_name = "test_model"
    mock_path.target = "model"
    mock_path.root = Path("/fake")
    mock_path.data_generated = Path("/fake/generated")

    with patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager"
        "._ModelManager__load_config",
        return_value={},
    ):
        with patch(
            "views_pipeline_core.modules.logging.LoggingModule.get_logger",
            return_value=MagicMock(),
        ):
            with patch(
                "views_pipeline_core.managers.ConfigurationManager",
                return_value=MagicMock(),
            ):
                with patch(
                    "views_pipeline_core.managers.model.model.ModelManager"
                    "._ModelManager__ascii_splash"
                ):
                    mgr = ForecastingModelManager(mock_path)

    mgr._args = MagicMock()
    mgr._args.run_type = "calibration"
    mgr._args.train = False
    mgr._args.evaluate = False
    mgr._args.forecast = False
    mgr._args.report = False
    mgr._wandb_module = MagicMock()
    mgr._wandb_notifications = False
    mgr._sweep = False
    mgr._partition_dict = {
        "calibration": {"train": (121, 444), "test": (445, 492)},
    }
    mgr._eval_type = "calibration"
    mgr._save_evaluations = MagicMock()
    mgr._generate_evaluation_table = MagicMock(return_value="")
    mgr._project = "test_model_calibration"
    mgr._model_path = mock_path
    return mgr


# ============================================================================
# C-19: Failure Mode Tests
# ============================================================================


class TestTrainingExceptionPropagation:
    """ModelTrainingException must propagate when _train_model_artifact raises."""

    def test_training_failure_raises_model_training_exception(self):
        mgr = _make_manager()
        mgr._train_model_artifact = MagicMock(
            side_effect=RuntimeError("training crashed")
        )

        with pytest.raises(ModelTrainingException, match="Training failed"):
            mgr._execute_model_training()


class TestEvaluationExceptionPropagation:
    """ModelEvaluationException must propagate when _evaluate_model_artifact raises."""

    def test_evaluation_failure_raises_model_evaluation_exception(self):
        mgr = _make_manager()
        mgr._evaluate_model_artifact = MagicMock(
            side_effect=RuntimeError("evaluation crashed")
        )
        mgr.configs = {
            "steps": list(range(1, 37)),
            "prediction_format": "dataframe",
            "name": "test",
            "sweep": False,
        }

        with pytest.raises(ModelEvaluationException, match="Evaluation failed"):
            mgr._execute_model_evaluation()


class TestForecastingExceptionPropagation:
    """ModelForecastingException must propagate when _forecast_model_artifact raises."""

    def test_forecasting_failure_raises_model_forecasting_exception(self):
        mgr = _make_manager()
        mgr._forecast_model_artifact = MagicMock(
            side_effect=RuntimeError("forecasting crashed")
        )
        mgr.configs = {
            "prediction_format": "dataframe",
            "name": "test",
            "sweep": False,
        }

        with pytest.raises(ModelForecastingException, match="Forecasting failed"):
            mgr._execute_model_forecasting()


class TestWrongSequenceCount:
    """_assert_predictions_in_step_window must reject wrong sequence count."""

    def test_wrong_count_raises_value_error(self):
        mgr = _make_manager()
        mgr.configs = {
            "steps": list(range(1, 37)),
            "prediction_format": "dataframe",
        }

        # 5 DataFrames when 13 expected (MAX_SHIFT_COUNT=12, so 12+1=13)
        idx = pd.MultiIndex.from_tuples(
            [(445, 1)], names=["month_id", "priogrid_gid"]
        )
        predictions = [pd.DataFrame({"pred_x": [0.1]}, index=idx)] * 5

        with pytest.raises(ValueError, match="13"):
            mgr._assert_predictions_in_step_window(predictions)


class TestDataFetchExceptionPropagation:
    """DataFetchException must propagate when data fetching fails."""

    def test_data_fetch_failure_raises_data_fetch_exception(self):
        mgr = _make_manager()
        mgr._project = "test_model_calibration"
        mgr.configs = {"level": "pgm"}
        # Mock the data_loader.get_data to raise — the wrapper in
        # _execute_data_fetching should catch it and re-raise as DataFetchException.
        mgr._data_loader = MagicMock()
        mgr._data_loader.get_data.side_effect = ConnectionError("connection refused")

        with pytest.raises(DataFetchException, match="Data fetching failed"):
            mgr._execute_data_fetching()


# ============================================================================
# C-20: Pipeline Ordering Safety Tests
# ============================================================================


class TestPipelineOrdering:
    """Verify pipeline stage execution order guarantees."""

    def test_execute_single_run_calls_data_fetch_before_tasks(self):
        """Data fetching must complete before model tasks begin."""
        mgr = _make_manager()
        call_order = []

        args = ForecastingModelArgs(run_type="calibration", train=True)

        with patch.object(
            ForecastingModelManager,
            "_execute_data_fetching",
            side_effect=lambda: call_order.append("data_fetch"),
        ):
            with patch.object(
                ForecastingModelManager,
                "_execute_model_tasks",
                side_effect=lambda: call_order.append("model_tasks"),
            ):
                with patch.object(
                    ForecastingModelManager,
                    "_assert_partition_config_accessible",
                ):
                    with patch(
                        "views_pipeline_core.managers.model.model.CoreConfigSniffer"
                    ):
                        mgr.configs = {"name": "test"}
                        mgr.execute_single_run(args)

        assert call_order == ["data_fetch", "model_tasks"], (
            f"Expected data_fetch before model_tasks, got {call_order}"
        )

    def test_task_order_matches_args_flags(self):
        """When all flags are set, order must be: train → evaluate → forecast."""
        mgr = _make_manager()
        mgr._args.train = True
        mgr._args.evaluate = True
        mgr._args.forecast = True
        mgr._args.report = False

        call_order = []

        with patch.object(
            ForecastingModelManager,
            "_execute_model_training",
            side_effect=lambda: call_order.append("train"),
        ):
            with patch.object(
                ForecastingModelManager,
                "_execute_model_evaluation",
                side_effect=lambda: call_order.append("evaluate"),
            ):
                with patch.object(
                    ForecastingModelManager,
                    "_execute_model_forecasting",
                    side_effect=lambda: call_order.append("forecast"),
                ):
                    mgr._execute_model_tasks()

        assert call_order == ["train", "evaluate", "forecast"], (
            f"Expected train→evaluate→forecast, got {call_order}"
        )

    def test_evaluation_without_data_loader_fails(self):
        """Calling _execute_model_evaluation without prior data fetch must fail."""
        mgr = _make_manager()
        mgr.configs = {
            "steps": list(range(1, 37)),
            "prediction_format": "dataframe",
            "name": "test",
            "sweep": False,
        }
        # _evaluate_model_artifact returns predictions that need data_loader context
        mgr._evaluate_model_artifact = MagicMock(return_value=[pd.DataFrame()])

        # _data_loader is not set (no _execute_data_fetching was called).
        # This should fail — either AttributeError or a wrapped exception.
        with pytest.raises((AttributeError, ModelEvaluationException)):
            mgr._execute_model_evaluation()
