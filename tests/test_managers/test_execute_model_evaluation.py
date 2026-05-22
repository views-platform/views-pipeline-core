"""
Characterization tests for ForecastingModelManager._execute_model_evaluation().

These tests capture the current behavior of the most complex method in the
god class (226 LOC) BEFORE further stage extractions (E3-E5).  They serve as
safety nets per Feathers' "Working Effectively with Legacy Code" — if any
extraction changes the observable behavior, these tests will fail.

Paths characterized:
  1. DF path — legacy DataFrame format (validate + save + evaluate metrics)
  2. PF streaming path — origin sink, staging, mmap reload
  3. skip_evaluation_metrics — PF path with metrics skipped
  4. No metrics in config — _has_evaluation_metrics() returns False
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Global mocks for external dependencies
mock_wandb = MagicMock()
mock_wandb.summary._as_dict.return_value = {}
mock_wandb.config = {}

sys.modules["wandb"] = mock_wandb
sys.modules["views_evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation.evaluation_frame"] = MagicMock()
sys.modules["art"] = MagicMock()

from views_pipeline_core.managers.model.model import ForecastingModelManager  # noqa: E402


# ── Shared helpers ──────────────────────────────────────────────────────────

MAX_SHIFT_COUNT = 12
N_SEQUENCES = MAX_SHIFT_COUNT + 1  # 13 — contract requirement


def _make_manager():
    """Build a ForecastingModelManager with all internals mocked."""
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    mock_path_manager.data_generated = Path("/tmp/data_generated")
    mock_path_manager.root = Path("/tmp")
    mock_path_manager.target = "model"
    mock_path_manager._target = "model"
    mock_path_manager.model_name = "test_model"

    with patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager"
        "._ModelManager__load_config",
        return_value={},
    ), patch(
        "views_pipeline_core.modules.logging.LoggingModule.get_logger",
        return_value=MagicMock(),
    ), patch(
        "views_pipeline_core.managers.ConfigurationManager",
        return_value=MagicMock(),
    ), patch(
        "views_pipeline_core.managers.model.model.ModelManager"
        "._ModelManager__ascii_splash",
    ):
        manager = ForecastingModelManager(mock_path_manager)

    manager._args = MagicMock()
    manager._args.run_type = "calibration"
    manager._args.artifact_name = "latest"
    manager._args.eval_type = "standard"
    manager._sweep = False
    manager._eval_type = "standard"
    manager._project = "test_project"
    # _prediction_format is a property reading from configs["prediction_format"]
    # Default is "dataframe" when key is absent — no need to set explicitly.
    manager._partition_dict = {
        "calibration": {"train": (1, 400), "test": (401, 448)},
    }
    _default_configs = {
        "regression_targets": ["target_sb"],
        "classification_targets": [],
        "regression_point_metrics": ["mse"],
        "targets": ["target_sb"],
        "name": "test",
        "sweep": False,
        "run_type": "calibration",
        "timestamp": "20260101",
        "level": "cm",
        "steps": list(range(1, 37)),
    }
    # Replace config_manager with a mock that returns our controlled config
    mock_cm = MagicMock()
    mock_cm.get_combined_config.return_value = _default_configs
    manager._config_manager = mock_cm
    manager._wandb_module = MagicMock()
    manager._wandb_module.initialize_run.return_value.__enter__ = MagicMock()
    manager._wandb_module.initialize_run.return_value.__exit__ = MagicMock(
        return_value=False
    )
    manager._wandb_notifications = False
    manager._io = MagicMock()
    manager._evaluation_stage = MagicMock()
    manager._evaluation_stage._wandb_module = manager._wandb_module
    manager._evaluation_stage._io = manager._io

    return manager


def _make_pred_dfs(n=N_SEQUENCES, base_origin=400):
    """Create n prediction DataFrames with correct rolling-origin structure.

    Each sequence i covers months [base_origin + i + 1, base_origin + i + 36]
    to match _get_evaluation_step_mappings output.
    """
    steps = list(range(1, 37))
    dfs = []
    for i in range(n):
        months = [base_origin + i + s for s in steps]
        idx = pd.MultiIndex.from_tuples(
            [(m, 1) for m in months],
            names=["month_id", "priogrid_gid"],
        )
        dfs.append(pd.DataFrame(
            {"pred_target_sb": [0.1] * len(steps), "target_sb": [0.1] * len(steps)},
            index=idx,
        ))
    return dfs


def _make_mock_report():
    """Mock NativeEvaluator.evaluate() return."""
    report = MagicMock()
    report.to_dict.return_value = {
        "target": "mock", "task": "regression", "pred_type": "point",
        "schemas": {"step": {}, "time_series": {}, "month": {}},
    }
    report.to_dataframe.return_value = pd.DataFrame()
    return report


# ============================================================
# PATH 1: DF (legacy DataFrame format)
# ============================================================


class TestDFPath:
    """Characterize the legacy DataFrame evaluation path."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch("views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer")
    def test_df_path_validates_and_saves_each_sequence(
        self, mock_sniffer_cls, mock_log_creation,
    ):
        """DF path must validate each sequence via CorePredictionSniffer and save it."""
        manager = _make_manager()
        pred_dfs = _make_pred_dfs()

        manager._evaluate_model_artifact = MagicMock(return_value=pred_dfs)

        manager._execute_model_evaluation()

        # CorePredictionSniffer instantiated for each sequence
        assert mock_sniffer_cls.call_count == N_SEQUENCES
        # save_predictions called for each sequence
        assert manager._io.save_predictions.call_count == N_SEQUENCES

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch("views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer")
    def test_df_path_calls_evaluate_prediction_dataframe(
        self, mock_sniffer_cls, mock_log_creation,
    ):
        """DF path must call _evaluate_prediction_dataframe with raw predictions."""
        manager = _make_manager()
        pred_dfs = _make_pred_dfs()

        manager._evaluate_model_artifact = MagicMock(return_value=pred_dfs)

        manager._execute_model_evaluation()

        # Thin delegate should be called with the prediction list
        manager._evaluation_stage.evaluate.assert_called_once()

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch("views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer")
    def test_df_path_type_enforcement_rejects_dict(
        self, mock_sniffer_cls, mock_log_creation,
    ):
        """DF path: if _evaluate_model_artifact returns dict, must raise ValueError (ADR-042)."""
        manager = _make_manager()
        # Return dict when prediction_format="dataframe" — contract violation
        manager._evaluate_model_artifact = MagicMock(
            return_value={"target_sb": [MagicMock()]}
        )

        from views_pipeline_core.exceptions import ModelEvaluationException
        with pytest.raises(ModelEvaluationException, match="prediction_format='dataframe'"):
            manager._execute_model_evaluation()


# ============================================================
# PATH 2: No metrics in config
# ============================================================


class TestNoMetrics:
    """Characterize behavior when no evaluation metrics are configured."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch("views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer")
    def test_no_metrics_skips_evaluation(
        self, mock_sniffer_cls, mock_log_creation,
    ):
        """When _has_evaluation_metrics() returns False, skip metric evaluation."""
        manager = _make_manager()
        # Override configs to remove all metric keys
        manager._config_manager.get_combined_config.return_value = {
            "targets": ["target_sb"],
            "name": "test",
            "sweep": False,
            "run_type": "calibration",
            "timestamp": "20260101",
            "level": "cm",
            "steps": list(range(1, 37)),
        }
        pred_dfs = _make_pred_dfs()
        manager._evaluate_model_artifact = MagicMock(return_value=pred_dfs)

        manager._execute_model_evaluation()

        # Predictions still validated and saved...
        assert manager._io.save_predictions.call_count == N_SEQUENCES
        # ...but evaluation stage NOT called
        manager._evaluation_stage.evaluate.assert_not_called()


# ============================================================
# PATH 3: PF streaming path
# ============================================================


class TestPFStreamingPath:
    """Characterize the PredictionFrame streaming evaluation path."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_pf_path_calls_streaming_evaluation(self, mock_log_creation):
        """PF path must call _evaluate_model_artifact_streaming with origin_sink."""
        manager = _make_manager()
        pf_configs = dict(manager.configs)
        pf_configs["prediction_format"] = "prediction_frame"
        pf_configs["skip_evaluation_metrics"] = True
        manager._config_manager.get_combined_config.return_value = pf_configs

        manager._evaluate_model_artifact_streaming = MagicMock()

        manager._execute_model_evaluation()

        manager._evaluate_model_artifact_streaming.assert_called_once()
        # Verify origin_sink is a callable keyword argument
        _, kwargs = manager._evaluate_model_artifact_streaming.call_args
        assert "origin_sink" in kwargs
        assert callable(kwargs["origin_sink"])

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_pf_path_type_enforcement_rejects_non_dict(self, mock_log_creation):
        """PF path: if _forecast_model_artifact returns non-dict during forecasting, must raise."""
        manager = _make_manager()
        pf_configs = dict(manager.configs)
        pf_configs["prediction_format"] = "prediction_frame"
        manager._config_manager.get_combined_config.return_value = pf_configs

        # Test the forecasting type guard (evaluation path uses streaming, so
        # we test forecasting for the type enforcement characterization)
        manager._forecast_model_artifact = MagicMock(return_value=pd.DataFrame())

        from views_pipeline_core.exceptions import ModelForecastingException
        with pytest.raises(ModelForecastingException, match="prediction_format='prediction_frame'"):
            manager._execute_model_forecasting()


# ============================================================
# PATH 4: skip_evaluation_metrics flag
# ============================================================


class TestSkipEvaluationMetrics:
    """Characterize skip_evaluation_metrics behavior."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_skip_evaluation_metrics_skips_stage(self, mock_log_creation):
        """When skip_evaluation_metrics=True, evaluation stage must NOT be called."""
        manager = _make_manager()
        pf_configs = dict(manager.configs)
        pf_configs["prediction_format"] = "prediction_frame"
        pf_configs["skip_evaluation_metrics"] = True
        manager._config_manager.get_combined_config.return_value = pf_configs

        manager._evaluate_model_artifact_streaming = MagicMock()

        manager._execute_model_evaluation()

        # Stage should not be called when metrics are skipped
        manager._evaluation_stage.evaluate.assert_not_called()


# ============================================================
# PATH 5: WandB lifecycle
# ============================================================


class TestWandBLifecycle:
    """Characterize WandB run lifecycle in _execute_model_evaluation."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch("views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer")
    def test_wandb_run_initialized_with_evaluate_job_type(
        self, mock_sniffer_cls, mock_log_creation,
    ):
        """WandB run must be initialized with job_type='evaluate'."""
        manager = _make_manager()
        pred_dfs = _make_pred_dfs()
        manager._evaluate_model_artifact = MagicMock(return_value=pred_dfs)

        manager._execute_model_evaluation()

        manager._wandb_module.initialize_run.assert_called_once_with(
            project=manager._project,
            config=manager.configs,
            job_type="evaluate",
        )

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch("views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer")
    def test_wandb_finish_run_called_even_on_failure(
        self, mock_sniffer_cls, mock_log_creation,
    ):
        """WandB finish_run must be called even when evaluation raises."""
        manager = _make_manager()
        manager._evaluate_model_artifact = MagicMock(
            side_effect=RuntimeError("boom")
        )

        from views_pipeline_core.exceptions import ModelEvaluationException
        with pytest.raises(ModelEvaluationException):
            manager._execute_model_evaluation()

        manager._wandb_module.finish_run.assert_called_once()


# ============================================================
# PATH 6: Sequence count validation
# ============================================================


class TestSequenceCountValidation:
    """Characterize _assert_predictions_in_step_window sequence count check."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch("views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer")
    def test_wrong_sequence_count_raises_value_error(
        self, mock_sniffer_cls, mock_log_creation,
    ):
        """DF path: wrong number of sequences must raise ValueError."""
        manager = _make_manager()
        # Return 5 instead of required 13
        pred_dfs = _make_pred_dfs(n=5)
        manager._evaluate_model_artifact = MagicMock(return_value=pred_dfs)

        from views_pipeline_core.exceptions import ModelEvaluationException
        with pytest.raises(ModelEvaluationException, match="Evaluation failed"):
            manager._execute_model_evaluation()


# ============================================================
# PATH 7: PF permanent persistence for ensemble consumption
# ============================================================


class TestPFPermanentPersistence:
    """Verify PF evaluation path persists to ensemble-discoverable location."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_pf_eval_persists_to_permanent_location(self, mock_log_creation, tmp_path):
        """PF origin_sink must save to data_generated/predictions_{run_type}_{ts}/origin_{i}/{target}/."""
        import numpy as np
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        manager = _make_manager()
        manager._model_path.data_generated = tmp_path / "data" / "generated"
        manager._model_path.data_generated.mkdir(parents=True)
        manager._model_path.get_latest_model_artifact_path.return_value = (
            Path("calibration_model_20260501_120000")
        )

        pf_configs = dict(manager.configs)
        pf_configs["prediction_format"] = "prediction_frame"
        pf_configs["skip_evaluation_metrics"] = True
        pf_configs["skip_predictions_delivery"] = True
        pf_configs["timestamp"] = "20260501_120000"
        manager._config_manager.get_combined_config.return_value = pf_configs

        pf = PredictionFrame(
            y_pred=np.ones((10, 4), dtype=np.float32),
            identifiers={"time": np.arange(10), "unit": np.arange(10)},
        )

        def fake_streaming(eval_type, artifact_name, origin_sink):
            origin_sink(0, {"target_sb": pf})

        manager._evaluate_model_artifact_streaming = MagicMock(
            side_effect=fake_streaming
        )

        manager._execute_model_evaluation()

        permanent_dir = (
            manager._model_path.data_generated
            / "predictions_calibration_20260501_120000"
            / "origin_0"
            / "target_sb"
        )
        assert (permanent_dir / "y_pred.npy").exists()
        assert (permanent_dir / "identifiers.npz").exists()

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_pf_eval_staging_cleaned_up_after_metrics(self, mock_log_creation, tmp_path):
        """Staging path must be cleaned up after evaluation completes."""
        import numpy as np
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        manager = _make_manager()
        manager._model_path.data_generated = tmp_path / "data" / "generated"
        manager._model_path.data_generated.mkdir(parents=True)
        manager._model_path.get_latest_model_artifact_path.return_value = (
            Path("calibration_model_20260501_120000")
        )

        pf_configs = dict(manager.configs)
        pf_configs["prediction_format"] = "prediction_frame"
        pf_configs["skip_evaluation_metrics"] = True
        pf_configs["skip_predictions_delivery"] = True
        pf_configs["timestamp"] = "20260501_120000"
        manager._config_manager.get_combined_config.return_value = pf_configs

        pf = PredictionFrame(
            y_pred=np.ones((10, 4), dtype=np.float32),
            identifiers={"time": np.arange(10), "unit": np.arange(10)},
        )

        def fake_streaming(eval_type, artifact_name, origin_sink):
            origin_sink(0, {"target_sb": pf})

        manager._evaluate_model_artifact_streaming = MagicMock(
            side_effect=fake_streaming
        )

        manager._execute_model_evaluation()

        staging_dir = manager._model_path.data_generated / "_pf_staging"
        assert not staging_dir.exists()


# ============================================================
# PATH 8: PF forecast permanent persistence
# ============================================================


class TestPFForecastPersistence:
    """Verify PF forecast path persists to ensemble-discoverable location."""

    def test_pf_forecast_persists_to_data_generated(self, tmp_path):
        """PF forecast must save to data_generated/predictions_{run_type}_{ts}/{target}/."""
        import numpy as np
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        manager = _make_manager()
        manager._model_path.data_generated = tmp_path / "data" / "generated"
        manager._model_path.data_generated.mkdir(parents=True)
        manager._model_path.get_latest_model_artifact_path.return_value = (
            Path("calibration_model_20260501_120000")
        )

        pf_configs = dict(manager.configs)
        pf_configs["prediction_format"] = "prediction_frame"
        pf_configs["timestamp"] = "20260501_120000"
        manager._config_manager.get_combined_config.return_value = pf_configs

        pf = PredictionFrame(
            y_pred=np.ones((10, 4), dtype=np.float32),
            identifiers={"time": np.arange(10), "unit": np.arange(10)},
        )
        manager._forecast_model_artifact = MagicMock(
            return_value={"target_sb": pf}
        )
        manager._forecasting_stage = MagicMock()

        manager._execute_model_forecasting()

        pf_dir = (
            manager._model_path.data_generated
            / "predictions_calibration_20260501_120000"
            / "target_sb"
        )
        assert (pf_dir / "y_pred.npy").exists()
        assert (pf_dir / "identifiers.npz").exists()

    def test_pf_forecast_still_calls_forecasting_stage(self, tmp_path):
        """PF save must not prevent ForecastingStage from running."""
        import numpy as np
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        manager = _make_manager()
        manager._model_path.data_generated = tmp_path / "data" / "generated"
        manager._model_path.data_generated.mkdir(parents=True)
        manager._model_path.get_latest_model_artifact_path.return_value = (
            Path("calibration_model_20260501_120000")
        )

        pf_configs = dict(manager.configs)
        pf_configs["prediction_format"] = "prediction_frame"
        pf_configs["timestamp"] = "20260501_120000"
        manager._config_manager.get_combined_config.return_value = pf_configs

        pf = PredictionFrame(
            y_pred=np.ones((10, 4), dtype=np.float32),
            identifiers={"time": np.arange(10), "unit": np.arange(10)},
        )
        manager._forecast_model_artifact = MagicMock(
            return_value={"target_sb": pf}
        )
        manager._forecasting_stage = MagicMock()

        manager._execute_model_forecasting()

        manager._forecasting_stage.process_and_save_forecast.assert_called_once()
