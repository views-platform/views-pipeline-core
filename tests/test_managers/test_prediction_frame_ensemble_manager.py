"""
TDD characterization tests for PredictionFrameEnsembleManager.

Verifies the composition-based ensemble orchestrator for sample-based
(PredictionFrame) predictions. Numpy all the way — no parquet.
"""

import logging
import subprocess

import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex
from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.managers.ensemble import EnsemblePathManager
from views_pipeline_core.managers.ensemble.dataframe_ensemble import EnsembleContext
from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    SUPPORTED_PF_AGGREGATION_METHODS,
    _aggregate_prediction_frames,
    PredictionFrameEnsembleManager,
)
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelManager,
)


# ============================================================================
# Helpers
# ============================================================================

def _pf_index(n_rows):
    return SpatioTemporalIndex(
        time=np.arange(n_rows, dtype=np.int64),
        unit=np.arange(1000, 1000 + n_rows, dtype=np.int64),
        level=SpatialLevel.PGM,
    )


def _make_pf(n_rows=100, n_samples=64, seed=42):
    """Create a PredictionFrame with deterministic random data."""
    rng = np.random.default_rng(seed)
    y_pred = rng.standard_normal((n_rows, n_samples)).astype(np.float32)
    return PredictionFrame(y_pred, _pf_index(n_rows))


def _make_pf_with_values(n_rows, n_samples, fill_value):
    """Create a PredictionFrame filled with a constant value."""
    y_pred = np.full((n_rows, n_samples), fill_value, dtype=np.float32)
    return PredictionFrame(y_pred, _pf_index(n_rows))


# ============================================================================
# Shared test data
# ============================================================================

COMBINED_CONFIGS = {
    "name": "test_pf_ensemble",
    "models": ["hydra_alpha", "hydra_beta"],
    "aggregation": "concat",
    "regression_targets": ["ged_sb"],
    "classification_targets": [],
    "targets": ["ged_sb"],
    "level": "pgm",
    "reconciliation": None,
    "reconcile_with": None,
    "use_weights": False,
    "weights": {},
    "timestamp": "20241105_120000",
    "deployment_status": "deployed",
    "prediction_format": "prediction_frame",
    "steps": list(range(1, 37)),
    "run_type": "calibration",
}

PARTITION_DICT = {
    "calibration": {
        "train": (121, 444),
        "test": (445, 492),
    },
    "validation": {
        "train": (121, 492),
        "test": (493, 540),
    },
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ensemble_path():
    """Create mock EnsemblePathManager with all required attributes."""
    mock = MagicMock(spec=EnsemblePathManager)
    mock.model_name = "test_pf_ensemble"
    mock.target = "ensemble"
    mock.root = Path("/test/root")
    mock.models = Path("/test/root/ensembles")
    mock.model_dir = Path("/test/root/ensembles/test_pf_ensemble")
    mock.artifacts = Path("/test/root/ensembles/test_pf_ensemble/artifacts")
    mock.data = Path("/test/root/ensembles/test_pf_ensemble/data")
    mock.data_generated = Path("/test/root/ensembles/test_pf_ensemble/data/generated")
    mock.data_raw = Path("/test/root/ensembles/test_pf_ensemble/data/raw")
    mock.reports = Path("/test/root/ensembles/test_pf_ensemble/reports")
    mock.logging = Path("/test/root/ensembles/test_pf_ensemble/logs")

    mock.get_scripts.return_value = {
        "config_deployment.py": "/test/configs/config_deployment.py",
        "config_hyperparameters.py": "/test/configs/config_hyperparameters.py",
        "config_meta.py": "/test/configs/config_meta.py",
        "config_partitions.py": "/test/configs/config_partitions.py",
        "main.py": "/test/main.py",
    }
    return mock


@pytest.fixture
def mock_wandb_module():
    mock = MagicMock()
    mock.login = MagicMock()
    mock.finish_run = MagicMock()
    mock.send_alert = MagicMock()
    mock.log = MagicMock()
    mock.initialize_run.return_value.__enter__ = MagicMock()
    mock.initialize_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock


@pytest.fixture
def mock_config_manager():
    mock = MagicMock()
    mock.get_combined_config.return_value = COMBINED_CONFIGS.copy()
    return mock


@pytest.fixture
def sample_args():
    return ForecastingModelArgs(
        run_type="calibration",
        train=False,
        evaluate=True,
        forecast=False,
        saved=True,
        eval_type="standard",
    )


@pytest.fixture
def ensemble_context(mock_ensemble_path):
    return EnsembleContext(
        configs=COMBINED_CONFIGS.copy(),
        model_path=mock_ensemble_path,
        run_type="calibration",
        project="test_pf_ensemble_calibration",
        eval_type="standard",
        args=ForecastingModelArgs(
            run_type="calibration",
            train=False,
            evaluate=True,
            forecast=False,
            saved=True,
            eval_type="standard",
        ),
        models=["hydra_alpha", "hydra_beta"],
        aggregation="concat",
        targets=["ged_sb"],
        reconciliation=None,
        reconcile_with=None,
        use_weights=False,
        weights={},
        timestamp="20241105_120000",
        deployment_status="deployed",
        prediction_format="prediction_frame",
        partition_dict=PARTITION_DICT,
    )


@pytest.fixture
def pf_manager(mock_ensemble_path, mock_wandb_module, mock_config_manager):
    """Create PredictionFrameEnsembleManager with all external deps mocked."""
    config_returns = [
        {"deployment_status": "deployed"},
        {"steps": list(range(1, 37))},
        {
            "name": "test_pf_ensemble",
            "aggregation": "concat",
            "regression_targets": ["ged_sb"],
            "classification_targets": [],
            "targets": ["ged_sb"],
            "level": "pgm",
        },
        {"models": ["hydra_alpha", "hydra_beta"]},
        PARTITION_DICT,
    ]

    patches = [
        patch.object(
            PredictionFrameEnsembleManager, "_load_config",
            side_effect=config_returns,
        ),
        patch(
            "views_pipeline_core.managers.configuration.ConfigurationManager",
            return_value=mock_config_manager,
        ),
        patch("views_pipeline_core.modules.logging.LoggingModule"),
        patch(
            "views_pipeline_core.modules.wandb.WandBModule",
            return_value=mock_wandb_module,
        ),
        patch("views_pipeline_core.managers.evaluation.stage.EvaluationStage"),
        patch("views_pipeline_core.managers.reporting.stage.ReportingStage"),
    ]

    for p in patches:
        p.start()

    try:
        mgr = PredictionFrameEnsembleManager(
            ensemble_path=mock_ensemble_path,
            wandb_notifications=False,
            use_prediction_store=False,
        )
        mgr._wandb_module = mock_wandb_module
        mgr._config_manager = mock_config_manager
        yield mgr
    finally:
        for p in patches:
            p.stop()


# ============================================================================
# Phase 1: Pure Aggregation Function
# ============================================================================

class TestPredictionFrameAggregation:
    """Tests for _aggregate_prediction_frames() pure function."""

    def test_concat_stacks_samples_axis(self):
        pf1 = _make_pf(n_rows=100, n_samples=64, seed=1)
        pf2 = _make_pf(n_rows=100, n_samples=64, seed=2)
        pf3 = _make_pf(n_rows=100, n_samples=64, seed=3)

        result = _aggregate_prediction_frames([pf1, pf2, pf3], method="concat")

        assert result.values.shape == (100, 192)

    def test_concat_preserves_identifiers(self):
        pf1 = _make_pf(n_rows=100, n_samples=64, seed=1)
        pf2 = _make_pf(n_rows=100, n_samples=64, seed=2)

        result = _aggregate_prediction_frames([pf1, pf2], method="concat")

        np.testing.assert_array_equal(result.identifiers["time"], pf1.identifiers["time"])
        np.testing.assert_array_equal(result.identifiers["unit"], pf1.identifiers["unit"])

    def test_arithmetic_mean_averages_samples(self):
        pf1 = _make_pf_with_values(100, 64, fill_value=2.0)
        pf2 = _make_pf_with_values(100, 64, fill_value=4.0)
        pf3 = _make_pf_with_values(100, 64, fill_value=6.0)

        result = _aggregate_prediction_frames(
            [pf1, pf2, pf3], method="arithmetic_mean"
        )

        assert result.values.shape == (100, 64)
        np.testing.assert_allclose(result.values, 4.0, atol=1e-6)

    def test_arithmetic_mean_preserves_identifiers(self):
        pf1 = _make_pf(n_rows=100, n_samples=64, seed=1)
        pf2 = _make_pf(n_rows=100, n_samples=64, seed=2)

        result = _aggregate_prediction_frames(
            [pf1, pf2], method="arithmetic_mean"
        )

        np.testing.assert_array_equal(result.identifiers["time"], pf1.identifiers["time"])
        np.testing.assert_array_equal(result.identifiers["unit"], pf1.identifiers["unit"])

    def test_mismatched_rows_raises(self):
        pf1 = _make_pf(n_rows=100, n_samples=64, seed=1)
        pf2 = _make_pf(n_rows=50, n_samples=64, seed=2)

        with pytest.raises(ValueError, match="n_rows"):
            _aggregate_prediction_frames([pf1, pf2], method="concat")

    def test_mismatched_identifiers_raises(self):
        pf1 = _make_pf(n_rows=100, n_samples=64, seed=1)
        pf2 = PredictionFrame(
            np.ones((100, 64), dtype=np.float32),
            SpatioTemporalIndex(
                time=np.arange(100, 200, dtype=np.int64),
                unit=np.arange(1000, 1100, dtype=np.int64),
                level=SpatialLevel.PGM,
            ),
        )

        with pytest.raises(ValueError, match="identifiers"):
            _aggregate_prediction_frames([pf1, pf2], method="concat")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _aggregate_prediction_frames([], method="concat")

    def test_single_pf_passthrough(self):
        pf = _make_pf(n_rows=100, n_samples=64, seed=42)

        for method in ("concat", "arithmetic_mean"):
            result = _aggregate_prediction_frames([pf], method=method)
            np.testing.assert_array_equal(result.values, pf.values)
            np.testing.assert_array_equal(
                result.identifiers["time"], pf.identifiers["time"]
            )

    def test_unsupported_method_raises(self):
        pf = _make_pf(n_rows=100, n_samples=64, seed=42)

        with pytest.raises(ValueError, match="median"):
            _aggregate_prediction_frames([pf], method="median")


# ============================================================================
# Phase 2: Constants and Context Reuse
# ============================================================================

class TestPredictionFrameEnsembleConstants:
    """Tests for module-level constants and EnsembleContext reuse."""

    def test_supported_aggregation_methods_is_frozenset(self):
        assert isinstance(SUPPORTED_PF_AGGREGATION_METHODS, frozenset)

    def test_supported_methods_contains_expected(self):
        assert {"concat", "arithmetic_mean"} <= SUPPORTED_PF_AGGREGATION_METHODS

    def test_ensemble_context_accepts_prediction_frame_format(self, mock_ensemble_path):
        ctx = EnsembleContext(
            configs=COMBINED_CONFIGS.copy(),
            model_path=mock_ensemble_path,
            run_type="calibration",
            project="test",
            eval_type="standard",
            args=MagicMock(),
            models=["a"],
            aggregation="concat",
            targets=["ged_sb"],
            reconciliation=None,
            reconcile_with=None,
            use_weights=False,
            weights={},
            timestamp="t",
            deployment_status="shadow",
            prediction_format="prediction_frame",
            partition_dict={},
        )
        assert ctx.prediction_format == "prediction_frame"

    def test_ensemble_context_is_frozen(self, mock_ensemble_path):
        ctx = EnsembleContext(
            configs=COMBINED_CONFIGS.copy(),
            model_path=mock_ensemble_path,
            run_type="calibration",
            project="test",
            eval_type="standard",
            args=MagicMock(),
            models=["a"],
            aggregation="concat",
            targets=["ged_sb"],
            reconciliation=None,
            reconcile_with=None,
            use_weights=False,
            weights={},
            timestamp="t",
            deployment_status="shadow",
            prediction_format="prediction_frame",
            partition_dict={},
        )
        with pytest.raises(FrozenInstanceError):
            ctx.prediction_format = "dataframe"


# ============================================================================
# Phase 3: Manager Construction
# ============================================================================

class TestPredictionFrameEnsembleConstruction:
    """Tests for PredictionFrameEnsembleManager constructor."""

    def test_no_inheritance_from_model_manager(self, pf_manager):
        assert not isinstance(pf_manager, ModelManager)

    def test_no_inheritance_from_forecasting_model_manager(self, pf_manager):
        assert not isinstance(pf_manager, ForecastingModelManager)

    def test_ensemble_path_stored(self, pf_manager, mock_ensemble_path):
        assert pf_manager._ensemble_path is mock_ensemble_path

    def test_wandb_module_constructed(self, pf_manager, mock_wandb_module):
        assert pf_manager._wandb_module is mock_wandb_module

    def test_evaluation_stage_constructed(self, pf_manager):
        assert hasattr(pf_manager, "_evaluation_stage")

    def test_no_prediction_store_by_default(self, pf_manager):
        assert pf_manager._use_prediction_store is False


# ============================================================================
# Phase 4: PredictionFrame Artifact Loading
# ============================================================================

class TestPredictionFrameLoading:
    """Tests for _load_or_generate_pf() method."""

    def test_load_existing_pf_from_disk(self, pf_manager, ensemble_context):
        pf = _make_pf()
        with patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.load_pf", return_value=pf) as mock_load, \
             patch("pathlib.Path.exists", return_value=True):
            result = pf_manager._load_or_generate_pf(
                model_path=MagicMock(),
                model_name="hydra_alpha",
                pf_dir=Path("/test/data/generated/predictions_calibration_20241105_120000/ged_sb"),
                ctx=ensemble_context,
                mmap=False,
            )
            assert result is pf
            mock_load.assert_called_once()

    def test_generate_when_missing(self, pf_manager, ensemble_context):
        pf = _make_pf()
        mock_model_path = MagicMock()

        with patch("pathlib.Path.exists", return_value=False), \
             patch.object(pf_manager, "_execute_shell_script") as mock_exec, \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.load_pf", return_value=pf):
            result = pf_manager._load_or_generate_pf(
                model_path=mock_model_path,
                model_name="hydra_alpha",
                pf_dir=Path("/test/data/generated/preds/ged_sb"),
                ctx=ensemble_context,
                mmap=False,
                evaluate=True,
            )
            mock_exec.assert_called_once()
            assert result is pf

    def test_raises_when_no_output_after_generation(self, pf_manager, ensemble_context):
        mock_model_path = MagicMock()

        with patch("pathlib.Path.exists", return_value=False), \
             patch.object(pf_manager, "_execute_shell_script"), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.load_pf", side_effect=FileNotFoundError):
            with pytest.raises(PipelineException):
                pf_manager._load_or_generate_pf(
                    model_path=mock_model_path,
                    model_name="hydra_alpha",
                    pf_dir=Path("/test/data/generated/preds/ged_sb"),
                    ctx=ensemble_context,
                    mmap=False,
                    evaluate=True,
                )

    def test_mmap_used_for_evaluation(self, pf_manager, ensemble_context):
        pf = _make_pf()
        with patch("pathlib.Path.exists", return_value=True), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.load_pf", return_value=pf) as mock_load:
            pf_manager._load_or_generate_pf(
                model_path=MagicMock(),
                model_name="hydra_alpha",
                pf_dir=Path("/test/preds/ged_sb"),
                ctx=ensemble_context,
                mmap=True,
            )
            _, kwargs = mock_load.call_args
            assert kwargs.get("mmap") is True

    def test_mmap_not_used_for_forecast(self, pf_manager, ensemble_context):
        pf = _make_pf()
        with patch("pathlib.Path.exists", return_value=True), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.load_pf", return_value=pf) as mock_load:
            pf_manager._load_or_generate_pf(
                model_path=MagicMock(),
                model_name="hydra_alpha",
                pf_dir=Path("/test/preds/ged_sb"),
                ctx=ensemble_context,
                mmap=False,
            )
            _, kwargs = mock_load.call_args
            assert kwargs.get("mmap") is False


# ============================================================================
# Phase 5: Evaluation Flow
# ============================================================================

class TestPredictionFrameEvaluationFlow:
    """Tests for _evaluate_ensemble() and evaluation delegation."""

    def test_evaluate_ensemble_aggregates_per_sequence(self, pf_manager, ensemble_context):
        pf1 = _make_pf(seed=1)

        with patch.object(pf_manager, "_evaluate_model_artifact", return_value={"ged_sb": [pf1]}), \
             patch.object(pf_manager, "_evaluate_predictions"), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.save_pf"), \
             patch(
                 "views_pipeline_core.managers.model.ForecastingModelManager._resolve_evaluation_sequence_number",
                 return_value=1,
             ), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble._aggregate_prediction_frames",
                 return_value=pf1,
             ) as mock_agg:
            pf_manager._evaluate_ensemble(ensemble_context)
            assert mock_agg.call_count >= 1

    def test_evaluate_context_has_prediction_frame_format(self, pf_manager, ensemble_context):
        pf = _make_pf()

        with patch.object(pf_manager, "_evaluate_model_artifact", return_value={"ged_sb": [pf]}), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.save_pf"), \
             patch(
                 "views_pipeline_core.managers.model.ForecastingModelManager._resolve_evaluation_sequence_number",
                 return_value=1,
             ), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble._aggregate_prediction_frames",
                 return_value=pf,
             ), \
             patch.object(pf_manager, "_evaluate_predictions") as mock_eval:
            pf_manager._evaluate_ensemble(ensemble_context)
            eval_ctx = mock_eval.call_args[0][1]
            assert eval_ctx.prediction_format == "prediction_frame"


# ============================================================================
# Phase 6: Forecasting Flow
# ============================================================================

class TestPredictionFrameForecastingFlow:
    """Tests for _forecast_ensemble()."""

    def test_forecast_ensemble_aggregates_models(self, pf_manager, ensemble_context):
        pf = _make_pf()

        with patch.object(pf_manager, "_forecast_model_artifact", return_value={"ged_sb": pf}), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.save_pf"), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.handle_ensemble_log_creation",
             ), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble._aggregate_prediction_frames",
                 return_value=pf,
             ) as mock_agg:
            pf_manager._forecast_ensemble(ensemble_context)
            assert mock_agg.call_count >= 1

    def test_forecast_saves_via_pf_save(self, pf_manager, ensemble_context):
        pf = _make_pf()

        with patch.object(pf_manager, "_forecast_model_artifact", return_value={"ged_sb": pf}), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.handle_ensemble_log_creation",
             ), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble._aggregate_prediction_frames",
                 return_value=pf,
             ), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.save_pf") as mock_save:
            pf_manager._forecast_ensemble(ensemble_context)
            mock_save.assert_called()

    def test_forecast_no_reconciliation(self, pf_manager, ensemble_context):
        assert not hasattr(pf_manager, "_apply_reconciliation")

    def test_forecast_returns_dict_of_pf(self, pf_manager, ensemble_context):
        pf = _make_pf()

        with patch.object(pf_manager, "_forecast_model_artifact", return_value={"ged_sb": pf}), \
             patch("views_pipeline_core.managers.ensemble.prediction_frame_ensemble.save_pf"), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.handle_ensemble_log_creation",
             ), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble._aggregate_prediction_frames",
                 return_value=pf,
             ):
            result = pf_manager._forecast_ensemble(ensemble_context)
            assert isinstance(result, dict)
            assert all(isinstance(v, PredictionFrame) for v in result.values())


# ============================================================================
# Phase 7: Training Flow
# ============================================================================

class TestPredictionFrameTrainingFlow:
    """Tests for _train_ensemble()."""

    def test_train_dispatches_subprocess_per_model(self, pf_manager, ensemble_context):
        with patch.object(pf_manager, "_train_model_artifact") as mock_train:
            pf_manager._train_ensemble(ensemble_context)
            assert mock_train.call_count == len(ensemble_context.models)

    def test_train_passes_train_flag_to_submodel(self, pf_manager, ensemble_context):
        with patch.object(pf_manager, "_create_model_args", return_value=MagicMock()) as mock_args, \
             patch.object(pf_manager, "_execute_shell_script"), \
             patch("views_pipeline_core.data.model_path.ModelPathManager"):
            pf_manager._train_model_artifact("hydra_alpha", ensemble_context)
            mock_args.assert_called_once_with(ensemble_context, train=True)


# ============================================================================
# Phase 8: Entry Point and Error Handling
# ============================================================================

class TestPredictionFrameEntryPoint:
    """Tests for execute_single_run() and error handling."""

    def test_execute_single_run_validates_args_type(self, pf_manager):
        with pytest.raises(ValueError, match="ForecastingModelArgs"):
            pf_manager.execute_single_run("not_args")

    def test_execute_single_run_calls_core_config_sniffer(self, pf_manager, sample_args):
        with patch(
            "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.CoreConfigSniffer"
        ) as mock_sniffer, \
             patch.object(pf_manager._wandb_module, "login"), \
             patch.object(pf_manager._config_manager, "update_for_single_run"), \
             patch.object(pf_manager, "_build_context", return_value=MagicMock(args=sample_args)), \
             patch.object(pf_manager, "_execute_model_tasks"), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.validate_ensemble_model"
             ):
            pf_manager.execute_single_run(sample_args)
            mock_sniffer.assert_called_once()
            mock_sniffer.return_value.sniff_all.assert_called_once()

    def test_execute_single_run_calls_wandb_login(self, pf_manager, sample_args):
        with patch(
            "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.CoreConfigSniffer"
        ), \
             patch.object(pf_manager._wandb_module, "login") as mock_login, \
             patch.object(pf_manager._config_manager, "update_for_single_run"), \
             patch.object(pf_manager, "_build_context", return_value=MagicMock(args=sample_args)), \
             patch.object(pf_manager, "_execute_model_tasks"), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.validate_ensemble_model"
             ):
            pf_manager.execute_single_run(sample_args)
            mock_login.assert_called_once()

    def test_wandb_login_failure_propagates(self, pf_manager, sample_args):
        with patch(
            "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.CoreConfigSniffer"
        ), \
             patch.object(
                 pf_manager._wandb_module, "login",
                 side_effect=RuntimeError("wandb login failed"),
             ):
            with pytest.raises(RuntimeError, match="wandb login failed"):
                pf_manager.execute_single_run(sample_args)

    def test_execute_model_tasks_dispatches_train(self, pf_manager):
        ctx = MagicMock()
        ctx.args.train = True
        ctx.args.evaluate = False
        ctx.args.forecast = False
        ctx.args.report = False

        with patch.object(pf_manager, "_execute_model_training") as mock_train:
            pf_manager._execute_model_tasks(ctx)
            mock_train.assert_called_once_with(ctx)

    def test_execute_model_tasks_dispatches_evaluate(self, pf_manager):
        ctx = MagicMock()
        ctx.args.train = False
        ctx.args.evaluate = True
        ctx.args.forecast = False
        ctx.args.report = False

        with patch.object(pf_manager, "_execute_model_evaluation") as mock_eval:
            pf_manager._execute_model_tasks(ctx)
            mock_eval.assert_called_once_with(ctx)

    def test_execute_model_tasks_dispatches_forecast(self, pf_manager):
        ctx = MagicMock()
        ctx.args.train = False
        ctx.args.evaluate = False
        ctx.args.forecast = True
        ctx.args.report = False

        with patch.object(pf_manager, "_execute_model_forecasting") as mock_fc:
            pf_manager._execute_model_tasks(ctx)
            mock_fc.assert_called_once_with(ctx)

    def test_subprocess_timeout_raises_pipeline_exception(self, pf_manager, ensemble_context):
        mock_model_path = MagicMock()
        mock_model_args = MagicMock()
        mock_model_args.to_shell_command.return_value = ["python", "main.py"]

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="test", timeout=7200)):
            with pytest.raises(PipelineException):
                pf_manager._execute_shell_script(
                    mock_model_path, "hydra_alpha", mock_model_args
                )

    def test_subprocess_failure_raises_pipeline_exception(self, pf_manager, ensemble_context):
        mock_model_path = MagicMock()
        mock_model_args = MagicMock()
        mock_model_args.to_shell_command.return_value = ["python", "main.py"]

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "test")):
            with pytest.raises(PipelineException):
                pf_manager._execute_shell_script(
                    mock_model_path, "hydra_alpha", mock_model_args
                )

    def test_validation_skipped_when_train_true(self, pf_manager, mock_wandb_module):
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
            evaluate=False,
            forecast=False,
            saved=False,
            eval_type="standard",
        )

        with patch(
            "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.CoreConfigSniffer"
        ), \
             patch.object(mock_wandb_module, "login"), \
             patch.object(pf_manager._config_manager, "update_for_single_run"), \
             patch.object(pf_manager, "_build_context", return_value=MagicMock(args=args)), \
             patch.object(pf_manager, "_execute_model_tasks"), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.validate_ensemble_model"
             ) as mock_validate:
            pf_manager.execute_single_run(args)
            mock_validate.assert_not_called()

    def test_validation_called_when_train_false(self, pf_manager, sample_args, mock_wandb_module):
        with patch(
            "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.CoreConfigSniffer"
        ), \
             patch.object(mock_wandb_module, "login"), \
             patch.object(pf_manager._config_manager, "update_for_single_run"), \
             patch.object(pf_manager, "_build_context", return_value=MagicMock(args=sample_args)), \
             patch.object(pf_manager, "_execute_model_tasks"), \
             patch(
                 "views_pipeline_core.managers.ensemble.prediction_frame_ensemble.validate_ensemble_model"
             ) as mock_validate:
            pf_manager.execute_single_run(sample_args)
            mock_validate.assert_called_once()


# ============================================================================
# Test config_modelset merge semantics (C-151)
# ============================================================================


class TestConfigModelsetMerge:
    """config_modelset → effective_meta merge (CIC §4)."""

    @pytest.fixture
    def _build(self, mock_ensemble_path, mock_wandb_module):
        """Factory: construct PF manager with controlled
        config_meta and config_modelset return values."""

        def _go(meta_dict, modelset_dict):
            config_returns = [
                {"deployment_status": "deployed"},
                {"steps": list(range(1, 37))},
                meta_dict,
                modelset_dict,
                PARTITION_DICT,
            ]
            mock_cm = MagicMock()
            mock_cm.get_combined_config.return_value = (
                COMBINED_CONFIGS.copy()
            )

            with patch.object(
                PredictionFrameEnsembleManager,
                "_load_config",
                side_effect=config_returns,
            ), patch(
                "views_pipeline_core.managers.configuration"
                ".ConfigurationManager",
                return_value=mock_cm,
            ) as MockCfgCls, patch(
                "views_pipeline_core.modules.logging.LoggingModule",
            ), patch(
                "views_pipeline_core.modules.wandb.WandBModule",
                return_value=mock_wandb_module,
            ), patch(
                "views_pipeline_core.managers.evaluation"
                ".stage.EvaluationStage",
            ), patch(
                "views_pipeline_core.managers.reporting"
                ".stage.ReportingStage",
            ):
                mgr = PredictionFrameEnsembleManager(
                    ensemble_path=mock_ensemble_path,
                    wandb_notifications=False,
                    use_prediction_store=False,
                )
                return mgr, MockCfgCls

        return _go

    def test_no_modelset_meta_unchanged(self, _build):
        """effective_meta equals config_meta when modelset is absent."""
        meta = {"name": "test_pf", "targets": ["ged_sb"]}
        _, MockCfgCls = _build(meta, None)
        passed_meta = MockCfgCls.call_args.kwargs["config_meta"]
        assert passed_meta == {"name": "test_pf", "targets": ["ged_sb"]}

    def test_modelset_disjoint_keys_merged(self, _build):
        """Disjoint keys from both configs appear in effective_meta."""
        meta = {"name": "test_pf", "targets": ["ged_sb"]}
        modelset = {"models": ["hydra_alpha", "hydra_beta"]}
        _, MockCfgCls = _build(meta, modelset)
        passed_meta = MockCfgCls.call_args.kwargs["config_meta"]
        assert passed_meta == {
            "name": "test_pf",
            "targets": ["ged_sb"],
            "models": ["hydra_alpha", "hydra_beta"],
        }

    def test_modelset_overlap_precedence(self, _build):
        """config_modelset values override config_meta on collision."""
        meta = {"models": ["old"], "description": "Test"}
        modelset = {"models": ["new_a", "new_b"]}
        _, MockCfgCls = _build(meta, modelset)
        passed_meta = MockCfgCls.call_args.kwargs["config_meta"]
        assert passed_meta["models"] == ["new_a", "new_b"]
        assert passed_meta["description"] == "Test"

    def test_modelset_overlap_emits_warning(self, _build, caplog):
        """Collision emits logger.warning naming the colliding keys."""
        with caplog.at_level(
            logging.WARNING,
            logger=(
                "views_pipeline_core.managers"
                ".ensemble.prediction_frame_ensemble"
            ),
        ):
            _build({"models": ["old"]}, {"models": ["new"]})
        assert "config_modelset overlaps config_meta" in caplog.text
        assert "models" in caplog.text

    def test_merge_does_not_mutate_original_config_meta(self, _build):
        """effective_meta is a copy — _config_meta stays unchanged."""
        meta = {"models": ["old"], "description": "Test"}
        modelset = {"models": ["new"]}
        mgr, _ = _build(meta, modelset)
        assert mgr._config_meta == {
            "models": ["old"],
            "description": "Test",
        }
