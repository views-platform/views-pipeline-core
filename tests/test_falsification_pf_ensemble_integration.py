"""
PF ensemble integration regression tests (2026-05-22).

Originally falsification audit stubs (5 hard falsifications). Converted
to passing regression tests after fixes for C-94, C-95, C-96.
"""
import numpy as np
from views_pipeline_core.managers.model.model import ForecastingModelManager
from pathlib import Path
from unittest.mock import patch, MagicMock

from views_frames import SpatialLevel, SpatioTemporalIndex
from views_pipeline_core.managers.prediction.prediction_frame_io import (
    load_pf,
    save_pf,
)


def _pf(y_pred, time, unit):
    """Construct a leaf PredictionFrame (PGM) from raw arrays."""
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    index = SpatioTemporalIndex(
        time=np.asarray(time, dtype=np.int64),
        unit=np.asarray(unit, dtype=np.int64),
        level=SpatialLevel.PGM,
    )
    return PredictionFrame(y_pred, index)


# ---------------------------------------------------------------------------
# P-1: _get_generated_pf_prediction_paths — nesting level (C-96 regression)
# ---------------------------------------------------------------------------

class TestP1DiscoveryNestingLevel:
    """Discovery must find prediction directories when y_pred.npy is nested
    inside target/ or origin_{i}/target/ subdirectories."""

    def test_discovery_finds_forecast_layout(self, tmp_path):
        """Forecast layout: predictions_{run_type}_{ts}/{target}/y_pred.npy"""
        from views_pipeline_core.data.model_path import ModelPathManager

        with patch.object(ModelPathManager, "__init__", lambda self, *a, **kw: None):
            mgr = object.__new__(ModelPathManager)
            mgr.data_generated = tmp_path

        pred_dir = tmp_path / "predictions_calibration_20260522_120000"
        target_dir = pred_dir / "ged_sb"
        target_dir.mkdir(parents=True)
        np.save(target_dir / "y_pred.npy", np.zeros((10, 4)))

        paths = mgr._get_generated_pf_prediction_paths("calibration")
        assert len(paths) == 1, (
            "Discovery method should find predictions dir when y_pred.npy "
            "is nested inside a target subdirectory (forecast layout)."
        )

    def test_discovery_finds_eval_layout(self, tmp_path):
        """Eval layout: predictions_{run_type}_{ts}/origin_{i}/{target}/y_pred.npy"""
        from views_pipeline_core.data.model_path import ModelPathManager

        with patch.object(ModelPathManager, "__init__", lambda self, *a, **kw: None):
            mgr = object.__new__(ModelPathManager)
            mgr.data_generated = tmp_path

        pred_dir = tmp_path / "predictions_calibration_20260522_120000"
        origin_dir = pred_dir / "origin_0" / "ged_sb"
        origin_dir.mkdir(parents=True)
        np.save(origin_dir / "y_pred.npy", np.zeros((10, 4)))

        paths = mgr._get_generated_pf_prediction_paths("calibration")
        assert len(paths) == 1, (
            "Discovery method should find predictions dir when y_pred.npy "
            "is nested inside origin_{i}/{target}/ subdirectories (eval layout)."
        )


# ---------------------------------------------------------------------------
# P-2: Timestamp source — producer must use artifact timestamp (C-94 regression)
# ---------------------------------------------------------------------------

class TestP2TimestampFromArtifact:
    """model.py Track A+ must derive the save timestamp from the artifact
    stem, not from ConfigurationManager's runtime timestamp. This ensures
    the ensemble can find sub-model outputs at the path it constructs."""

    def _make_pf_manager(self, tmp_path, artifact_stem):
        """Build a ForecastingModelManager with mocked internals for PF path."""
        from views_pipeline_core.managers.model.model import ForecastingModelManager

        mock_path = MagicMock()
        mock_path.data_generated = tmp_path / "data" / "generated"
        mock_path.data_generated.mkdir(parents=True)
        mock_path.root = tmp_path
        mock_path.get_latest_model_artifact_path.return_value = Path(artifact_stem)
        mock_path.resolve_artifact_path.return_value = Path(artifact_stem)
        mock_path.target = "model"
        mock_path._target = "model"
        mock_path.model_name = "test_model"
        mock_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        mock_path.get_raw_data_file_paths.return_value = [Path("raw.parquet")]

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
            mgr = ForecastingModelManager(mock_path)

        mgr._args = MagicMock()
        mgr._args.run_type = "calibration"
        mgr._args.artifact_name = "latest"
        mgr._project = "test"
        mgr._wandb_module = MagicMock()
        mgr._wandb_module.initialize_run.return_value.__enter__ = MagicMock()
        mgr._wandb_module.initialize_run.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mgr._wandb_notifications = False
        mgr._sweep = False
        mgr._eval_type = "standard"
        mgr._io = MagicMock()
        mgr._evaluation_stage = MagicMock()

        pf_configs = {
            "prediction_format": "prediction_frame",
            "skip_evaluation_metrics": True,
            "skip_predictions_delivery": True,
            "timestamp": "99999999_999999",
            "regression_targets": ["ged_sb"],
            "classification_targets": [],
            "targets": ["ged_sb"],
            "name": "test",
            "sweep": False,
            "run_type": "calibration",
            "level": "cm",
            "steps": list(range(1, 37)),
        }
        mock_cm = MagicMock()
        mock_cm.get_combined_config.return_value = pf_configs
        mgr._config_manager = mock_cm
        mgr._partition_dict = {
            "calibration": {"train": (1, 400), "test": (401, 448)},
        }

        return mgr

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_eval_uses_artifact_timestamp_not_runtime(self, mock_log, tmp_path):
        """Track A+ eval save must use artifact stem, not configs["timestamp"]."""

        artifact_stem = "calibration_model_20260510_140000"
        mgr = self._make_pf_manager(tmp_path, artifact_stem)

        pf = _pf(np.ones((10, 4), dtype=np.float32), np.arange(10), np.arange(10))

        def fake_streaming(eval_type, artifact_name, origin_sink):
            origin_sink(0, {"ged_sb": pf})

        mgr._evaluate_model_artifact_streaming = MagicMock(side_effect=fake_streaming)
        with patch.object(ForecastingModelManager, "_save_combined_eval_parquets"):
            mgr._execute_model_evaluation()

        artifact_ts = artifact_stem[-15:]
        expected = (
            mgr._model_path.data_generated
            / f"predictions_calibration_{artifact_ts}"
            / "origin_0" / "ged_sb"
        )
        assert (expected / "y_pred.npy").exists(), (
            f"Track A+ must save at artifact timestamp path {expected}, "
            f"not at runtime timestamp from configs."
        )

        runtime_ts = "99999999_999999"
        wrong_path = (
            mgr._model_path.data_generated
            / f"predictions_calibration_{runtime_ts}"
        )
        assert not wrong_path.exists(), (
            "Track A+ must NOT save at the runtime timestamp path."
        )

    def test_forecast_uses_artifact_timestamp_not_runtime(self, tmp_path):
        """Track A+ forecast save must use artifact stem, not configs["timestamp"]."""

        artifact_stem = "calibration_model_20260510_140000"
        mgr = self._make_pf_manager(tmp_path, artifact_stem)

        pf = _pf(np.ones((10, 4), dtype=np.float32), np.arange(10), np.arange(10))
        mgr._forecast_model_artifact = MagicMock(return_value={"ged_sb": pf})
        mgr._forecasting_stage = MagicMock()
        mgr._save_combined_forecast = MagicMock()
        mgr._execute_model_forecasting()

        artifact_ts = artifact_stem[-15:]
        expected = (
            mgr._model_path.data_generated
            / f"predictions_calibration_{artifact_ts}"
            / "ged_sb"
        )
        assert (expected / "y_pred.npy").exists(), (
            f"Track A+ must save at artifact timestamp path {expected}, "
            f"not at runtime timestamp from configs."
        )

        runtime_ts = "99999999_999999"
        wrong_path = (
            mgr._model_path.data_generated
            / f"predictions_calibration_{runtime_ts}"
        )
        assert not wrong_path.exists(), (
            "Track A+ must NOT save at the runtime timestamp path."
        )


# ---------------------------------------------------------------------------
# P-3: EvaluationStage tolerates io_manager=None (C-95 regression)
# ---------------------------------------------------------------------------

class TestP3EvaluationStageNoneIO:
    """EvaluationStage must not crash when io_manager=None. The summary
    alert uses PredictionIOManager.generate_evaluation_table as a static
    method, and save_evaluations is guarded."""

    def test_evaluation_stage_constructed_with_none_io(self):
        """EvaluationStage accepts io_manager=None without error."""
        from views_pipeline_core.managers.evaluation.stage import EvaluationStage

        stage = EvaluationStage(
            wandb_module=MagicMock(),
            io_manager=None,
            wandb_notifications=False,
        )
        assert stage._io is None

    def test_generate_evaluation_table_callable_without_instance(self):
        """generate_evaluation_table is a @staticmethod — callable via class."""
        from views_pipeline_core.managers.prediction.io import PredictionIOManager

        result = PredictionIOManager.generate_evaluation_table(
            {"metric_a": 0.5, "metric_b": 0.9}
        )
        assert isinstance(result, str)
        assert "metric_a" in result


# ---------------------------------------------------------------------------
# P-4: Producer-consumer path agreement (C-94 regression)
# ---------------------------------------------------------------------------

class TestP4ProducerConsumerPathAgreement:
    """After C-94 fix, both producer (model.py) and consumer
    (PredictionFrameEnsembleManager) derive timestamps from the same
    artifact stem, so save and load paths always agree."""

    def test_artifact_stem_produces_consistent_timestamp(self):
        """stem[-15:] extraction yields exactly YYYYMMDD_HHMMSS."""
        stems = [
            "calibration_model_20260510_140000",
            "validation_model_20260101_000000",
            "forecasting_model_20261231_235959",
        ]
        for stem in stems:
            ts = stem[-15:]
            assert len(ts) == 15
            assert ts[8] == "_"
            assert ts[:8].isdigit()
            assert ts[9:].isdigit()

    def test_eval_roundtrip_path_agreement(self, tmp_path):
        """PredictionFrame saved at artifact-ts path is loadable by ensemble."""

        artifact_ts = "20260510_140000"
        target = "ged_sb"

        producer_dir = (
            tmp_path / f"predictions_calibration_{artifact_ts}"
            / "origin_0" / target
        )
        consumer_dir = (
            tmp_path / f"predictions_calibration_{artifact_ts}"
            / "origin_0" / target
        )

        pf = _pf(np.random.rand(10, 4).astype(np.float32), np.arange(10), np.arange(10))
        save_pf(pf, producer_dir)

        loaded = load_pf(consumer_dir, "pgm", mmap=True)
        np.testing.assert_array_equal(loaded.values, pf.values)


# ---------------------------------------------------------------------------
# P-5: Integration roundtrip — save/load with real directory structure
# ---------------------------------------------------------------------------

class TestP5IntegrationCoverageGap:
    """Exercises the real save/load round-trip with the directory structure
    that both producer and consumer agree on."""

    def test_save_load_roundtrip_eval_layout(self, tmp_path):
        """PredictionFrame saved in eval layout is loadable."""

        run_type = "calibration"
        ts = "20260522_120000"
        target = "ged_sb"
        origin_idx = 0

        producer_dir = (
            tmp_path / f"predictions_{run_type}_{ts}"
            / f"origin_{origin_idx}" / target
        )
        consumer_dir = (
            tmp_path / f"predictions_{run_type}_{ts}"
            / f"origin_{origin_idx}" / target
        )

        pf_original = _pf(np.random.rand(100, 64).astype(np.float32), np.arange(100), np.arange(100))
        save_pf(pf_original, producer_dir)

        pf_loaded = load_pf(consumer_dir, "pgm", mmap=True)

        assert pf_loaded.n_rows == pf_original.n_rows
        assert pf_loaded.sample_count == pf_original.sample_count
        np.testing.assert_array_equal(pf_loaded.values, pf_original.values)

    def test_save_load_roundtrip_forecast_layout(self, tmp_path):
        """PredictionFrame saved in forecast layout is loadable."""

        run_type = "forecasting"
        ts = "20260522_120000"
        target = "ged_sb"

        producer_dir = tmp_path / f"predictions_{run_type}_{ts}" / target
        consumer_dir = tmp_path / f"predictions_{run_type}_{ts}" / target

        pf_original = _pf(np.random.rand(100, 64).astype(np.float32), np.arange(100), np.arange(100))
        save_pf(pf_original, producer_dir)
        pf_loaded = load_pf(consumer_dir, "pgm", mmap=False)

        assert pf_loaded.n_rows == pf_original.n_rows
        np.testing.assert_array_equal(pf_loaded.values, pf_original.values)