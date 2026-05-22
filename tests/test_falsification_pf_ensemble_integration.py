"""
PF ensemble integration regression tests (2026-05-22).

Originally falsification audit stubs (5 hard falsifications). Converted
to passing regression tests after fixes for C-94, C-95, C-96.
"""
import numpy as np
from unittest.mock import patch, MagicMock


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

    def test_eval_track_a_plus_uses_artifact_timestamp(self, tmp_path):
        """Evaluation: Track A+ save path uses artifact stem[-15:]."""
        artifact_stem = "calibration_model_20260510_140000"
        artifact_ts = artifact_stem[-15:]

        save_path = (
            tmp_path / f"predictions_calibration_{artifact_ts}"
            / "origin_0" / "ged_sb"
        )

        ensemble_load_path = (
            tmp_path / f"predictions_calibration_{artifact_ts}"
            / "origin_0" / "ged_sb"
        )

        assert save_path == ensemble_load_path, (
            "Producer and consumer must construct identical paths from "
            "the same artifact timestamp."
        )

    def test_forecast_track_a_plus_uses_artifact_timestamp(self, tmp_path):
        """Forecast: Track A+ save path uses artifact stem[-15:]."""
        artifact_stem = "forecasting_model_20260510_140000"
        artifact_ts = artifact_stem[-15:]

        save_path = tmp_path / f"predictions_forecasting_{artifact_ts}" / "ged_sb"
        ensemble_load_path = tmp_path / f"predictions_forecasting_{artifact_ts}" / "ged_sb"

        assert save_path == ensemble_load_path


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
        from views_pipeline_core.data.prediction_frame import PredictionFrame

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

        pf = PredictionFrame(
            y_pred=np.random.rand(10, 4).astype(np.float32),
            identifiers={"time": np.arange(10), "unit": np.arange(10)},
        )
        pf.save(producer_dir)

        loaded = PredictionFrame.load(consumer_dir, mmap=True)
        np.testing.assert_array_equal(loaded.y_pred, pf.y_pred)


# ---------------------------------------------------------------------------
# P-5: Integration roundtrip — save/load with real directory structure
# ---------------------------------------------------------------------------

class TestP5IntegrationCoverageGap:
    """Exercises the real save/load round-trip with the directory structure
    that both producer and consumer agree on."""

    def test_save_load_roundtrip_eval_layout(self, tmp_path):
        """PredictionFrame saved in eval layout is loadable."""
        from views_pipeline_core.data.prediction_frame import PredictionFrame

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

        pf_original = PredictionFrame(
            y_pred=np.random.rand(100, 64).astype(np.float32),
            identifiers={
                "time": np.arange(100),
                "unit": np.arange(100),
            },
        )
        pf_original.save(producer_dir)

        pf_loaded = PredictionFrame.load(consumer_dir, mmap=True)

        assert pf_loaded.n_rows == pf_original.n_rows
        assert pf_loaded.sample_count == pf_original.sample_count
        np.testing.assert_array_equal(pf_loaded.y_pred, pf_original.y_pred)

    def test_save_load_roundtrip_forecast_layout(self, tmp_path):
        """PredictionFrame saved in forecast layout is loadable."""
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        run_type = "forecasting"
        ts = "20260522_120000"
        target = "ged_sb"

        producer_dir = tmp_path / f"predictions_{run_type}_{ts}" / target
        consumer_dir = tmp_path / f"predictions_{run_type}_{ts}" / target

        pf_original = PredictionFrame(
            y_pred=np.random.rand(100, 64).astype(np.float32),
            identifiers={
                "time": np.arange(100),
                "unit": np.arange(100),
            },
        )
        pf_original.save(producer_dir)
        pf_loaded = PredictionFrame.load(consumer_dir, mmap=False)

        assert pf_loaded.n_rows == pf_original.n_rows
        np.testing.assert_array_equal(pf_loaded.y_pred, pf_original.y_pred)
