"""
Falsification audit: PR #84 merge readiness (2026-05-23).

F-1: Docstring lie in _get_generated_pf_prediction_paths after C-96 fix.
F-3: Timestamp mismatch when artifact_name is non-latest.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# F-1: Docstring claims y_pred.npy check that no longer exists (C-96 fix)
# ---------------------------------------------------------------------------

class TestF1DocstringAccuracy:
    """The docstring of _get_generated_pf_prediction_paths must not claim
    that directories are filtered by y_pred.npy existence, because the C-96
    fix removed that check."""

    def test_docstring_does_not_mention_y_pred_existence_check(self):
        """Docstring must accurately reflect that discovery is by naming
        convention only, not by y_pred.npy existence."""
        from views_pipeline_core.data.model_path import ModelPathManager

        doc = ModelPathManager._get_generated_pf_prediction_paths.__doc__
        assert "contain y_pred.npy" not in doc, (
            "Docstring still claims the method checks for y_pred.npy existence, "
            "but that check was removed in C-96 fix. Documentation lie."
        )


# ---------------------------------------------------------------------------
# F-3: Timestamp from latest artifact vs named artifact
# ---------------------------------------------------------------------------

class TestF3ArtifactTimestampAgreement:
    """When a user passes --artifact_name pointing to a non-latest artifact,
    the PF save path timestamp must match the artifact actually evaluated,
    not the latest artifact."""

    def _make_manager_with_two_artifacts(self, tmp_path):
        """Build a ForecastingModelManager whose model_path has two artifacts:
        an old one and a latest one."""
        from views_pipeline_core.managers.model.model import ForecastingModelManager

        mock_path = MagicMock()
        mock_path.data_generated = tmp_path / "data" / "generated"
        mock_path.data_generated.mkdir(parents=True)
        mock_path.root = tmp_path
        mock_path.target = "model"
        mock_path._target = "model"
        mock_path.model_name = "test_model"

        old_artifact = Path("calibration_model_20260101_120000.pt")
        latest_artifact = Path("calibration_model_20260510_140000.pt")
        mock_path.get_latest_model_artifact_path.return_value = latest_artifact
        mock_path.artifacts.__truediv__ = lambda self, name: Path(name)

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
        mgr._args.artifact_name = "calibration_model_20260101_120000.pt"
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
        return mgr, old_artifact, latest_artifact

    @pytest.mark.xfail(reason="C-99: get_latest_model_artifact_path ignores artifact_name", strict=True)
    def test_forecast_save_uses_named_artifact_timestamp(self, tmp_path):
        """When --artifact_name names a non-latest artifact, PF forecast save
        path must use that artifact's timestamp, not the latest's.

        Currently FAILS: model.py always calls get_latest_model_artifact_path()
        regardless of args.artifact_name, so saves use latest timestamp."""
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        mgr, old_artifact, latest_artifact = self._make_manager_with_two_artifacts(tmp_path)

        pf = PredictionFrame(
            y_pred=np.ones((10, 4), dtype=np.float32),
            identifiers={"time": np.arange(10), "unit": np.arange(10)},
        )
        mgr._forecast_model_artifact = MagicMock(return_value={"ged_sb": pf})
        mgr._forecasting_stage = MagicMock()
        mgr._execute_model_forecasting()

        named_ts = old_artifact.stem[-15:]
        latest_ts = latest_artifact.stem[-15:]
        named_path = (
            mgr._model_path.data_generated
            / f"predictions_calibration_{named_ts}"
            / "ged_sb"
        )
        assert (named_path / "y_pred.npy").exists(), (
            f"When --artifact_name specifies a non-latest artifact, "
            f"PF save path must use its timestamp ({named_ts}), "
            f"not the latest artifact's timestamp ({latest_ts})."
        )
