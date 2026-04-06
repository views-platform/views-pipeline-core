"""
Tests for ForecastingStage — ADR-045 E4 Stage pattern for forecast post-processing.

These tests verify that ForecastingStage:
1. Receives an explicit ForecastingContext (not self)
2. Enforces type contracts (ADR-042) for both PF and DF paths
3. Validates DF predictions via CorePredictionSniffer
4. Persists predictions via io_manager
5. Creates execution logs and sends WandB alerts
"""
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.modules["wandb"] = MagicMock()
sys.modules["views_evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation.evaluation_frame"] = MagicMock()
sys.modules["art"] = MagicMock()

from views_pipeline_core.managers.forecasting.stage import (  # noqa: E402
    ForecastingContext,
    ForecastingStage,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_context(**overrides):
    """Build a minimal ForecastingContext with sensible defaults."""
    defaults = dict(
        configs={
            "name": "test_model",
            "targets": ["lr_sb"],
            "level": "cm",
            "timestamp": "20260101",
        },
        model_path=MagicMock(),
        run_type="forecasting",
        prediction_format="dataframe",
    )
    defaults.update(overrides)
    mp = defaults["model_path"]
    mp.model_name = "test_model"
    mp.target = "model"
    mp._target = "model"
    mp.data_generated = Path("/tmp/data_generated")
    return ForecastingContext(**defaults)


def _make_stage():
    """Build a ForecastingStage with mocked collaborators."""
    return ForecastingStage(
        wandb_module=MagicMock(),
        io_manager=MagicMock(),
        wandb_notifications=False,
    )


def _make_pred_df():
    """Create a simple prediction DataFrame."""
    idx = pd.MultiIndex.from_tuples(
        [(500, 1), (500, 2)], names=["month_id", "priogrid_gid"],
    )
    return pd.DataFrame({"pred_lr_sb": [0.1, 0.2]}, index=idx)


# ── Context tests ───────────────────────────────────────────────────────────


class TestForecastingContext:
    """ForecastingContext must be an immutable value object."""

    def test_context_is_frozen(self):
        ctx = _make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.prediction_format = "prediction_frame"

    def test_context_inherits_base_stage_context(self):
        from views_pipeline_core.types import BaseStageContext
        assert issubclass(ForecastingContext, BaseStageContext)

    def test_context_has_prediction_format_field(self):
        ctx = _make_context(prediction_format="prediction_frame")
        assert ctx.prediction_format == "prediction_frame"


# ── DF path tests ───────────────────────────────────────────────────────────


class TestDFPath:
    """Verify DataFrame (legacy) forecast path."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch(
        "views_pipeline_core.modules.validation.core_prediction_sniffer"
        ".CorePredictionSniffer"
    )
    def test_df_path_validates_via_sniffer(self, mock_sniffer_cls, mock_log):
        """DF path must call CorePredictionSniffer.sniff_predictions."""
        stage = _make_stage()
        ctx = _make_context(prediction_format="dataframe")
        predictions = _make_pred_df()

        stage.process_and_save_forecast(predictions, ctx)

        mock_sniffer_cls.assert_called_once_with(level="cm")
        mock_sniffer_cls.return_value.sniff_predictions.assert_called_once()

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch(
        "views_pipeline_core.modules.validation.core_prediction_sniffer"
        ".CorePredictionSniffer"
    )
    def test_df_path_saves_predictions(self, mock_sniffer_cls, mock_log):
        """DF path must save predictions via io_manager."""
        stage = _make_stage()
        ctx = _make_context(prediction_format="dataframe")
        predictions = _make_pred_df()

        stage.process_and_save_forecast(predictions, ctx)

        stage._io.save_predictions.assert_called_once()

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_df_path_type_enforcement_rejects_dict(self, mock_log):
        """DF path: dict return must raise ValueError (ADR-042)."""
        stage = _make_stage()
        ctx = _make_context(prediction_format="dataframe")

        with pytest.raises(ValueError, match="prediction_format='dataframe'"):
            stage.process_and_save_forecast({"lr_sb": MagicMock()}, ctx)


# ── PF path tests ───────────────────────────────────────────────────────────


class TestPFPath:
    """Verify PredictionFrame forecast path."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch(
        "views_pipeline_core.managers.prediction.prediction_frame_converter"
        ".PredictionFrameConverter"
    )
    def test_pf_path_converts_and_saves_per_target(
        self, mock_converter_cls, mock_log,
    ):
        """PF path must convert each target and save via io_manager."""
        stage = _make_stage()
        ctx = _make_context(prediction_format="prediction_frame")

        mock_pf = MagicMock()
        predictions = {"lr_sb": mock_pf, "lr_ns": MagicMock()}

        mock_converter_cls.return_value.to_prediction_df.return_value = _make_pred_df()

        stage.process_and_save_forecast(predictions, ctx)

        # Converted once per target
        assert mock_converter_cls.return_value.to_prediction_df.call_count == 2
        # Saved once per target
        assert stage._io.save_predictions.call_count == 2

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_pf_path_type_enforcement_rejects_non_dict(self, mock_log):
        """PF path: non-dict return must raise ValueError (ADR-042)."""
        stage = _make_stage()
        ctx = _make_context(prediction_format="prediction_frame")

        with pytest.raises(ValueError, match="prediction_format='prediction_frame'"):
            stage.process_and_save_forecast(_make_pred_df(), ctx)


# ── Cross-cutting tests ────────────────────────────────────────────────────


class TestCrossCutting:
    """Verify log creation and WandB alerts."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch(
        "views_pipeline_core.modules.validation.core_prediction_sniffer"
        ".CorePredictionSniffer"
    )
    def test_creates_execution_log(self, mock_sniffer_cls, mock_log):
        """Must call handle_single_log_creation after save."""
        stage = _make_stage()
        ctx = _make_context()
        predictions = _make_pred_df()

        stage.process_and_save_forecast(predictions, ctx)

        mock_log.assert_called_once_with(
            model_path=ctx.model_path,
            config=ctx.configs,
            train=False,
        )

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    @patch(
        "views_pipeline_core.modules.validation.core_prediction_sniffer"
        ".CorePredictionSniffer"
    )
    def test_sends_wandb_alert(self, mock_sniffer_cls, mock_log):
        """Must send WandB completion alert."""
        stage = _make_stage()
        ctx = _make_context()
        predictions = _make_pred_df()

        stage.process_and_save_forecast(predictions, ctx)

        stage._wandb_module.send_alert.assert_called_once()
        call_kwargs = stage._wandb_module.send_alert.call_args[1]
        assert "completed successfully" in call_kwargs["title"]
