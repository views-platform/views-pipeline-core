"""Tests for ForecastingStage — simplified for unified path.

The stage is now a thin backward-compat shim. The unified forecast
persistence logic lives in ForecastingMixin._save_combined_forecast().
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from views_pipeline_core.managers.forecasting.stage import (
    ForecastingStage,
    ForecastingContext,
)


class TestForecastingStageContract:
    """Stage construction and context contract."""

    def test_stage_constructs_with_collaborators(self):
        stage = ForecastingStage(
            wandb_module=MagicMock(),
            io_manager=MagicMock(),
            wandb_notifications=False,
            savers=[],
        )
        assert stage._wandb_module is not None
        assert stage._io is not None
        assert stage._savers == []

    def test_process_and_save_forecast_is_deprecated(self):
        """The unified path handles persistence directly."""
        stage = ForecastingStage(
            wandb_module=MagicMock(),
            io_manager=MagicMock(),
        )
        ctx = ForecastingContext(
            configs={"name": "test"},
            model_path=MagicMock(),
            run_type="calibration",
            prediction_format="dataframe",
        )
        # Should not raise — just log a warning
        stage.process_and_save_forecast(MagicMock(), ctx)
