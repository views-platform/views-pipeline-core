"""
Tests for TrainingStage — ADR-045 E5 Stage pattern for training post-processing.

These tests verify that TrainingStage:
1. Receives an explicit TrainingContext (not self)
2. Creates execution logs via handle_single_log_creation
3. Sends WandB completion alerts with sweep info
"""
import sys
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

sys.modules["wandb"] = MagicMock()
sys.modules["views_evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation.evaluation_frame"] = MagicMock()
sys.modules["art"] = MagicMock()

from views_pipeline_core.managers.training.stage import (  # noqa: E402
    TrainingContext,
    TrainingStage,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_context(**overrides):
    """Build a minimal TrainingContext with sensible defaults."""
    defaults = dict(
        configs={
            "name": "test_model",
            "targets": ["lr_sb"],
            "level": "cm",
        },
        model_path=MagicMock(),
        run_type="calibration",
        sweep=False,
    )
    defaults.update(overrides)
    mp = defaults["model_path"]
    mp.model_name = "test_model"
    mp.target = "model"
    return TrainingContext(**defaults)


def _make_stage():
    """Build a TrainingStage with mocked collaborators."""
    return TrainingStage(
        wandb_module=MagicMock(),
        wandb_notifications=False,
    )


# ── Context tests ───────────────────────────────────────────────────────────


class TestTrainingContext:
    """TrainingContext must be an immutable value object."""

    def test_context_is_frozen(self):
        ctx = _make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.sweep = True

    def test_context_inherits_base_stage_context(self):
        from views_pipeline_core.types import BaseStageContext
        assert issubclass(TrainingContext, BaseStageContext)

    def test_context_has_sweep_field(self):
        ctx = _make_context(sweep=True)
        assert ctx.sweep is True


# ── Finalize training tests ─────────────────────────────────────────────────


class TestFinalizeTraining:
    """Verify post-training bookkeeping."""

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_creates_execution_log(self, mock_log):
        """Must call handle_single_log_creation with train=True."""
        stage = _make_stage()
        ctx = _make_context()

        stage.finalize_training(ctx)

        mock_log.assert_called_once_with(
            model_path=ctx.model_path,
            config=ctx.configs,
            train=True,
        )

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_sends_wandb_alert(self, mock_log):
        """Must send WandB completion alert."""
        stage = _make_stage()
        ctx = _make_context()

        stage.finalize_training(ctx)

        stage._wandb_module.send_alert.assert_called_once()
        call_kwargs = stage._wandb_module.send_alert.call_args[1]
        assert "completed successfully" in call_kwargs["title"]

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_alert_text_includes_sweep_flag(self, mock_log):
        """Alert text must indicate sweep status."""
        stage = _make_stage()
        ctx = _make_context(sweep=True)

        stage.finalize_training(ctx)

        call_kwargs = stage._wandb_module.send_alert.call_args[1]
        assert "Sweep: True" in call_kwargs["text"]

    @patch("views_pipeline_core.files.utils.handle_single_log_creation")
    def test_non_sweep_alert_text(self, mock_log):
        """Non-sweep alert text must show Sweep: False."""
        stage = _make_stage()
        ctx = _make_context(sweep=False)

        stage.finalize_training(ctx)

        call_kwargs = stage._wandb_module.send_alert.call_args[1]
        assert "Sweep: False" in call_kwargs["text"]