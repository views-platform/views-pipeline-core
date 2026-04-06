"""
Tests for EvaluationStage — first implementation of ADR-045 Stage pattern.

These tests verify that EvaluationStage:
1. Receives an explicit EvaluationContext (not self)
2. Delegates to NativeEvaluator via EvaluationAdapter
3. Publishes results to WandB and disk via PredictionIOManager
4. Handles edge cases (empty targets, missing columns)
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

from views_pipeline_core.managers.evaluation.stage import (  # noqa: E402
    EvaluationContext,
    EvaluationStage,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_context(**overrides):
    """Build a minimal EvaluationContext with sensible defaults."""
    defaults = dict(
        configs={
            "regression_targets": ["lr_sb"],
            "classification_targets": [],
            "regression_point_metrics": ["MSE"],
            "steps": list(range(1, 37)),
            "sweep": False,
            "run_type": "calibration",
        },
        model_path=MagicMock(),
        prediction_format="dataframe",
        partition_dict={"calibration": {"train": (121, 444), "test": (445, 492)}},
        run_type="calibration",
        data_loader=MagicMock(),
        prepare_actuals_df=lambda df: df,
    )
    defaults.update(overrides)
    return EvaluationContext(**defaults)


def _make_stage():
    """Build an EvaluationStage with mocked collaborators."""
    return EvaluationStage(
        wandb_module=MagicMock(),
        io_manager=MagicMock(),
    )


def _make_mock_report():
    """Mock NativeEvaluator.evaluate() return."""
    report = MagicMock()
    report.to_dict.return_value = {
        "target": "lr_sb", "task": "regression", "pred_type": "point",
        "schemas": {"step": {}, "time_series": {}, "month": {}},
    }
    report.to_dataframe.return_value = pd.DataFrame()
    return report


# ── GREEN: Context is frozen ─────────────────────────────────────────────────


class TestEvaluationContext:
    """EvaluationContext must be an immutable value object."""

    def test_context_is_frozen(self):
        ctx = _make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.run_type = "validation"

    def test_context_fields_accessible(self):
        ctx = _make_context(run_type="calibration")
        assert ctx.run_type == "calibration"
        assert ctx.prediction_format == "dataframe"
        assert "lr_sb" in ctx.configs["regression_targets"]


# ── GREEN: Stage receives context, not self ──────────────────────────────────


class TestEvaluationStageContract:
    """EvaluationStage must operate on explicit context, not god-class internals."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    def test_evaluate_calls_native_evaluator(self, mock_read):
        """Stage must delegate metric computation to NativeEvaluator."""
        stage = _make_stage()
        ctx = _make_context()
        ctx.model_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        mock_read.return_value = pd.DataFrame(
            {"lr_sb": [0.1]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )
        mock_report = _make_mock_report()
        eval_mod = sys.modules["views_evaluation"]
        eval_mod.NativeEvaluator.return_value.evaluate.return_value = mock_report

        df_pred = pd.DataFrame(
            {"pred_lr_sb": [0.5]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )

        stage.evaluate([df_pred], ctx)

        eval_mod.NativeEvaluator.assert_called_once_with(ctx.configs)
        assert eval_mod.NativeEvaluator.return_value.evaluate.call_count == 1

    @patch("views_pipeline_core.files.utils.read_dataframe")
    def test_evaluate_logs_to_wandb(self, mock_read):
        """Stage must log evaluation results to WandB."""
        stage = _make_stage()
        ctx = _make_context()
        ctx.model_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        mock_read.return_value = pd.DataFrame(
            {"lr_sb": [0.1]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )
        eval_mod = sys.modules["views_evaluation"]
        eval_mod.NativeEvaluator.return_value.evaluate.return_value = _make_mock_report()

        df_pred = pd.DataFrame(
            {"pred_lr_sb": [0.5]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )

        stage.evaluate([df_pred], ctx)

        stage._wandb_module.log_evaluation_results.assert_called_once()

    @patch("views_pipeline_core.files.utils.read_dataframe")
    def test_evaluate_saves_to_disk(self, mock_read):
        """Stage must save evaluation DataFrames via io_manager when not sweeping."""
        stage = _make_stage()
        ctx = _make_context()
        ctx.model_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        mock_read.return_value = pd.DataFrame(
            {"lr_sb": [0.1]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )
        eval_mod = sys.modules["views_evaluation"]
        eval_mod.NativeEvaluator.return_value.evaluate.return_value = _make_mock_report()

        df_pred = pd.DataFrame(
            {"pred_lr_sb": [0.5]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )

        stage.evaluate([df_pred], ctx)

        stage._io.save_evaluations.assert_called_once()

    def test_evaluate_with_empty_targets_is_noop(self):
        """Empty target lists should complete without error."""
        stage = _make_stage()
        ctx = _make_context(configs={
            "regression_targets": [],
            "classification_targets": [],
            "sweep": False,
        })

        with patch("views_pipeline_core.files.utils.read_dataframe"):
            ctx.model_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
            # Should not crash, just do nothing
            stage.evaluate([], ctx)

        stage._wandb_module.log_evaluation_results.assert_not_called()
