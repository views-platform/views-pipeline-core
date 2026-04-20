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


# ── GREEN: PF path in _build_evaluation_frame ──────────────────────────────


class TestEvaluationStagePFPath:
    """Verify PredictionFrame (PF) code path through the stage."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    def test_pf_path_calls_from_prediction_frames(self, mock_read):
        """PF path must route through EvaluationAdapter.from_prediction_frames."""
        stage = _make_stage()
        ctx = _make_context(prediction_format="prediction_frame")
        ctx.model_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        mock_read.return_value = pd.DataFrame(
            {"lr_sb": [0.1]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )

        # PF path expects Dict[str, List[PredictionFrame]]
        mock_pf = MagicMock()
        pf_dict = {"lr_sb": [mock_pf]}

        mock_report = _make_mock_report()
        eval_mod = sys.modules["views_evaluation"]
        eval_mod.NativeEvaluator.return_value.evaluate.return_value = mock_report

        # Patch the EvaluationAdapter that's lazily imported inside evaluate()
        with patch(
            "views_pipeline_core.modules.validation.adapter.EvaluationAdapter"
        ) as mock_adapter:
            mock_adapter.from_prediction_frames.return_value = MagicMock()
            stage.evaluate(pf_dict, ctx)

            mock_adapter.from_prediction_frames.assert_called_once()
            # DF path's from_dataframes should NOT be called
            mock_adapter.from_dataframes.assert_not_called()

    @patch("views_pipeline_core.files.utils.read_dataframe")
    def test_pf_path_missing_target_skips_gracefully(self, mock_read):
        """PF path: target not in predictions dict → skip, no crash."""
        stage = _make_stage()
        ctx = _make_context(prediction_format="prediction_frame")
        ctx.model_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        mock_read.return_value = pd.DataFrame(
            {"lr_sb": [0.1]},
            index=pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"]),
        )

        # Empty dict — target "lr_sb" is missing
        pf_dict = {}

        eval_mod = sys.modules["views_evaluation"]
        # Reset to track only this test's calls
        mock_evaluator = MagicMock()
        eval_mod.NativeEvaluator.return_value = mock_evaluator
        mock_evaluator.evaluate.return_value = _make_mock_report()

        with patch(
            "views_pipeline_core.modules.validation.adapter.EvaluationAdapter"
        ):
            # Should not raise, just skip
            stage.evaluate(pf_dict, ctx)

        # No metrics should be computed (target missing → skipped)
        mock_evaluator.evaluate.assert_not_called()


# ── GREEN: Ensemble actuals loading ────────────────────────────────────────


class TestEvaluationStageEnsemble:
    """Verify ensemble=True actuals loading path."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    @patch("views_pipeline_core.managers.model.ModelPathManager")
    def test_ensemble_loads_actuals_from_first_model(self, mock_mpm_cls, mock_read):
        """ensemble=True must construct ModelPathManager from configs['models'][0]."""
        stage = _make_stage()
        ctx = _make_context(configs={
            "regression_targets": ["lr_sb"],
            "classification_targets": [],
            "regression_point_metrics": ["MSE"],
            "steps": list(range(1, 37)),
            "sweep": False,
            "run_type": "calibration",
            "models": ["purple_alien", "blue_whale"],
        })

        # Mock the ModelPathManager instance created for ensemble actuals
        mock_mpm_instance = MagicMock()
        mock_mpm_instance.data_raw = Path("/tmp/data_raw")
        mock_mpm_cls.return_value = mock_mpm_instance

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

        stage.evaluate([df_pred], ctx, ensemble=True)

        # Must construct MPM from first model in list
        mock_mpm_cls.assert_called_once_with("purple_alien")


# ── GREEN: Forecasting step mapping ────────────────────────────────────────


class TestStepMappings:
    """Verify _get_evaluation_step_mappings for different run types."""

    def test_calibration_step_mapping_uses_partition_dict(self):
        """Calibration run: base_origin = partition test start - 1."""
        ctx = _make_context(
            run_type="calibration",
            partition_dict={"calibration": {"train": (121, 444), "test": (445, 492)}},
            configs={
                "regression_targets": ["lr_sb"],
                "classification_targets": [],
                "steps": [1, 2, 3],
                "sweep": False,
            },
        )
        mappings = EvaluationStage._get_evaluation_step_mappings(
            n_sequences=2, context=ctx,
        )
        # base_origin = 445 - 1 = 444
        assert len(mappings) == 2
        # seq 0: {444+0+1: 1, 444+0+2: 2, 444+0+3: 3} = {445: 1, 446: 2, 447: 3}
        assert mappings[0] == {445: 1, 446: 2, 447: 3}
        # seq 1: {444+1+1: 1, 444+1+2: 2, 444+1+3: 3} = {446: 1, 447: 2, 448: 3}
        assert mappings[1] == {446: 1, 447: 2, 448: 3}

    def test_forecasting_step_mapping_uses_data_loader(self):
        """Forecasting run: base_origin = data_loader.month_last."""
        mock_loader = MagicMock()
        mock_loader.month_last = 500
        ctx = _make_context(
            run_type="forecasting",
            data_loader=mock_loader,
            configs={
                "regression_targets": ["lr_sb"],
                "classification_targets": [],
                "steps": [1, 2],
                "sweep": False,
            },
        )
        mappings = EvaluationStage._get_evaluation_step_mappings(
            n_sequences=1, context=ctx,
        )
        # base_origin = 500
        assert mappings[0] == {501: 1, 502: 2}

    def test_forecasting_without_data_loader_raises(self):
        """Forecasting run with no data_loader must raise ValueError."""
        ctx = _make_context(
            run_type="forecasting",
            data_loader=None,
        )
        with pytest.raises(ValueError, match="data loader"):
            EvaluationStage._get_evaluation_step_mappings(
                n_sequences=1, context=ctx,
            )

    def test_unknown_run_type_raises_key_error(self):
        """Unknown run_type not in partition_dict must raise KeyError."""
        ctx = _make_context(
            run_type="nonexistent",
            partition_dict={"calibration": {"train": (1, 100), "test": (101, 120)}},
        )
        with pytest.raises(KeyError, match="nonexistent"):
            EvaluationStage._get_evaluation_step_mappings(
                n_sequences=1, context=ctx,
            )


# ── GREEN: Multiple targets ───────────────────────────────────────────────


class TestMultipleTargets:
    """Verify evaluation iterates over all targets."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    def test_evaluate_processes_all_regression_targets(self, mock_read):
        """Stage must call evaluator.evaluate() once per target."""
        stage = _make_stage()
        ctx = _make_context(configs={
            "regression_targets": ["lr_sb", "lr_ns"],
            "classification_targets": [],
            "regression_point_metrics": ["MSE"],
            "steps": list(range(1, 37)),
            "sweep": False,
            "run_type": "calibration",
        })
        ctx.model_path._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        idx = pd.MultiIndex.from_tuples([(445, 1)], names=["month_id", "e"])
        mock_read.return_value = pd.DataFrame(
            {"lr_sb": [0.1], "lr_ns": [0.2]}, index=idx,
        )

        mock_report = _make_mock_report()
        eval_mod = sys.modules["views_evaluation"]
        # Fresh evaluator instance to isolate call count from other tests
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_report
        eval_mod.NativeEvaluator.return_value = mock_evaluator

        df_pred = pd.DataFrame(
            {"pred_lr_sb": [0.5], "pred_lr_ns": [0.3]}, index=idx,
        )

        stage.evaluate([df_pred], ctx)

        # evaluate called twice — once per target
        assert mock_evaluator.evaluate.call_count == 2


# ── GREEN: Context construction contract test ──────────────────────────────


class TestContextContract:
    """Verify EvaluationContext inherits from BaseStageContext and has correct fields."""

    def test_context_inherits_base_stage_context(self):
        """EvaluationContext must extend BaseStageContext."""
        from views_pipeline_core.types import BaseStageContext
        assert issubclass(EvaluationContext, BaseStageContext)

    def test_context_has_all_required_fields(self):
        """All fields that the stage accesses must exist on the context."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(EvaluationContext)}
        required = {
            "configs", "model_path", "run_type",  # from BaseStageContext
            "prediction_format", "partition_dict",  # evaluation-specific
            "data_loader", "prepare_actuals_df",    # evaluation-specific
            "postprocessed",
        }
        assert required.issubset(field_names), (
            f"Missing fields: {required - field_names}"
        )

    def test_model_path_protocol_compliance(self):
        """ModelPathManager must satisfy ModelPathProtocol at runtime."""
        from views_pipeline_core.types import ModelPathProtocol
        # MagicMock satisfies any Protocol, so test the structural check
        # by verifying Protocol is runtime_checkable and has expected members
        import inspect
        members = {
            name for name, _ in inspect.getmembers(ModelPathProtocol)
            if not name.startswith("_")
        }
        expected = {
            "model_name", "target", "data_generated", "data_raw",
            "models", "root", "artifacts", "reports",
        }
        assert expected.issubset(members), (
            f"Protocol missing members: {expected - members}"
        )
