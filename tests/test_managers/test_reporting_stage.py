"""
Tests for ReportingStage — ADR-045 E3 Stage pattern for report generation.

These tests verify that ReportingStage:
1. Receives an explicit ReportingContext (not self)
2. Dispatches correctly between model and ensemble paths
3. Delegates to ForecastReportTemplate / EvaluationReportTemplate
4. Sends WandB alerts on success
5. Raises appropriate errors for missing data
"""
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("views_reporting")

sys.modules["wandb"] = MagicMock()
sys.modules["views_evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation.evaluation_frame"] = MagicMock()
sys.modules["art"] = MagicMock()

from views_pipeline_core.managers.reporting.stage import (  # noqa: E402
    ReportingContext,
    ReportingStage,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_context(**overrides):
    """Build a minimal ReportingContext with sensible defaults."""
    defaults = dict(
        configs={
            "name": "test_model",
            "targets": ["lr_sb"],
            "run_type": "calibration",
            "timestamp": "20260101",
            "level": "cm",
        },
        model_path=MagicMock(),
        run_type="calibration",
        entity="views_pipeline",
    )
    defaults.update(overrides)
    # Set up model_path defaults
    mp = defaults["model_path"]
    mp.model_name = "test_model"
    mp.target = "model"
    mp.models = Path("/tmp/models")
    mp.reports = Path("/tmp/reports")
    mp._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    mp._get_generated_predictions_data_file_paths.return_value = [
        Path("predictions.parquet")
    ]
    return ReportingContext(**defaults)


def _make_stage():
    """Build a ReportingStage with mocked collaborators."""
    return ReportingStage(
        wandb_module=MagicMock(),
        wandb_notifications=False,
    )


# ── GREEN: Context is frozen ────────────────────────────────────────────────


class TestReportingContext:
    """ReportingContext must be an immutable value object."""

    def test_context_is_frozen(self):
        ctx = _make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.run_type = "validation"

    def test_context_fields_accessible(self):
        ctx = _make_context(run_type="calibration", entity="views_pipeline")
        assert ctx.run_type == "calibration"
        assert ctx.entity == "views_pipeline"
        assert "lr_sb" in ctx.configs["targets"]

    def test_context_inherits_base_stage_context(self):
        from views_pipeline_core.types import BaseStageContext
        assert issubclass(ReportingContext, BaseStageContext)

    def test_context_has_entity_field(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(ReportingContext)}
        assert "entity" in field_names


# ── GREEN: Forecast report — model path ─────────────────────────────────────


class TestForecastReportModel:
    """Verify forecast report generation for single-model path."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    @patch(
        "views_reporting.templates.reports.forecast.ForecastReportTemplate"
    )
    def test_model_path_generates_report(self, mock_template_cls, mock_read):
        """Single model: loads historical from own raw data, delegates to template."""
        import pandas as pd

        stage = _make_stage()
        ctx = _make_context()
        ctx.model_path.target = "model"

        mock_read.return_value = pd.DataFrame({"lr_sb": [0.1]})
        mock_template_cls.return_value.generate.return_value = Path(
            "/tmp/report.html"
        )

        result = stage.generate_forecast_report(ctx)

        assert result == Path("/tmp/report.html")
        mock_template_cls.assert_called_once_with(
            config=ctx.configs,
            model_path=ctx.model_path,
            run_type=ctx.run_type,
        )
        mock_template_cls.return_value.generate.assert_called_once()

    @patch("views_pipeline_core.files.utils.read_dataframe")
    @patch(
        "views_reporting.templates.reports.forecast.ForecastReportTemplate"
    )
    def test_model_path_sends_wandb_alert(self, mock_template_cls, mock_read):
        """Forecast report must send WandB alert on success."""
        import pandas as pd

        stage = _make_stage()
        ctx = _make_context()
        ctx.model_path.target = "model"

        mock_read.return_value = pd.DataFrame({"lr_sb": [0.1]})
        mock_template_cls.return_value.generate.return_value = Path(
            "/tmp/report.html"
        )

        stage.generate_forecast_report(ctx)

        stage._wandb_module.send_alert.assert_called_once()
        call_kwargs = stage._wandb_module.send_alert.call_args[1]
        assert "Forecast Report Generated" in call_kwargs["title"]


# ── GREEN: Forecast report — ensemble path ──────────────────────────────────


class TestForecastReportEnsemble:
    """Verify forecast report generation for ensemble path."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    @patch("views_pipeline_core.managers.model.ModelManager")
    @patch("views_pipeline_core.managers.model.ModelPathManager")
    @patch(
        "views_reporting.templates.reports.forecast.ForecastReportTemplate"
    )
    def test_ensemble_path_loads_from_sub_models(
        self, mock_template_cls, mock_mpm_cls, mock_mm_cls, mock_read,
    ):
        """Ensemble: creates ModelPathManager per sub-model to load historical data."""
        import pandas as pd

        stage = _make_stage()
        ctx = _make_context(
            configs={
                "name": "test_ensemble",
                "targets": ["lr_sb"],
                "models": ["purple_alien", "blue_whale"],
                "run_type": "calibration",
            },
        )
        ctx.model_path.target = "ensemble"

        # Mock sub-model path managers
        mock_sub_mp = MagicMock()
        mock_sub_mp._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
        mock_mpm_cls.return_value = mock_sub_mp

        # Mock sub-model configs
        mock_mm_cls.return_value.configs = {"targets": ["lr_sb"]}

        idx = pd.RangeIndex(10)
        mock_read.return_value = pd.DataFrame({"lr_sb": range(10)}, index=idx)
        mock_template_cls.return_value.generate.return_value = Path(
            "/tmp/ensemble_report.html"
        )

        result = stage.generate_forecast_report(ctx)

        assert result == Path("/tmp/ensemble_report.html")
        # ModelPathManager constructed once per sub-model
        assert mock_mpm_cls.call_count == 2
        mock_mpm_cls.assert_any_call(model_path="purple_alien", validate=True)
        mock_mpm_cls.assert_any_call(model_path="blue_whale", validate=True)


# ── GREEN: Forecast report — missing data ───────────────────────────────────


class TestForecastReportErrors:
    """Verify error handling for missing forecast data."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    def test_missing_forecast_data_raises_file_not_found(self, mock_read):
        """Missing forecast dataframe must raise FileNotFoundError."""
        import pandas as pd

        stage = _make_stage()
        ctx = _make_context()
        ctx.model_path.target = "model"

        # First call (historical) succeeds, second call (forecast) fails
        mock_read.side_effect = [
            pd.DataFrame({"lr_sb": [0.1]}),
            FileNotFoundError("not found"),
        ]

        with pytest.raises(FileNotFoundError, match="Forecast dataframe"):
            stage.generate_forecast_report(ctx)

    def test_invalid_target_type_raises_value_error(self):
        """Invalid target value must raise ValueError."""
        stage = _make_stage()
        ctx = _make_context()
        ctx.model_path.target = "invalid"

        with pytest.raises(ValueError, match="Invalid target type"):
            stage.generate_forecast_report(ctx)


# ── GREEN: Evaluation report ────────────────────────────────────────────────


class TestEvaluationReport:
    """Verify evaluation report generation."""

    @patch("views_pipeline_core.modules.wandb.get_latest_run")
    @patch(
        "views_reporting.templates.reports.evaluation.EvaluationReportTemplate"
    )
    def test_evaluation_report_calls_template_per_target(
        self, mock_template_cls, mock_get_run,
    ):
        """Must create one EvaluationReportTemplate per target."""
        stage = _make_stage()
        ctx = _make_context(
            configs={
                "name": "test",
                "targets": ["lr_sb", "lr_ns"],
                "run_type": "calibration",
            },
        )

        mock_get_run.return_value = MagicMock()
        mock_template_cls.return_value.generate.return_value = Path(
            "/tmp/eval_report.html"
        )

        stage.generate_evaluation_report(ctx)

        # Template instantiated twice — once per target
        assert mock_template_cls.call_count == 2
        # get_latest_run called with correct entity
        mock_get_run.assert_called_once_with(
            entity="views_pipeline",
            model_name="test_model",
            run_type="calibration",
        )

    @patch("views_pipeline_core.modules.wandb.get_latest_run")
    @patch(
        "views_reporting.templates.reports.evaluation.EvaluationReportTemplate"
    )
    def test_evaluation_report_sends_wandb_alert(
        self, mock_template_cls, mock_get_run,
    ):
        """Evaluation report must send WandB alert on success."""
        stage = _make_stage()
        ctx = _make_context()

        mock_get_run.return_value = MagicMock()
        mock_template_cls.return_value.generate.return_value = Path(
            "/tmp/eval_report.html"
        )

        stage.generate_evaluation_report(ctx)

        stage._wandb_module.send_alert.assert_called_once()
        call_kwargs = stage._wandb_module.send_alert.call_args[1]
        assert "Evaluation Report Generated" in call_kwargs["title"]
