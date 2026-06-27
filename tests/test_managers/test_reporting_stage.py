"""
Tests for ReportingStage — ADR-045 E3 Stage pattern for report generation.

These tests verify that ReportingStage:
1. Receives an explicit ReportingContext (not self)
2. Dispatches correctly between model and ensemble paths
3. Delegates to ForecastReportTemplate / EvaluationReportTemplate
4. Sends WandB alerts on success
5. Raises appropriate errors for missing data
"""
import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("views_reporting")

# The MetricFrame EvaluationSource consumer (views-reporting #173) may be absent in an older
# released views-reporting; the source-based eval-report tests skip there (the producer epic
# #224 integrates against the dev-installed consumer; publish is the final step).
_HAS_EVAL_SOURCE = importlib.util.find_spec("views_reporting.sources") is not None
if _HAS_EVAL_SOURCE:
    # Cache views_reporting.sources (real) once so @patch("...MetricFrameFileSource") resolves.
    # It imports views_evaluation.evaluation.metric_frame, but several other test modules poison
    # sys.modules["views_evaluation*"] with MagicMocks at collection; temporarily restore the
    # real package for this one import, then put their mocks back exactly as they were.
    _ve_saved = {
        k: sys.modules.pop(k)
        for k in [
            k for k in list(sys.modules)
            if k == "views_evaluation" or k.startswith("views_evaluation.")
        ]
    }
    try:
        import views_reporting.sources  # noqa: F401
    finally:
        sys.modules.update(_ve_saved)

from views_pipeline_core.managers.reporting.stage import (  # noqa: E402
    ReportingContext,
    ReportingStage,
)


@pytest.fixture(autouse=True, scope="module")
def _mock_optional_deps():
    """Mock heavy/optional deps the stage imports lazily (wandb, views_evaluation, art) for
    this module's tests only, restoring sys.modules on teardown so the mocks never leak into
    other modules (the reporting stage imports none of them at module load)."""
    names = [
        "wandb",
        "views_evaluation",
        "views_evaluation.evaluation",
        "views_evaluation.evaluation.evaluation_frame",
        "art",
    ]
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules[n] = MagicMock()
    yield
    for n, mod in saved.items():
        if mod is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = mod


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


# ── GREEN: Forecast report — PredictionFrame path ─────────────────────────


class TestForecastReportPredictionFrame:
    """Verify forecast report generation for PredictionFrame models."""

    @patch("views_pipeline_core.files.utils.read_dataframe")
    @patch(
        "views_reporting.templates.reports.forecast.ForecastReportTemplate"
    )
    def test_pf_path_passes_prediction_format_and_path(
        self, mock_template_cls, mock_read,
    ):
        """PF path: template.generate() receives prediction_format and prediction_path."""
        import pandas as pd

        stage = _make_stage()
        ctx = _make_context(prediction_format="prediction_frame")
        ctx.model_path.target = "model"
        ctx.model_path._get_generated_pf_prediction_paths.return_value = [
            Path("/tmp/predictions_forecasting_20260601")
        ]

        mock_read.return_value = pd.DataFrame({"lr_sb": [0.1]})
        mock_template_cls.return_value.generate.return_value = Path(
            "/tmp/report.html"
        )

        result = stage.generate_forecast_report(ctx)

        assert result == Path("/tmp/report.html")
        mock_read.assert_called_once()  # historical only — PF path skips read_dataframe for forecast
        mock_template_cls.return_value.generate.assert_called_once_with(
            historical_dataframe=mock_read.return_value,
            prediction_format="prediction_frame",
            prediction_path=Path("/tmp/predictions_forecasting_20260601"),
        )

    @patch("views_pipeline_core.files.utils.read_dataframe")
    @patch(
        "views_reporting.templates.reports.forecast.ForecastReportTemplate"
    )
    def test_pf_path_missing_predictions_raises_file_not_found(
        self, mock_template_cls, mock_read,
    ):
        """PF path: missing prediction directory raises FileNotFoundError."""
        import pandas as pd

        stage = _make_stage()
        ctx = _make_context(prediction_format="prediction_frame")
        ctx.model_path.target = "model"
        ctx.model_path._get_generated_pf_prediction_paths.return_value = []

        mock_read.return_value = pd.DataFrame({"lr_sb": [0.1]})

        with pytest.raises(FileNotFoundError, match="PredictionFrame"):
            stage.generate_forecast_report(ctx)

    @patch("views_pipeline_core.files.utils.read_dataframe")
    @patch(
        "views_reporting.templates.reports.forecast.ForecastReportTemplate"
    )
    def test_df_path_unchanged_with_default_format(
        self, mock_template_cls, mock_read,
    ):
        """Default prediction_format='dataframe' takes the existing parquet path."""
        import pandas as pd

        stage = _make_stage()
        ctx = _make_context()  # default prediction_format="dataframe"
        ctx.model_path.target = "model"

        mock_read.return_value = pd.DataFrame({"lr_sb": [0.1]})
        mock_template_cls.return_value.generate.return_value = Path(
            "/tmp/report.html"
        )

        stage.generate_forecast_report(ctx)

        mock_template_cls.return_value.generate.assert_called_once_with(
            forecast_dataframe=mock_read.return_value,
            historical_dataframe=mock_read.return_value,
        )


# ── GREEN: Evaluation report ────────────────────────────────────────────────


@pytest.mark.skipif(
    not _HAS_EVAL_SOURCE,
    reason="views-reporting lacks the EvaluationSource consumer (#173)",
)
class TestEvaluationReport:
    """Eval report renders from the persisted MetricFrame via MetricFrameFileSource
    (ADR-018 / C-108) — no render-time WandB scrape (the retired C-48/C-110 path)."""

    @patch("views_reporting.templates.reports.evaluation.EvaluationReportTemplate")
    @patch("views_reporting.sources.MetricFrameFileSource")
    def test_renders_from_source_at_locked_path_per_target(
        self, mock_source_cls, mock_template_cls,
    ):
        """One source + template per target; the source is built at the locked
        cross-repo path (root=<data_generated>, primary_model=<model>)."""
        stage = _make_stage()
        ctx = _make_context(
            configs={"name": "test", "targets": ["lr_sb", "lr_ns"], "run_type": "calibration"},
        )
        ctx.model_path.data_generated = Path("/tmp/dg")
        mock_source_cls.return_value.metric_frame.return_value = MagicMock()  # frame present
        mock_template_cls.return_value.generate.return_value = Path("/tmp/eval_report.html")

        stage.generate_evaluation_report(ctx)

        assert mock_source_cls.call_count == 2
        mock_source_cls.assert_any_call(
            root=Path("/tmp/dg"),
            run_type="calibration",
            target="lr_sb",
            primary_model="test_model",
        )
        assert mock_template_cls.call_count == 2
        mock_template_cls.return_value.generate.assert_called_with(
            source=mock_source_cls.return_value, target="lr_ns",
        )

    @patch("views_reporting.templates.reports.evaluation.EvaluationReportTemplate")
    @patch("views_reporting.sources.MetricFrameFileSource")
    def test_skips_target_without_persisted_frame(
        self, mock_source_cls, mock_template_cls,
    ):
        """No MetricFrame for the target -> skip it, render nothing, no alert."""
        stage = _make_stage()
        ctx = _make_context()
        mock_source_cls.return_value.metric_frame.return_value = None  # absent

        result = stage.generate_evaluation_report(ctx)

        assert result is None
        mock_template_cls.assert_not_called()
        stage._wandb_module.send_alert.assert_not_called()

    @patch("views_pipeline_core.modules.wandb.get_latest_run")
    @patch("views_reporting.templates.reports.evaluation.EvaluationReportTemplate")
    @patch("views_reporting.sources.MetricFrameFileSource")
    def test_no_wandb_scrape_on_eval_path(
        self, mock_source_cls, mock_template_cls, mock_get_run,
    ):
        """The render-time get_latest_run scrape is retired (C-48/C-110)."""
        stage = _make_stage()
        ctx = _make_context()
        mock_source_cls.return_value.metric_frame.return_value = MagicMock()
        mock_template_cls.return_value.generate.return_value = Path("/tmp/eval_report.html")

        stage.generate_evaluation_report(ctx)

        mock_get_run.assert_not_called()

    @patch("views_reporting.templates.reports.evaluation.EvaluationReportTemplate")
    @patch("views_reporting.sources.MetricFrameFileSource")
    def test_frame_load_error_propagates(
        self, mock_source_cls, mock_template_cls,
    ):
        """A present-but-unreadable MetricFrame fails loud — not silently skipped or
        degraded (Fail Loud; re-establishes the C-180 no-swallow contract for the
        MetricFrame path, replacing the retired get_latest_run transient-error test).
        Absent frame -> None -> skip; corrupt frame -> raise."""
        stage = _make_stage()
        ctx = _make_context()
        mock_source_cls.return_value.metric_frame.side_effect = OSError("corrupt frame")

        with pytest.raises(OSError):
            stage.generate_evaluation_report(ctx)

        mock_template_cls.assert_not_called()
        stage._wandb_module.send_alert.assert_not_called()

    @patch("views_reporting.templates.reports.evaluation.EvaluationReportTemplate")
    @patch("views_reporting.sources.MetricFrameFileSource")
    def test_sends_wandb_alert_on_success(
        self, mock_source_cls, mock_template_cls,
    ):
        """Evaluation report sends a WandB alert when a report is rendered."""
        stage = _make_stage()
        ctx = _make_context()
        mock_source_cls.return_value.metric_frame.return_value = MagicMock()
        mock_template_cls.return_value.generate.return_value = Path("/tmp/eval_report.html")

        stage.generate_evaluation_report(ctx)

        stage._wandb_module.send_alert.assert_called_once()
        assert "Evaluation Report Generated" in stage._wandb_module.send_alert.call_args[1]["title"]
