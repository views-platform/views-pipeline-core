"""
ReportingStage — ADR-045 Stage pattern for report generation.

Extracted from ForecastingModelManager._execute_forecast_reporting()
and _execute_evaluation_reporting().  Receives an explicit, frozen
ReportingContext rather than reaching into a parent class's internals.

Responsibilities:
  - Load historical + forecast data for forecast reports
  - Render evaluation reports from the persisted MetricFrame via an injected
    EvaluationSource (ADR-018 / C-108) — not a render-time WandB scrape
  - Delegate to ForecastReportTemplate / EvaluationReportTemplate
  - Publish completion alerts via WandB
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from views_pipeline_core.types import BaseStageContext

from views_pipeline_core.managers.configuration.configuration import combined_targets
from views_pipeline_core.modules.validation.core_config_sniffer import (
    DEPRECATED_REPORT_PREDICTION_FORMATS,
)
logger = logging.getLogger(__name__)


def _warn_if_report_format_is_deprecated(prediction_format: str, model_name: str) -> None:
    """Announce that a report-enabled `dataframe` run is on a deprecated path (#211).

    Reaching this module at all means the run is report-enabled, which is why the check
    lives here rather than in `CoreConfigSniffer`: the sniffer reads a config dict and
    cannot see the `-re` flag, so it cannot tell a `dataframe` model that reports from
    one that does not. Only the second is deprecated.

    **This is the warn half of D-36, deliberately not the reject half.** The register
    settled the direction — reject `dataframe` reports loudly rather than silently take
    the OOM path — but rejecting is only safe once report-bearing models have somewhere
    to go, and which of views-models' `dataframe` configs are report-enabled has not been
    audited. Rejecting first would break runs that have no alternative. The set this reads
    is named so that flip is a one-name change, not a condition edit.

    ## Why `logger.warning` and not `warnings.warn(..., DeprecationWarning)`

    Because ADR-056 makes generated model mains silence `DeprecationWarning` and
    `FutureWarning` by name, and reports run inside exactly those mains. A
    `DeprecationWarning` here would be filtered out in the only process that emits it —
    the same defect #366 fixed one layer up, and it would be self-inflicted this time.
    `UserWarning` would survive that filter today, but it would be borrowing a category
    to dodge our own suppression rather than because the category fits. The run log is
    the channel an operator actually reads, and nothing filters it.
    """
    if prediction_format not in DEPRECATED_REPORT_PREDICTION_FORMATS:
        return

    logger.warning(
        "DEPRECATED: model '%s' is generating a report with "
        "prediction_format='%s'. This path hands list-in-cell values to "
        "views-reporting to densify, which is the #181 out-of-memory failure; "
        "prediction_format='prediction_frame' is bounded and is the supported path "
        "for reports. This will become a hard error once report-bearing models have "
        "migrated (#211, register D-36). Nothing is being redirected for you — the "
        "run continues on the path you chose.",
        model_name,
        prediction_format,
    )


def _require_dense_report_consumer() -> None:
    """Fail loud if the installed views-reporting cannot consume the dense
    ``prediction_format="prediction_frame"`` report path (register C-190).

    pipeline-core declares no pin on views-reporting and consumes whatever is
    installed; the dense report path dereferences views-reporting's consumer by
    faith. Probe a **public** capability — ``views_reporting.statistics.calculate_map_frame``,
    the bounded ``views_frames_summarize``-backed MAP the dense path relies on and
    that the pre-migration (OOM-prone) views-reporting lacks. This is a capability
    probe, not a version check — and deliberately stays one now that it could be a
    version check: views-reporting reached 0.3.3 on PyPI on 2026-08-02, so the old
    justification here ("not yet meaningfully versioned") is retired. A capability probe
    remains the better instrument because this repo declares no pin (ADR-054, #375) and
    consumes whatever is installed, so the question that matters is what the installed
    build can *do*, not what it is called. It touches only a public symbol (not private internals — avoids the C-135
    coupling). Raises with an actionable remediation rather than letting an
    AttributeError surface deep inside the template.
    """
    try:
        from views_reporting.statistics import calculate_map_frame  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The installed views-reporting cannot consume the dense "
            "prediction_frame report path: 'views_reporting.statistics.calculate_map_frame' "
            "is missing. Upgrade views-reporting to a build with the views-frames dense "
            "consumer (the migrated report path), or run the report with "
            "prediction_format='dataframe'. "
            f"Underlying import error: {e}"
        ) from e


def _require_evaluation_source_consumer() -> None:
    """Fail loud if the installed views-reporting lacks the MetricFrame ``EvaluationSource``
    consumer (ADR-018 / C-108, views-reporting #173) — the typed eval-of-record report path
    this stage now requires. Mirrors :func:`_require_dense_report_consumer` (C-190): probe a
    public capability with an actionable remediation rather than letting an ImportError
    surface deep in report generation.
    """
    try:
        from views_reporting.sources import MetricFrameFileSource  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The installed views-reporting lacks the MetricFrame EvaluationSource consumer "
            "('views_reporting.sources.MetricFrameFileSource'). The evaluation report now "
            "renders from the persisted MetricFrame (ADR-018 / C-108), not a render-time "
            "WandB scrape. Upgrade views-reporting to a build with EvaluationSource (#173). "
            f"Underlying import error: {e}"
        ) from e


@dataclass(frozen=True)
class ReportingContext(BaseStageContext):
    """Immutable context for report generation.

    Extends BaseStageContext (configs, model_path, run_type) with
    reporting-specific fields.
    """
    entity: str  # WandB entity; set at wandb.init() upstream. No stage method reads it now
    # (the eval path retired get_latest_run); retained for call-site/back-compat.
    prediction_format: str = "dataframe"


class ReportingStage:
    """Report generation: forecast maps and evaluation metric reports.

    Receives an explicit, frozen ReportingContext.  Does not inherit from
    or access ForecastingModelManager.  WandB lifecycle (initialize_run /
    finish_run) stays in the facade — this stage contains only the
    business logic that runs inside the try block.

    Collaborators (injected at construction):
      - wandb_module: WandBModule — alerts only (no lifecycle management)
    """

    def __init__(self, wandb_module, wandb_notifications: bool = False):
        self._wandb_module = wandb_module
        self._wandb_notifications = wandb_notifications

    def generate_forecast_report(self, context: ReportingContext) -> Path:
        """Load historical + forecast data, generate HTML forecast report.

        Args:
            context: Frozen ReportingContext with configuration, paths, run type.

        Returns:
            Path to the generated HTML report file.

        Raises:
            FileNotFoundError: If forecast dataframe is not found.
            ValueError: If model_path.target is not 'model' or 'ensemble'.
        """
        logger.info(
            f"Generating forecast report for "
            f"{context.model_path.target} {context.configs['name']}..."
        )

        # --- Load historical data ---
        historical_df = self._load_historical_data(context)

        # --- Generate report ---
        from views_reporting.templates.reports.forecast import (
            ForecastReportTemplate,
        )

        forecast_template = ForecastReportTemplate(
            config=context.configs,
            model_path=context.model_path,
            run_type=context.run_type,
        )

        _warn_if_report_format_is_deprecated(
            context.prediction_format, context.configs.get("name", "<unnamed>")
        )

        if context.prediction_format == "prediction_frame":
            # C-190: fail loud at the boundary if the installed views-reporting
            # can't consume the dense path, rather than crash deep in the template.
            _require_dense_report_consumer()
            try:
                prediction_path = (
                    context.model_path._get_generated_pf_prediction_paths(
                        run_type=context.run_type
                    )[0]
                )
            except (FileNotFoundError, IndexError) as e:
                raise FileNotFoundError(
                    f"No PredictionFrame prediction directory found. Run the "
                    f"pipeline in forecasting mode with '--run_type forecasting' "
                    f"to generate predictions. More info: {e}"
                ) from e
            logger.info("Using PredictionFrame predictions from %s", prediction_path)
            report_path = forecast_template.generate(
                historical_dataframe=historical_df,
                prediction_format="prediction_frame",
                prediction_path=prediction_path,
            )
        else:
            from views_pipeline_core.files.utils import read_dataframe

            try:
                forecast_df = read_dataframe(
                    context.model_path._get_generated_predictions_data_file_paths(
                        run_type=context.run_type
                    )[0]
                )
                logger.info("Using latest forecast dataframe")
            except (FileNotFoundError, IndexError) as e:
                raise FileNotFoundError(
                    f"Forecast dataframe was probably not found. Please run the "
                    f"pipeline in forecasting mode with '--run_type forecasting' "
                    f"to generate the forecast dataframe. More info: {e}"
                ) from e
            report_path = forecast_template.generate(
                forecast_dataframe=forecast_df,
                historical_dataframe=historical_df,
            )

        self._wandb_module.send_alert(
            title="Forecast Report Generated",
            text=(
                f"Forecast report for {context.model_path.target} "
                f"{context.model_path.model_name} has been successfully "
                f"generated and saved locally at {report_path}."
            ),
            notifications_enabled=self._wandb_notifications,
            models_path=context.model_path.models,
        )

        return report_path

    def generate_evaluation_report(self, context: ReportingContext) -> Optional[Path]:
        """Generate HTML evaluation report(s) from the persisted MetricFrame(s).

        Renders from the typed evaluation-of-record MetricFrame via an injected
        ``MetricFrameFileSource`` (ADR-018 / C-108) — **not** a render-time WandB scrape
        (``get_latest_run``), which was the C-48/C-110 wrong-run failure class. One report
        per target; a target with no persisted frame is skipped (its eval may have been
        capability-skipped or not run).

        Args:
            context: Frozen ReportingContext with configuration, paths, run type.

        Returns:
            Path to the last generated HTML report, or None if nothing was rendered.
        """
        _require_evaluation_source_consumer()
        from views_reporting.sources import MetricFrameFileSource
        from views_reporting.templates.reports.evaluation import (
            EvaluationReportTemplate,
        )

        targets = combined_targets(context.configs)
        if not targets:
            logger.warning("No targets configured — skipping evaluation report generation.")
            return None

        model_name = context.model_path.model_name
        report_path = None
        for target in targets:
            # LOCKED cross-repo path contract (C-202): root=<data_generated> matches the
            # producer's <data_generated>/<model>/<run_type>/metricframe_<target> layout
            # (EvaluationStage._save_metric_frame ↔ MetricFrameFileSource._frame_dir).
            source = MetricFrameFileSource(
                root=context.model_path.data_generated,
                run_type=context.run_type,
                target=target,
                primary_model=model_name,
            )
            if source.metric_frame(model_name) is None:
                logger.warning(
                    "No MetricFrame for %s (%s, target=%s) at the eval-of-record path; "
                    "skipping evaluation report for this target.",
                    model_name, context.run_type, target,
                )
                continue
            evaluation_template = EvaluationReportTemplate(
                config=context.configs,
                model_path=context.model_path,
                run_type=context.run_type,
            )
            report_path = evaluation_template.generate(source=source, target=target)

        if report_path is None:
            return None

        self._wandb_module.send_alert(
            title="Evaluation Report Generated",
            text=(
                f"Evaluation report for {context.model_path.model_name} "
                f"has been successfully generated and saved locally at "
                f"{report_path}."
            ),
            notifications_enabled=self._wandb_notifications,
            models_path=context.model_path.models,
        )

        return report_path

    @staticmethod
    def _load_historical_data(context: ReportingContext):
        """Load historical actuals for forecast report.

        Dispatches between single-model and ensemble paths based on
        context.model_path.target.
        """
        import pandas as pd
        from views_pipeline_core.files.utils import read_dataframe

        if context.model_path.target == "ensemble":
            from views_pipeline_core.managers.model import (
                ModelPathManager,
                ModelManager,
            )

            models = context.configs.get("models")
            reference_index = None
            historical_df = None

            for model in models:
                mp = ModelPathManager(model_path=model, validate=True)
                config = ModelManager(
                    model_path=mp,
                    wandb_notifications=False,
                    use_prediction_store=False,
                ).configs
                df = read_dataframe(
                    file_path=mp._get_raw_data_file_paths(
                        run_type=context.run_type
                    )[0]
                )
                if reference_index is None or historical_df is None:
                    reference_index = df.index
                    historical_df = pd.DataFrame(index=reference_index)
                targets = combined_targets(config)
                targets = targets if isinstance(targets, list) else [targets]
                for target in targets:
                    if target not in historical_df.columns:
                        if df.index.equals(reference_index):
                            historical_df[target] = df[target]
                        else:
                            logger.warning(
                                f"Index mismatch for target {target} in "
                                f"model {model}. Skipping this target."
                            )

            return historical_df

        elif context.model_path.target == "model":
            return read_dataframe(
                context.model_path._get_raw_data_file_paths(
                    run_type=context.run_type
                )[0]
            )

        else:
            raise ValueError(
                f"Invalid target type: {context.model_path.target}. "
                f"Expected 'model' or 'ensemble'."
            )
