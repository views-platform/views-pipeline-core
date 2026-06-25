"""
ReportingStage — ADR-045 Stage pattern for report generation.

Extracted from ForecastingModelManager._execute_forecast_reporting()
and _execute_evaluation_reporting().  Receives an explicit, frozen
ReportingContext rather than reaching into a parent class's internals.

Responsibilities:
  - Load historical + forecast data for forecast reports
  - Fetch latest WandB run for evaluation reports
  - Delegate to ForecastReportTemplate / EvaluationReportTemplate
  - Publish completion alerts via WandB
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from views_pipeline_core.types import BaseStageContext

logger = logging.getLogger(__name__)


def _require_dense_report_consumer() -> None:
    """Fail loud if the installed views-reporting cannot consume the dense
    ``prediction_format="prediction_frame"`` report path (register C-190).

    pipeline-core declares no pin on views-reporting and consumes whatever is
    installed; the dense report path dereferences views-reporting's consumer by
    faith. Probe a **public** capability — ``views_reporting.statistics.calculate_map_frame``,
    the bounded ``views_frames_summarize``-backed MAP the dense path relies on and
    that the pre-migration (OOM-prone) views-reporting lacks. This is a capability
    probe, not a version check (views-reporting is not yet meaningfully versioned),
    and it touches only a public symbol (not private internals — avoids the C-135
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


@dataclass(frozen=True)
class ReportingContext(BaseStageContext):
    """Immutable context for report generation.

    Extends BaseStageContext (configs, model_path, run_type) with
    reporting-specific fields.
    """
    entity: str  # WandB entity for get_latest_run(), e.g. "views_pipeline"
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
        """Fetch latest WandB run, generate HTML evaluation report per target.

        Args:
            context: Frozen ReportingContext with configuration, paths, run type.

        Returns:
            Path to the last generated HTML report file.
        """
        from views_pipeline_core.modules.wandb import get_latest_run
        from views_reporting.templates.reports.evaluation import (
            EvaluationReportTemplate,
        )

        latest_run = get_latest_run(
            entity=context.entity,
            model_name=context.model_path.model_name,
            run_type=context.run_type,
        )
        # get_latest_run returns None when the run is genuinely absent (no cloud
        # project, or no finished metrics-bearing run — see issue #177). The
        # downstream template dereferences wandb_run.summary unconditionally, so
        # we degrade here rather than crash. Transient errors are NOT swallowed:
        # they propagate from get_latest_run so a flaky fetch stays visible.
        if latest_run is None:
            logger.warning(
                f"No finished WandB run with metrics found for "
                f"{context.model_path.model_name} ({context.run_type}); "
                f"skipping evaluation report generation."
            )
            return None

        targets = context.configs["targets"]
        if not targets:
            logger.warning("No targets configured — skipping evaluation report generation.")
            return None

        report_path = None
        for target in targets:
            evaluation_template = EvaluationReportTemplate(
                config=context.configs,
                model_path=context.model_path,
                run_type=context.run_type,
            )
            report_path = evaluation_template.generate(
                wandb_run=latest_run, target=target,
            )

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
                targets = config.get("targets")
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
