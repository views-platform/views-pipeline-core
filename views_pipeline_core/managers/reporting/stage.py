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


@dataclass(frozen=True)
class ReportingContext(BaseStageContext):
    """Immutable context for report generation.

    Extends BaseStageContext (configs, model_path, run_type) with
    reporting-specific fields.
    """
    entity: str  # WandB entity for get_latest_run(), e.g. "views_pipeline"


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
        from views_pipeline_core.files.utils import read_dataframe

        logger.info(
            f"Generating forecast report for "
            f"{context.model_path.target} {context.configs['name']}..."
        )

        # --- Load historical data ---
        historical_df = self._load_historical_data(context)

        # --- Load forecast data ---
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

        # --- Generate report ---
        from views_pipeline_core.templates.reports.forecast import (
            ForecastReportTemplate,
        )

        forecast_template = ForecastReportTemplate(
            config=context.configs,
            model_path=context.model_path,
            run_type=context.run_type,
        )
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
        from views_pipeline_core.templates.reports.evaluation import (
            EvaluationReportTemplate,
        )

        latest_run = get_latest_run(
            entity=context.entity,
            model_name=context.model_path.model_name,
            run_type=context.run_type,
        )

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
