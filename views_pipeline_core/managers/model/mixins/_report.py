"""ReportingMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the report concern methods. It is mixed into
ForecastingModelManager via multiple inheritance; all methods read/write
``self._*`` attributes that are set on the combined instance by
ModelManager.__init__ and ForecastingModelManager.__init__.

Backward compatibility: every method keeps its exact name and signature.
r2darts2's DartsForecastingModelManager (which subclasses
ForecastingModelManager) continues to work unchanged.
"""
from __future__ import annotations

# Imports are kept minimal — each mixin imports only what its methods use.
# Heavy imports (pandas, pyarrow) are deferred to runtime inside method bodies
# to preserve import purity (the base manager must remain pandas-free at
# module scope; see _lazy.py and tests/test_import_purity.py).

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelEvaluationException,
    ModelTrainingException,
    PipelineException,
)
from views_pipeline_core.managers.reporting.stage import ReportingStage, ReportingContext

logger = logging.getLogger(__name__)


class ReportingMixin:
    """Mixin providing report methods for ForecastingModelManager."""

    def _execute_forecast_reporting(self) -> None:
        """
        Generate forecast visualization report.

        Delegates to ReportingStage.generate_forecast_report() (ADR-045 E3).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="report")
            - Generates HTML report via ReportingStage
            - Sends completion notification
        """
        from views_pipeline_core.managers.reporting.stage import ReportingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                context = ReportingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    prediction_format=self._prediction_format,
                )
                self._reporting_stage.generate_forecast_report(context)
            except PipelineException:
                raise
            except Exception:
                logger.error(f"Forecast report generation failed: {traceback.format_exc()}")
                raise PipelineException(
                    f"Forecast report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_evaluation_reporting(self) -> None:
        """
        Generate evaluation visualization report.

        Delegates to ReportingStage.generate_evaluation_report() (ADR-045 E3).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="report")
            - Generates HTML report via ReportingStage
            - Sends completion notification
        """
        from views_pipeline_core.managers.reporting.stage import ReportingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                context = ReportingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    prediction_format=self._prediction_format,
                )
                self._reporting_stage.generate_evaluation_report(context)
            except PipelineException:
                raise
            except Exception:
                logger.error(f"Evaluation report generation failed: {traceback.format_exc()}")
                raise PipelineException(
                    f"Evaluation report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

