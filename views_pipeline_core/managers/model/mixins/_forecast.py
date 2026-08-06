"""ForecastingMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the forecast concern methods. It is mixed into
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
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from views_pipeline_core.exceptions import ModelForecastingException, PipelineException
from views_pipeline_core.modules.frames.prediction_frame_io import load_pf, save_pf
from views_pipeline_core.managers.forecasting.stage import ForecastingStage, ForecastingContext

logger = logging.getLogger(__name__)


class ForecastingMixin:
    """Mixin providing forecast methods for ForecastingModelManager."""

    def _execute_model_forecasting(self) -> None:
        """
        Generate future predictions.

        Calls the abstract _forecast_model_artifact() (subclass-specific),
        then delegates post-processing to ForecastingStage (ADR-045 E4).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="forecast")
            - Generates predictions via abstract method
            - Validates, converts, and saves via ForecastingStage
            - Sends completion notification
        """
        import traceback
        from views_pipeline_core.managers.forecasting.stage import ForecastingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="forecast",
        ):
            try:
                predictions = self._forecast_model_artifact(self.args.artifact_name)
                if (
                    self._prediction_format == "prediction_frame"
                    and isinstance(predictions, dict)
                ):
                    _ts = self._model_path.resolve_artifact_path(
                        self.args.run_type, self.args.artifact_name
                    ).stem[-15:]
                    for target, pf in predictions.items():
                        save_pf(
                            pf,
                            self._model_path.data_generated
                            / f"predictions_{self.args.run_type}_{_ts}"
                            / target
                        )
                context = ForecastingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    prediction_format=self._prediction_format,
                )
                self._forecasting_stage.process_and_save_forecast(predictions, context)
            except Exception as e:
                logger.error(
                    f"Error forecasting {self._model_path.target}: {e}", exc_info=True
                )
                raise ModelForecastingException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

