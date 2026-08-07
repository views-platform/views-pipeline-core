"""
ForecastingStage — ADR-045 Stage pattern for forecast post-processing.

Simplified: the DF/PF dispatch has been removed. The unified
_execute_model_forecasting in the mixin now handles DF→PF conversion
and writes the combined parquet directly. This stage is retained for
backward compatibility — it's still constructed by ForecastingModelManager
but process_and_save_forecast is no longer called by the unified path.
"""
import logging
from dataclasses import dataclass

from views_pipeline_core.types import BaseStageContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastingContext(BaseStageContext):
    """Immutable context for forecast post-processing."""
    prediction_format: str  # "dataframe" or "prediction_frame"


class ForecastingStage:
    """Forecast post-processing stage (retained for backward compat).

    The unified forecast path in ForecastingMixin._execute_model_forecasting
    now handles all persistence directly. This class is kept so that
    ForecastingModelManager.__init__ can still construct it (r2darts2 and
    other consumers may reference self._forecasting_stage).
    """

    def __init__(self, wandb_module, io_manager, wandb_notifications: bool = False,
                 savers=None):
        self._wandb_module = wandb_module
        self._io = io_manager
        self._wandb_notifications = wandb_notifications
        self._savers = savers or []

    def process_and_save_forecast(self, predictions, context: ForecastingContext) -> None:
        """Legacy entry point — no-op in the unified path.

        The unified _execute_model_forecasting handles all persistence
        directly via _save_combined_forecast(). This method is retained
        for backward compatibility but should not be called.
        """
        logger.warning(
            "process_and_save_forecast is deprecated — the unified "
            "forecast path handles persistence directly."
        )
