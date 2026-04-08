"""
ForecastingStage — ADR-045 Stage pattern for forecast post-processing.

Extracted from ForecastingModelManager._execute_model_forecasting().
Receives predictions (the result of the abstract _forecast_model_artifact
call) and handles type enforcement (ADR-042), validation, conversion,
persistence, and log creation.

The abstract method _forecast_model_artifact() stays in the facade
because subclasses implement it on self.  This stage receives the
prediction result, not the method.

Responsibilities:
  - Type enforcement (ADR-042): reject mismatched return types
  - DF path: validate via CorePredictionSniffer, save via io_manager
  - PF path: convert via PredictionFrameConverter, save per target
  - Create execution log entry
  - Send WandB completion alert
"""
import logging
from dataclasses import dataclass

from views_pipeline_core.types import BaseStageContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastingContext(BaseStageContext):
    """Immutable context for forecast post-processing.

    Extends BaseStageContext (configs, model_path, run_type) with
    forecasting-specific fields.
    """
    prediction_format: str  # "dataframe" or "prediction_frame"


class ForecastingStage:
    """Forecast post-processing: validate, convert, persist, log.

    Receives predictions from the facade (which called the abstract
    _forecast_model_artifact).  Does not inherit from or access
    ForecastingModelManager.  WandB lifecycle stays in the facade.

    Collaborators (injected at construction):
      - wandb_module: WandBModule — alerts only (no lifecycle management)
      - io_manager: PredictionIOManager — prediction persistence
    """

    def __init__(self, wandb_module, io_manager, wandb_notifications: bool = False,
                 savers=None):
        self._wandb_module = wandb_module
        self._io = io_manager
        self._wandb_notifications = wandb_notifications
        self._savers = savers or []

    def process_and_save_forecast(self, predictions, context: ForecastingContext) -> None:
        """Validate, convert (if PF), and persist forecast predictions.

        Args:
            predictions: Raw output from _forecast_model_artifact().
                - DF path: pd.DataFrame with pred_{target} columns
                - PF path: Dict[str, PredictionFrame]
            context: Frozen ForecastingContext with configuration and paths.

        Raises:
            ValueError: If prediction type mismatches declared prediction_format
                (ADR-042 type enforcement).
        """
        logger.info(
            f"Forecasting {context.model_path.target} {context.configs['name']}..."
        )

        if context.prediction_format == "prediction_frame":
            self._process_pf_path(predictions, context)
        else:
            self._process_df_path(predictions, context)

        # --- Create execution log ---
        from views_pipeline_core.files.utils import handle_single_log_creation

        handle_single_log_creation(
            model_path=context.model_path,
            config=context.configs,
            train=False,
        )

        # --- Completion alert ---
        self._wandb_module.send_alert(
            title=(
                f"Forecasting for {context.model_path.target} "
                f"{context.configs['name']} completed successfully."
            ),
            notifications_enabled=self._wandb_notifications,
        )

    def _process_pf_path(self, predictions, context: ForecastingContext) -> None:
        """PF path: type enforcement, conversion, and per-target persistence."""
        # ADR-042 type enforcement guard (fail-loud)
        if not isinstance(predictions, dict):
            raise ValueError(
                f"prediction_format='prediction_frame' declared but "
                f"_forecast_model_artifact() returned "
                f"{type(predictions).__name__}, expected "
                f"Dict[str, PredictionFrame]. Model contract violation."
            )

        if self._savers:
            self._save_via_savers(predictions, context)
        else:
            self._save_via_io_manager(predictions, context)

    def _save_via_savers(self, predictions, context: ForecastingContext) -> None:
        """Persist each target's PredictionFrame through the composed saver chain."""
        from views_pipeline_core.configs.pipeline import PipelineConfig
        from views_pipeline_core.managers.prediction.file_namer import PredictionFileNamer
        from views_pipeline_core.managers.prediction.savers import PredictionMetadata

        namer = PredictionFileNamer(
            context.run_type,
            context.configs.get("timestamp", ""),
            PipelineConfig.dataframe_format,
        )

        for target, pf in predictions.items():
            metadata = PredictionMetadata(
                model_name=context.model_path.model_name,
                level=context.configs["level"],
                target=target,
                run_type=context.run_type,
                filename=namer.prediction_name(),
            )
            for saver in self._savers:
                saver.save(pf, context.model_path.data_generated, metadata)

    def _save_via_io_manager(self, predictions, context: ForecastingContext) -> None:
        """Legacy PF path: convert to DataFrame and delegate to io_manager."""
        from views_pipeline_core.managers.prediction.prediction_frame_converter import (
            PredictionFrameConverter,
        )

        converter = PredictionFrameConverter()
        for target, pf in predictions.items():
            df_target = converter.to_prediction_df(pf, target)
            converter.audit_prediction_structure(pf, df_target, target)
            self._io.save_predictions(
                df_target,
                context.model_path.data_generated,
                run_type=context.run_type,
                timestamp=context.configs.get("timestamp", ""),
                level=context.configs.get("level"),
                targets=context.configs.get("targets"),
            )

    def _process_df_path(self, predictions, context: ForecastingContext) -> None:
        """DF path: type enforcement, sniff validation, and persistence."""
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            CorePredictionSniffer,
        )

        # ADR-042 type enforcement guard (fail-loud)
        if isinstance(predictions, dict):
            raise ValueError(
                "prediction_format='dataframe' declared but "
                "_forecast_model_artifact() returned a dict, expected "
                "pd.DataFrame. Model contract violation."
            )

        CorePredictionSniffer(level=context.configs["level"]).sniff_predictions(
            predictions, targets=context.configs["targets"],
        )

        self._io.save_predictions(
            predictions,
            context.model_path.data_generated,
            run_type=context.run_type,
            timestamp=context.configs.get("timestamp", ""),
            level=context.configs.get("level"),
            targets=context.configs.get("targets"),
        )
