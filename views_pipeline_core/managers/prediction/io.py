# LEGACY DataFrame tier — pandas by design; retires with roadmap G5–G7 (#313/#307). See C-226.
"""Prediction I/O — single-responsibility persistence for predictions.

Extracted from ForecastingModelManager to isolate I/O concerns from orchestration.
Each method handles one persistence task: local files, WandB logging, or prediction store.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.files.utils import save_dataframe
from views_pipeline_core.managers.prediction.file_namer import PredictionFileNamer
from views_pipeline_core.managers.prediction.vendor_faults import (
    upload_transport_faults,
)

logger = logging.getLogger(__name__)


class PredictionIOManager:
    """Handles all prediction persistence: local files, prediction store, Appwrite.

    Single Responsibility: This class owns HOW predictions are persisted.
    The orchestration layer (ForecastingModelManager) owns WHAT to persist and WHEN.
    """

    def __init__(
        self,
        model_path,
        wandb_module,
        wandb_notifications: bool,
        use_prediction_store: bool = False,
        datastore=None,
        pred_store_name: Optional[str] = None,
    ):
        self._model_path = model_path
        self._wandb_module = wandb_module
        self._wandb_notifications = wandb_notifications
        self._use_prediction_store = use_prediction_store
        self._datastore = datastore
        self._pred_store_name = pred_store_name

    def save_predictions(
        self,
        df_predictions,
        path_generated: Union[str, Path],
        run_type: str,
        timestamp: str,
        level: Optional[str] = None,
        targets: Optional[list] = None,
        sequence_number: Optional[int] = None,
        target_identifier: Optional[str] = None,
        send_alert: bool = True,
    ) -> None:
        """Save predictions to disk and optionally to prediction store.

        Args:
            df_predictions: Predictions as pd.DataFrame or pa.Table.
            path_generated: Directory for saving.
            run_type: Run type for filename (e.g., "forecasting", "calibration").
            timestamp: Timestamp string for filename.
            level: Spatial level ("pgm" or "cm") for prediction store metadata.
            targets: List of target column names for prediction store metadata.
            sequence_number: Sequence number for evaluation runs. None for forecasting.
            target_identifier: Target name for multi-target models (e.g., "ged_sb_best").
                When provided, included in the filename to prevent collisions
                across targets. Required for any path that saves per-target.
            send_alert: Whether to send a WandB alert.

        Raises:
            PipelineException: If save fails.
        """
        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            namer = PredictionFileNamer(
                run_type, timestamp, PipelineConfig.dataframe_format,
            )
            predictions_name = namer.prediction_name(sequence_number, target_identifier=target_identifier)

            if isinstance(df_predictions, pa.Table):
                pq.write_table(df_predictions, path_generated / predictions_name)
            else:
                save_dataframe(df_predictions, path_generated / predictions_name)

            if self._use_prediction_store:
                self._upload_to_prediction_store(
                    df_predictions, path_generated, predictions_name,
                    level=level, targets=targets,
                )

            if send_alert:
                self._wandb_module.send_alert(
                    title="Predictions Saved",
                    text=f"Predictions saved at {path_generated.relative_to(self._model_path.root)}.",
                    notifications_enabled=self._wandb_notifications,
                )

        except Exception as e:
            logger.error(f"Error saving predictions: {e}", exc_info=True)
            raise PipelineException(
                f"Error saving predictions: {e}",
                wandb_module=self._wandb_module,
            )

    def _upload_to_prediction_store(
        self, df_predictions, path_generated: Path, predictions_name: str,
        level: Optional[str] = None, targets: Optional[list] = None,
    ) -> None:
        """Upload predictions to views-forecasts store and Appwrite datastore."""
        if isinstance(df_predictions, pa.Table):
            raise NotImplementedError(
                "Prediction store upload via PredictionIOManager is not supported "
                "for Arrow Tables. The PF forecasting path uses composed savers "
                "(ViewsForecastsSaver, AppwriteSaver) which handle this directly. "
                "This guard protects the legacy DF path only."
            )

        name = f"{self._model_path.model_name}_{predictions_name.split('.')[0]}"
        df_predictions.forecasts.set_run(self._pred_store_name)
        df_predictions.forecasts.to_store(name=name, overwrite=True)

        if self._datastore is not None:
            # ADR-047: Appwrite is SECONDARY EXTERNAL — log at error, never raise.
            # But `upload_data` reports failure by RETURN VALUE, so the result must be
            # inspected or a failed delivery reads as a successful one (C-227, #330).
            try:
                result = self._datastore.upload_data(
                    file=path_generated / predictions_name,
                    filename=predictions_name,
                    loa=level,
                    name=self._model_path.model_name,
                    targets=targets or [],
                    category="forecast",
                    description="",
                    type=self._model_path.target,
                )
            except upload_transport_faults() as e:
                logger.error(
                    f"Error uploading predictions to datastore: {e}", exc_info=True
                )
                return

            if result is None or getattr(result, "success", False):
                logger.info("Forecasts uploaded to Appwrite Datastore successfully.")
            else:
                logger.error(
                    "Appwrite upload FAILED for %s — the forecast was NOT delivered. "
                    "code=%s error=%s",
                    predictions_name,
                    getattr(result, "code", None),
                    getattr(result, "error", None),
                )

    @staticmethod
    def generate_evaluation_table(metric_dict: Dict) -> str:
        """Format metrics dict as markdown table.

        Args:
            metric_dict: WandB summary metrics dictionary.

        Returns:
            Formatted markdown table string.
        """
        from tabulate import tabulate

        rows = []
        for key, value in metric_dict.items():
            try:
                if not str(key).startswith("_"):
                    rows.append({"Metric": key, "Value": float(value)})
            except (ValueError, TypeError):
                continue
        metric_df = pd.DataFrame(rows, columns=["Metric", "Value"]).sort_values(
            by="Metric"
        )
        result = tabulate(metric_df, headers="keys", tablefmt="grid")
        logger.info(f"Evaluation metrics:\n{result}")
        return f"```\n{result}\n```"
