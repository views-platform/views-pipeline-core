"""Prediction I/O — single-responsibility persistence for predictions and evaluations.

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
            predictions_name = namer.prediction_name(sequence_number)

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
                "Prediction store upload is not yet supported for Arrow Tables "
                "(PredictionFrame evaluation path). Save as parquet only, or "
                "disable prediction_store for this model."
            )

        name = f"{self._model_path.model_name}_{predictions_name.split('.')[0]}"
        df_predictions.forecasts.set_run(self._pred_store_name)
        df_predictions.forecasts.to_store(name=name, overwrite=True)

        if self._datastore is not None:
            try:
                self._datastore.upload_data(
                    file=path_generated / predictions_name,
                    filename=predictions_name,
                    loa=level,
                    name=self._model_path.model_name,
                    targets=targets or [],
                    category="forecast",
                    description="",
                    type=self._model_path.target,
                )
                logger.info("Forecasts uploaded to Appwrite Datastore successfully.")
            except Exception as e:
                logger.error(
                    f"Error uploading predictions to datastore: {e}", exc_info=True
                )

    def save_evaluations(
        self,
        df_step_wise_evaluation: pd.DataFrame,
        df_time_series_wise_evaluation: pd.DataFrame,
        df_month_wise_evaluation: pd.DataFrame,
        path_generated: Union[str, Path],
        target_identifier: str,
        run_type: str,
        timestamp: str,
    ) -> None:
        """Save evaluation metrics to disk and WandB.

        Args:
            df_step_wise_evaluation: Metrics per prediction step.
            df_time_series_wise_evaluation: Metrics per time series.
            df_month_wise_evaluation: Metrics per month.
            path_generated: Directory for saving files.
            target_identifier: Target identifier for filename.
            run_type: Run type for filename.
            timestamp: Timestamp string for filename.

        Raises:
            PipelineException: If save fails.
        """
        import wandb

        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            namer = PredictionFileNamer(
                run_type, timestamp, PipelineConfig.dataframe_format,
            )
            eval_step_path = namer.evaluation_name("step", target_identifier)
            eval_ts_path = namer.evaluation_name("ts", target_identifier)
            eval_month_path = namer.evaluation_name("month", target_identifier)

            save_dataframe(df_month_wise_evaluation, path_generated / eval_month_path)
            save_dataframe(df_time_series_wise_evaluation, path_generated / eval_ts_path)
            save_dataframe(df_step_wise_evaluation, path_generated / eval_step_path)

            self._wandb_module.save(str(path_generated / eval_month_path))
            self._wandb_module.save(str(path_generated / eval_ts_path))
            self._wandb_module.save(str(path_generated / eval_step_path))

            self._wandb_module.log(
                {
                    "evaluation_metrics_month": wandb.Table(
                        dataframe=df_month_wise_evaluation
                    ),
                    "evaluation_metrics_ts": wandb.Table(
                        dataframe=df_time_series_wise_evaluation
                    ),
                    "evaluation_metrics_step": wandb.Table(
                        dataframe=df_step_wise_evaluation
                    ),
                }
            )

            self._wandb_module.send_alert(
                title=f"{self._model_path.target.title()} Outputs Saved",
                text=f"Evaluation metrics saved at {path_generated.relative_to(self._model_path.root)}.",
                notifications_enabled=self._wandb_notifications,
            )

        except Exception as e:
            logger.error(f"Error saving model outputs: {e}", exc_info=True)
            raise PipelineException(
                f"Error saving model outputs: {e}",
                wandb_module=self._wandb_module,
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
