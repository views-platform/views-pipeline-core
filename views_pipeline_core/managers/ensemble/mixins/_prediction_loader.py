"""PredictionLoaderMixin — load-or-generate predictions for ensemble members.

Extracted from EnsembleManager. Handles loading predictions from the
prediction store, local parquet files, or generating them via shell script.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.data.handlers import _CDataset

logger = logging.getLogger(__name__)


class PredictionLoaderMixin:
    """Mixin providing prediction loading methods for EnsembleManager."""

    def _load_or_generate_prediction(
        self,
        model_path,
        model_name: str,
        name: str,
        path_generated: Path,
        run_type: str,
        ts: str,
        sequence_number: int = None,
        evaluate: bool = False,
        forecast: bool = False,
    ) -> pd.DataFrame:
        """Load existing prediction or generate new one if not found.

        Tries prediction store first (if enabled), then local parquet
        file, then generates via shell script.
        """
        if self._use_prediction_store:
            try:
                pred = pd.DataFrame.forecasts.read_store(
                    run=self._pred_store_name, name=name
                )
                logger.info(f"Loading existing prediction {name} from prediction store")
                return pred
            except (ImportError, KeyError, IndexError, ValueError, AttributeError, OSError) as e:
                logger.info(
                    f"No existing {run_type} predictions found ({type(e).__name__}). Generating..."
                )
        else:
            seq_suffix = (
                f"_{str(sequence_number).zfill(2)}"
                if sequence_number is not None
                else ""
            )
            file_path = (
                path_generated
                / f"predictions_{run_type}_{ts}{seq_suffix}{PipelineConfig.dataframe_format}"
            )
            if file_path.exists():
                pred = read_dataframe(file_path)
                logger.info(f"Loading existing prediction {name} from local file")
                return pred
            else:
                logger.info(
                    f"No existing {run_type} predictions found. Generating..."
                )

        # Generate new predictions
        model_args = self._create_model_args(evaluate=evaluate, forecast=forecast)
        self._execute_shell_script(model_path, model_name, model_args)

        # Load the newly generated prediction
        if self._use_prediction_store:
            return pd.DataFrame.forecasts.read_store(
                run=self._pred_store_name, name=name
            )
        else:
            prediction_files = model_path.get_generated_predictions_data_file_paths(run_type)
            if not prediction_files:
                raise PipelineException(
                    f"No prediction files found for {model_name} after generation",
                    wandb_module=self._wandb_module,
                )
            if sequence_number is not None:
                seq_suffix = f"_{str(sequence_number).zfill(2)}"
                matching = [f for f in prediction_files if f.stem.endswith(seq_suffix)]
                if not matching:
                    raise PipelineException(
                        f"No prediction file for sequence {sequence_number} found for {model_name}",
                        wandb_module=self._wandb_module,
                    )
                latest_prediction_file = matching[0]
            else:
                latest_prediction_file = prediction_files[0]
            logger.info(f"Loading newly generated prediction from {latest_prediction_file}")
            return read_dataframe(latest_prediction_file)

    def _load_c_dataset(self, cm_model: str, c_dataframe: Optional[pd.DataFrame]):
        """Load C dataset from prediction store, local path, or provided DataFrame."""


        if c_dataframe is not None:
            logger.info(f"Using provided C dataset for model {cm_model}")
            return _CDataset(source=c_dataframe)

        if self._use_prediction_store:
            try:
                from views_forecasts.extensions import ViewsMetadata
                logger.info(f"Fetching latest C dataset for {cm_model} from prediction store")
                run_id = ViewsMetadata().get_run_id_from_name(self._pred_store_name)
                all_runs = ViewsMetadata().with_name(cm_model).fetch()["name"].to_list()
                reconcile_with_forecasts = [
                    fc for fc in all_runs if cm_model in fc and "forecasting" in fc
                ]
                reconcile_with_forecasts.sort()
                reconcile_with_forecast = reconcile_with_forecasts[-1]
                return _CDataset(
                    source=pd.DataFrame.forecasts.read_store(
                        run=run_id, name=reconcile_with_forecast
                    )
                )
            except (ImportError, KeyError, IndexError, ValueError, OSError) as e:
                logger.warning(f"Could not find latest C dataset for {cm_model} in prediction store: {e}")

        # Try local path
        try:
            from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager
            logger.info(f"Fetching latest C dataset for {cm_model} from local path")
            return _CDataset(
                source=EnsemblePathManager(cm_model).get_generated_predictions_data_file_paths(
                    run_type=self.configs["run_type"]
                )[0]
            )
        except (IndexError, OSError) as e:
            logger.warning(f"Could not find latest C dataset for {cm_model} locally: {e}")
            return None
