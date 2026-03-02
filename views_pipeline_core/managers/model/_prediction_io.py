"""
Mixin: PredictionIOMixin
========================
Data-layer helpers for ForecastingModelManager.

Responsibilities
----------------
- Coerce raw prediction payloads (pd / pl DataFrames, LazyFrames, file
  paths) into typed ``SpatioTemporalDataset`` objects.
- Validate dataset structure before persistence.
- Write prediction datasets and evaluation outputs to disk and, where
  configured, to the central prediction store.

This mixin contains **no pipeline orchestration** — it knows nothing about
WandB run lifecycles or argument parsing.  Every method operates on already-
materialised data and the shared ``self`` state injected by
``ForecastingModelManager``.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import polars as pl

from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.modules.dataset.core import (
    CountryMonthDataset,
    PriogridMonthDataset,
    SpatioTemporalDataset,
)

logger = logging.getLogger(__name__)


class PredictionIOMixin:
    """
    Mixin providing prediction coercion, validation, and persistence.

    Intended to be used exclusively as a base class of
    ``ForecastingModelManager``.  All methods rely on ``self`` attributes
    set by ``ModelManager.__init__`` (``self.configs``,
    ``self._model_path``, ``self._wandb_module``, etc.).
    """

    # ------------------------------------------------------------------
    # Dataset class resolution
    # ------------------------------------------------------------------

    @staticmethod
    def dataset_class(loa: str) -> Optional[type]:
        """Return a partial constructor for the correct dataset class.

        Args:
            loa: Level of analysis — ``"cm"`` (country-month) or
                ``"pgm"`` (PRIO-grid-month).

        Returns:
            ``partial(CountryMonthDataset)`` or
            ``partial(PriogridMonthDataset)``, or ``None`` if *loa* is
            unrecognised.
        """
        dataset_classes = {"cm": CountryMonthDataset, "pgm": PriogridMonthDataset}
        dataset_cls = dataset_classes.get(loa)
        if dataset_cls:
            return partial(dataset_cls)
        return None

    # ------------------------------------------------------------------
    # Prediction coercion helpers
    # ------------------------------------------------------------------

    def _coerce_to_dataset(
        self,
        data: Union[
            pl.DataFrame,
            str,
            Path,
            SpatioTemporalDataset,
        ],
        target_cols: Optional[List[str]] = None,
    ) -> SpatioTemporalDataset:
        """Convert a prediction payload to the appropriate dataset object.

        Accepts:
            - ``str`` or ``Path`` pointing to a parquet / CSV file (lazy-scanned)
            - ``pl.DataFrame`` (from internal modules such as AggregationModule)
            - An already-constructed ``SpatioTemporalDataset`` (returned as-is)

        User-facing abstract methods (``_forecast_model_artifact``,
        ``_evaluate_model_artifact``, ``_evaluate_sweep``) must return
        ``Path`` objects only.  Raw ``pd.DataFrame`` / ``pl.LazyFrame``
        are **not** accepted — the dataset class constructor handles the
        actual loading and provides schema verification.

        Args:
            data: The raw data source to wrap.
            target_cols: Target column names, required when loading
                historical data (no ``pred_`` columns) so the dataset
                constructor can distinguish historical from forecast mode.

        Returns:
            A ``CountryMonthDataset`` or ``PriogridMonthDataset`` wrapping *data*.

        Raises:
            ValueError: If the configured level of analysis is unknown.
            TypeError:  If *data* has an unsupported type.
        """
        if isinstance(data, SpatioTemporalDataset):
            return data

        loa = self.configs.get("level")
        dataset_cls_partial = self.dataset_class(loa)
        if dataset_cls_partial is None:
            raise ValueError(
                f"Cannot coerce predictions: unknown level of analysis '{loa}'. "
                "Expected 'cm' or 'pgm'."
            )

        if not isinstance(data, (pl.DataFrame, pd.DataFrame, str, Path)):
            raise TypeError(
                f"Cannot coerce predictions: unsupported type {type(data).__name__}. "
                "Expected Path, str, pd.DataFrame, pl.DataFrame, or SpatioTemporalDataset."
            )

        # Convert pandas to Polars so the dataset constructor receives a
        # uniform type.  Pandas DataFrames with a MultiIndex are reset so
        # the index columns become regular columns.
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                data = pl.from_pandas(data.reset_index())
            else:
                data = pl.from_pandas(data)

        kwargs = {"data": data}
        if target_cols is not None:
            kwargs["target_cols"] = target_cols
        return dataset_cls_partial(**kwargs)

    def _coerce_predictions_to_datasets(
        self,
        raw_predictions: List[
            Union[
                pl.DataFrame,
                str,
                Path,
                SpatioTemporalDataset,
            ]
        ],
    ) -> List[SpatioTemporalDataset]:
        """Convert a list of prediction payloads to dataset objects."""
        return [self._coerce_to_dataset(item) for item in raw_predictions]

    # ------------------------------------------------------------------
    # Dataset-level validation and persistence (Polars-native)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_prediction_dataset(
        dataset: SpatioTemporalDataset,
        targets: Union[str, List[str]],
    ) -> None:
        """Validate a prediction dataset structure (Polars-native).

        Performs the same structural checks as the legacy
        ``validate_prediction_dataframe`` but operates entirely on the
        dataset object and its underlying LazyFrame — no pandas
        conversion required.

        Checks:
            1. Dataset is non-empty.
            2. Every declared target has a corresponding ``pred_{target}``
               column.
            3. The dataset contains a recognised time column
               (``month_id``).

        Args:
            dataset: The prediction dataset to validate.
            targets: Target variable name(s).

        Raises:
            ValueError: On any validation failure.
        """
        if dataset._n_rows == 0:
            raise ValueError("Prediction dataset is empty")

        if isinstance(targets, str):
            targets = [targets]

        cols = set(dataset.columns)
        missing = [t for t in targets if f"pred_{t}" not in cols]
        if missing:
            raise ValueError(
                f"Missing prediction columns for targets: {missing}. "
                f"Expected columns: {['pred_' + t for t in missing]}"
            )

        if dataset.time_col not in cols:
            raise ValueError(
                f"Time column '{dataset.time_col}' not found in dataset columns"
            )

        logger.info(
            f"Dataset validation passed: {dataset._n_rows:,} rows, "
            f"{len(cols)} cols, targets={targets}"
        )

    def _save_prediction_dataset(
        self,
        dataset: SpatioTemporalDataset,
        path_generated: Union[str, Path],
        sequence_number: int = None,
        send_alert: bool = True,
    ) -> None:
        """Save a prediction dataset to disk (Polars-native).

        Writes the dataset directly to parquet via Polars, avoiding any
        pandas conversion.  If the prediction store is enabled, conversion
        to pandas happens only at that external-API boundary.

        Args:
            dataset: Prediction dataset to save.
            path_generated: Target directory.
            sequence_number: Sequence index (evaluation runs) or None.
            send_alert: Send a WandB notification on completion.
        """
        from views_pipeline_core.files.utils import generate_output_file_name

        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            self._predictions_name = generate_output_file_name(
                "predictions",
                self.configs["run_type"],
                self.configs["timestamp"],
                sequence_number,
                file_extension=PipelineConfig().dataframe_format,
            )

            save_path = path_generated / self._predictions_name

            # Materialise and write directly via Polars
            df_pl: pl.DataFrame = dataset.collect()
            df_pl.write_parquet(save_path)

            # Register the saved prediction in the artifact registry
            stage = "evaluate" if sequence_number is not None else "forecast"
            entry = self._artifact_registry.register(
                filepath=save_path,
                run_type=self.configs["run_type"],
                stage=stage,
                parent_id=getattr(self, '_current_train_artifact_id', None),
                metadata={"timestamp": self.configs.get("timestamp")},
            )
            logger.info(
                f"Registered prediction dataset artifact {entry.id} "
                f"(stage={stage!r}, file={save_path.name!r})"
            )

            # Prediction store uses the pandas forecasts accessor — convert
            # only at this external boundary.
            if self._use_prediction_store:
                df_pd = dataset.get_subset_dataframe(return_pandas=True)
                name = f"{self._model_path.model_name}_{self._predictions_name.split('.')[0]}"
                df_pd.forecasts.set_run(self._pred_store_name)
                df_pd.forecasts.to_store(name=name, overwrite=True)

                if self._datastore is not None:
                    try:
                        self._datastore.upload_data(
                            file=save_path,
                            filename=self._predictions_name,
                            loa=self.configs.get("level"),
                            name=self._model_path.model_name,
                            targets=self.configs.get("targets"),
                            category="forecast",
                            description="",
                            type=self._model_path.target,
                        )
                        logger.info(
                            "Forecasts uploaded to Appwrite Datastore successfully."
                        )
                    except Exception as e:
                        logger.error(
                            f"Error uploading predictions to datastore: {e}",
                            exc_info=True,
                        )

            if send_alert:
                self._wandb_module.send_alert(
                    title="Predictions Saved",
                    text=f"Predictions saved at {path_generated.relative_to(self._model_path.root)}.",
                    notifications_enabled=self._wandb_notifications,
                )

        except Exception as e:
            raise PipelineException(
                f"Error saving predictions: {e}",
                wandb_module=self._wandb_module,
            )

    @staticmethod
    def _dataset_to_pandas(dataset: SpatioTemporalDataset) -> pd.DataFrame:
        """Extract a pandas DataFrame with MultiIndex from a dataset object.

        Used only at external-API boundaries (EvaluationManager, WandB
        Tables, prediction store) where pandas is unavoidable.
        """
        return dataset.get_subset_dataframe(return_pandas=True)

    @staticmethod
    def _resolve_evaluation_sequence_number(eval_type: str) -> int:
        """
        Get number of evaluation sequences for type.

        Maps evaluation type to sequence count for temporal evaluation.

        Evaluation Types:
            - standard: 12 sequences (1 year)
            - long: 36 sequences (3 years)
            - complete: None (full period, needs calculation)
            - live: 12 sequences (current year)

        Args:
            eval_type: Type of evaluation

        Returns:
            Number of sequences, or None for complete type

        Raises:
            ValueError: If eval_type invalid

        Example:
            >>> n = ForecastingModelManager._resolve_evaluation_sequence_number("standard")
            >>> print(n)
            12
        """
        if eval_type == "standard":
            return 12
        elif eval_type == "long":
            return 36
        elif eval_type == "complete":
            return None  # currently set as None because sophisticated calculation is needed
        elif eval_type == "live":
            return 12
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")

    # ------------------------------------------------------------------
    # Legacy pandas-based persistence (deprecated — routes through dataset path)
    # ------------------------------------------------------------------

    def _save_predictions(
        self,
        df_predictions: Union[pl.DataFrame, str, Path, SpatioTemporalDataset],
        path_generated: Union[str, Path],
        sequence_number: int = None,
        send_alert: bool = True,
    ) -> None:
        """
        Save predictions to disk and prediction store.

        .. deprecated::
            Use ``_save_prediction_dataset`` directly with a
            ``SpatioTemporalDataset``.  This method coerces the input to
            a dataset and delegates.

        Accepts Path (preferred), pl.DataFrame, or SpatioTemporalDataset.
        Data is converted to a dataset object and saved via the
        Polars-native path.

        Args:
            df_predictions: Predictions (Path, pl.DataFrame, or dataset).
            path_generated: Directory for saving.
            sequence_number: Sequence number for evaluation runs.
            send_alert: Whether to send a WandB alert.

        Raises:
            PipelineException: If save fails.
        """
        import warnings
        warnings.warn(
            "_save_predictions is deprecated. Use _save_prediction_dataset "
            "with a SpatioTemporalDataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = self._coerce_to_dataset(df_predictions)
        self._save_prediction_dataset(
            dataset, path_generated, sequence_number, send_alert,
        )

    def _save_model_artifact(self, run_type: str) -> None:
        """
        Upload model artifact to WandB.

        Creates WandB artifact from saved model file for versioning
        and tracking.

        Internal Use:
            Called after training to version model artifacts.

        Args:
            run_type: Run type for artifact naming

        Raises:
            PipelineException: If artifact upload fails

        Side Effects:
            - Creates WandB artifact
            - Uploads model file
            - Logs artifact reference
        """
        try:
            _latest_model_artifact_path = (
                self._model_path.get_latest_model_artifact_path(run_type=run_type)
            )

            self._wandb_module.log_artifact(
                artifact_path=_latest_model_artifact_path,
                artifact_name=f"{run_type}_{self._model_path.target}_artifact",
                artifact_type=self._model_path.target,
                description=f"Latest {run_type} {self._model_path.target} artifact",
            )

            logger.info(
                f"Artifact for run type: {run_type} saved to WandB successfully."
            )

        except Exception as e:
            raise PipelineException(
                f"Error saving artifact to WandB: {e}",
                wandb_module=self._wandb_module,
            )

    def _save_eval_report(self, eval_report, path_reports, target_identifier):
        """
        Save evaluation metrics report as JSON.

        Internal Use:
            Called during evaluation reporting.

        Args:
            eval_report: Dictionary of evaluation metrics
            path_reports: Directory for saving reports
            target_identifier: Target identifier code for filename

        Raises:
            PipelineException: If save fails
        """
        import json

        from views_pipeline_core.files.utils import generate_evaluation_report_name

        try:
            path_reports = Path(path_reports)
            path_reports.mkdir(parents=True, exist_ok=True)

            eval_report_path = generate_evaluation_report_name(
                self.configs["run_type"],
                target_identifier,
                self.configs["timestamp"],
                file_extension=".json",
            )

            with open(path_reports / eval_report_path, "w") as f:
                json.dump(eval_report, f)

            # Register the report in the artifact registry
            entry = self._artifact_registry.register(
                filepath=path_reports / eval_report_path,
                run_type=self.configs["run_type"],
                stage="report",
                parent_id=getattr(self, '_current_train_artifact_id', None),
                metadata={"target": target_identifier},
            )
            logger.info(
                f"Registered evaluation report artifact {entry.id} "
                f"(file={eval_report_path!r})"
            )

        except Exception as e:
            raise PipelineException(
                f"Error saving evaluation report: {e}",
                wandb_module=self._wandb_module,
            )

    def _save_evaluations(
        self,
        df_step_wise_evaluation: Union[pd.DataFrame, pl.DataFrame],
        df_time_series_wise_evaluation: Union[pd.DataFrame, pl.DataFrame],
        df_month_wise_evaluation: Union[pd.DataFrame, pl.DataFrame],
        path_generated: Union[str, Path],
        target_identifier: str,
    ) -> None:
        """
        Save evaluation metrics to disk and WandB.

        Saves three levels of evaluation metrics (step, time-series, month)
        to parquet files and logs to WandB.  Accepts either pandas or
        Polars DataFrames; Polars DataFrames are written directly while
        pandas DataFrames are routed through the legacy save path.

        Internal Use:
            Called by _evaluate_prediction_dataframe().

        Args:
            df_step_wise_evaluation: Metrics per prediction step.
            df_time_series_wise_evaluation: Metrics per time series.
            df_month_wise_evaluation: Metrics per month.
            path_generated: Directory for saving files.
            target_identifier: Target identifier for filename.

        Side Effects:
            - Saves three parquet files.
            - Logs tables to WandB.
            - Sends completion notification.

        Raises:
            PipelineException: If save fails.
        """
        import wandb

        from views_pipeline_core.files.utils import (
            generate_evaluation_file_name,
            save_dataframe,
        )

        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            eval_step_path = generate_evaluation_file_name(
                "step",
                target_identifier,
                self.configs["run_type"],
                self.configs["timestamp"],
                PipelineConfig().dataframe_format,
            )
            eval_ts_path = generate_evaluation_file_name(
                "ts",
                target_identifier,
                self.configs["run_type"],
                self.configs["timestamp"],
                PipelineConfig().dataframe_format,
            )
            eval_month_path = generate_evaluation_file_name(
                "month",
                target_identifier,
                self.configs["run_type"],
                self.configs["timestamp"],
                PipelineConfig().dataframe_format,
            )

            save_dataframe(df_month_wise_evaluation, path_generated / eval_month_path)
            save_dataframe(
                df_time_series_wise_evaluation, path_generated / eval_ts_path
            )
            save_dataframe(df_step_wise_evaluation, path_generated / eval_step_path)

            # Register evaluation files in the artifact registry
            for eval_path in (eval_month_path, eval_ts_path, eval_step_path):
                entry = self._artifact_registry.register(
                    filepath=path_generated / eval_path,
                    run_type=self.configs["run_type"],
                    stage="evaluate",
                    parent_id=getattr(self, '_current_train_artifact_id', None),
                    metadata={
                        "target": target_identifier,
                        "timestamp": self.configs.get("timestamp"),
                    },
                )
                logger.info(
                    f"Registered evaluation artifact {entry.id} "
                    f"(file={eval_path!r})"
                )

            self._wandb_module.save(str(path_generated / eval_month_path))
            self._wandb_module.save(str(path_generated / eval_ts_path))
            self._wandb_module.save(str(path_generated / eval_step_path))

            self._wandb_module.log(
                {
                    "evaluation_metrics_month": wandb.Table(
                        dataframe=df_month_wise_evaluation if isinstance(df_month_wise_evaluation, pd.DataFrame)
                        else df_month_wise_evaluation.to_pandas()
                    ),
                    "evaluation_metrics_ts": wandb.Table(
                        dataframe=df_time_series_wise_evaluation if isinstance(df_time_series_wise_evaluation, pd.DataFrame)
                        else df_time_series_wise_evaluation.to_pandas()
                    ),
                    "evaluation_metrics_step": wandb.Table(
                        dataframe=df_step_wise_evaluation if isinstance(df_step_wise_evaluation, pd.DataFrame)
                        else df_step_wise_evaluation.to_pandas()
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
