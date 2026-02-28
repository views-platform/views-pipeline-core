"""
Mixin: EvaluationMixin
======================
Metric computation and WandB evaluation-logging helpers for
ForecastingModelManager.

Responsibilities
----------------
- Load ground-truth actuals and align them with model predictions.
- Drive the EvaluationManager for each target across regression and
  classification tasks.
- Log per-step, per-time-series, and per-month metric tables to WandB.
- Format a human-readable summary table for WandB alerts.

This mixin contains **no I/O or pipeline orchestration** — persistence of
evaluation artefacts is delegated to ``PredictionIOMixin._save_evaluations``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import polars as pl

from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.modules.dataset.core import SpatioTemporalDataset

logger = logging.getLogger(__name__)


class EvaluationMixin:
    """
    Mixin providing evaluation metric computation and WandB logging.

    Intended to be used exclusively as a base class of
    ``ForecastingModelManager``.  All methods rely on ``self`` attributes
    set by ``ModelManager.__init__``.
    """

    def _evaluate_prediction_dataframe(
        self, predictions, eval_type, ensemble=False
    ) -> None:
        """
        Calculate evaluation metrics from predictions.

        Computes metrics at multiple aggregation levels (step, time-series,
        month) and logs to WandB. Saves results to disk.

        Internal Use:
            Called by evaluation and sweep methods.

        Args:
            predictions: List of SpatioTemporalDataset objects, a single
                dataset, or a list of pd.DataFrames (legacy callers).
            eval_type: Evaluation type
            ensemble: Whether predictions from ensemble model

        Side Effects:
            - Calculates metrics using EvaluationManager
            - Logs metrics to WandB
            - Saves evaluation files
            - Sends summary notification

        Note:
            - Loads actual values from viewser data
            - Processes each task type separately (regression/classification)
            - Groups metrics by conflict type
            - Enforces scalar predictions for point metrics
            - Predictions stay as dataset objects; conversion to pandas
              happens only at the EvaluationManager boundary.
        """
        import wandb
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager
        from views_pipeline_core.files.utils import read_dataframe

        # --- Load actuals (pandas kept for prepare_actuals_df hook compat) ---
        if not ensemble:
            df_path = self._model_path._get_raw_data_file_paths(
                run_type=self.args.run_type
            )[0]
        else:
            # Local import to avoid circular dependency with model.py
            from views_pipeline_core.managers.model.model import ModelPathManager

            df_path = (
                ModelPathManager(self.configs["models"][0]).data_raw
                / f"{self.configs['run_type']}_viewser_df{PipelineConfig().dataframe_format}"
            )

        df_viewser = read_dataframe(df_path)
        logger.info(f"df_viewser read from {df_path}")
        df_viewser = self.prepare_actuals_df(df_viewser)

        all_targets = self.configs.get(
            "regression_targets", []
        ) + self.configs.get("classification_targets", [])
        df_actual = df_viewser[all_targets]

        # --- Normalise predictions to a list (datasets or DataFrames) ---
        list_preds = predictions if isinstance(predictions, list) else [predictions]

        tasks = {
            "regression": self.configs.get("regression_targets", []),
            "classification": self.configs.get("classification_targets", []),
        }

        evaluation_manager = EvaluationManager()

        for task_type, targets in tasks.items():
            if not targets:
                continue

            logger.info(f"Processing {task_type} tasks for evaluation...")

            for target in targets:
                logger.info(
                    f"Calculating {task_type} evaluation metrics for {target}"
                )

                # Column presence check — works with datasets and DataFrames
                first = list_preds[0]
                if isinstance(first, SpatioTemporalDataset):
                    has_col = f"pred_{target}" in first.columns
                else:
                    has_col = f"pred_{target}" in first.columns  # pd / pl DataFrame
                if not has_col:
                    logger.warning(
                        f"Column pred_{target} not found in prediction columns. Skipping."
                    )
                    continue

                target_identifier = target

                # --- Convert predictions to pandas at the EvaluationManager boundary ---
                raw_pred_dfs = []
                for ds in list_preds:
                    if isinstance(ds, SpatioTemporalDataset):
                        df = self._dataset_to_pandas(ds)
                    elif isinstance(ds, pl.LazyFrame):
                        df = ds.collect().to_pandas()
                    elif isinstance(ds, pl.DataFrame):
                        df = ds.to_pandas()
                    else:
                        df = ds  # already pandas
                    raw_pred_dfs.append(df[[f"pred_{target}"]])

                eval_result_dict = evaluation_manager.evaluate(
                    df_actual[[target]],
                    raw_pred_dfs,
                    target,
                    self.configs,
                )

                # Initialize local variables to avoid UnboundLocalError
                step_wise_evaluation, df_step_wise_evaluation = (
                    {},
                    pd.DataFrame(),
                )
                time_series_wise_evaluation, df_time_series_wise_evaluation = (
                    {},
                    pd.DataFrame(),
                )
                month_wise_evaluation, df_month_wise_evaluation = (
                    {},
                    pd.DataFrame(),
                )

                # Safety check: ensure all expected keys are present
                for eval_key in ["step", "time_series", "month"]:
                    try:
                        res = eval_result_dict[eval_key]

                        if not isinstance(res, (list, tuple)) or len(res) < 2:
                            raise ValueError(
                                f"Expected 2-tuple, got {type(res)} with length "
                                f"{len(res) if hasattr(res, '__len__') else 'N/A'}"
                            )

                        if eval_key == "step":
                            step_wise_evaluation, df_step_wise_evaluation = res
                        elif eval_key == "time_series":
                            (
                                time_series_wise_evaluation,
                                df_time_series_wise_evaluation,
                            ) = res
                        elif eval_key == "month":
                            month_wise_evaluation, df_month_wise_evaluation = res

                    except (KeyError, TypeError, ValueError, IndexError) as e:
                        logger.warning(
                            f"Evaluation for {target} returned invalid data for "
                            f"'{eval_key}': {e}. Skipping WandB/File logging for "
                            "this component."
                        )

                self._wandb_module.log_evaluation_results(
                    step_wise_evaluation,
                    month_wise_evaluation,
                    time_series_wise_evaluation,
                    target_identifier,
                )

                if not self.configs["sweep"]:
                    self._save_evaluations(
                        df_step_wise_evaluation,
                        df_time_series_wise_evaluation,
                        df_month_wise_evaluation,
                        self._model_path.data_generated,
                        target_identifier,
                    )

        self._wandb_module.send_alert(
            title=f"Metrics for {self._model_path.model_name}",
            text=f"{self._generate_evaluation_table(wandb.summary._as_dict())}",
            notifications_enabled=self._wandb_notifications,
        )

    def _generate_evaluation_table(self, metric_dict: Dict) -> str:
        """
        Format metrics as markdown table.

        Creates readable table from WandB summary metrics for
        notifications and reports.

        Internal Use:
            Called when sending evaluation notifications.

        Args:
            metric_dict: WandB summary metrics dictionary

        Returns:
            Formatted markdown table string

        Example:
            >>> table = self._generate_evaluation_table(wandb.summary._as_dict())
            >>> print(table)
            ```
            | Metric | Value |
            |--------|-------|
            | MSE    | 0.045 |
            ```
        """
        from tabulate import tabulate

        metric_df = pd.DataFrame(columns=["Metric", "Value"])
        for key, value in metric_dict.items():
            try:
                if not str(key).startswith("_"):
                    value = float(value)
                    metric_df = pd.concat(
                        [metric_df, pd.DataFrame([{"Metric": key, "Value": value}])],
                        ignore_index=True,
                    ).sort_values(by="Metric")
            except Exception:
                continue
        result = tabulate(metric_df, headers="keys", tablefmt="grid")
        print(result)
        return f"```\n{result}\n```"
