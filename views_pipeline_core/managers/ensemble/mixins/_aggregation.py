"""AggregationMixin — ensemble aggregation and reconciliation.

Extracted from EnsembleManager. Handles aggregating predictions from
multiple constituent models and applying reconciliation.
"""
from __future__ import annotations

import logging
import tqdm
from typing import Dict, List, Optional

import pandas as pd

from views_pipeline_core.modules.aggregation.aggregator import AggregationModule
from views_pipeline_core.data.handlers import _ViewsDataset, _PGDataset
from views_pipeline_core.managers.configuration.configuration import combined_targets
from views_pipeline_core.exceptions import PipelineException

logger = logging.getLogger(__name__)

# ADR-034: priogrid_gid → priogrid_id rename
_ENTITY_RENAME = {"priogrid_gid": "priogrid_id"}


class AggregationMixin:
    """Mixin providing aggregation and reconciliation methods for EnsembleManager."""

    def _evaluate_ensemble(self) -> List[pd.DataFrame]:
        """Evaluate all constituent models and aggregate their predictions."""


        eval_results: Dict[str, List[pd.DataFrame]] = {}
        for model_name in tqdm.tqdm(self.configs["models"], desc="Evaluating ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            eval_results[model_name] = self._evaluate_model_artifact(model_name)
            for j, df in enumerate(eval_results[model_name]):
                months = df.index.get_level_values("month_id")
                logger.info(
                    f"AFTER _evaluate_model_artifact | model={model_name} | "
                    f"seq={j:02d} | month_id=[{months.min()}, {months.max()}] | n={len(df)}"
                )

        n_outputs = len(next(iter(eval_results.values())))
        aggregated_outputs: List[pd.DataFrame] = []

        tqdm.tqdm.write("Aggregating metrics...")
        for i in range(n_outputs):
            model_dfs_i = {}
            for model_name, dfs in eval_results.items():
                if i >= len(dfs):
                    raise ValueError(
                        f"Model '{model_name}' returned only {len(dfs)} outputs, "
                        f"but at least {i+1} are required."
                    )
                model_dfs_i[model_name] = dfs[i]
            df_agg = self._get_aggregated_df(
                df_to_aggregate=model_dfs_i,
                aggregation=self.configs["aggregation"],
            )
            aggregated_outputs.append(df_agg)

        return aggregated_outputs

    def _forecast_ensemble(self) -> pd.DataFrame:
        """Generate ensemble forecasts, aggregate, and optionally reconcile."""


        model_dfs: Dict[str, pd.DataFrame] = {}
        for model_name in tqdm.tqdm(self.configs["models"], desc="Forecasting ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            df = self._forecast_model_artifact(model_name)
            model_dfs[model_name] = df

        df_prediction = self._get_aggregated_df(
            df_to_aggregate=model_dfs, aggregation=self.configs["aggregation"]
        )
        df_prediction = _ViewsDataset(source=df_prediction).dataframe

        # Apply reconciliation if configured
        if self._EnsembleManager__activate_reconciliation:
            df_prediction = self._apply_reconciliation(df_prediction)

        if not isinstance(df_prediction, pd.DataFrame):
            raise TypeError(
                f"Expected predictions to be a DataFrame, got {type(df_prediction)} instead."
            )

        return df_prediction

    def _get_aggregated_df(
        self,
        df_to_aggregate: Dict[str, pd.DataFrame],
        aggregation: str,
    ) -> pd.DataFrame:
        """Aggregate model predictions using the AggregationModule."""
        if not df_to_aggregate:
            raise ValueError("df_to_aggregate must contain at least one DataFrame.")

        first_df = next(iter(df_to_aggregate.values()))
        index_cols = [_ENTITY_RENAME.get(c, c) for c in first_df.index.names]
        target_cols = ["pred_" + col for col in combined_targets(self.configs)]

        agg = AggregationModule(index_cols=index_cols, target_cols=target_cols)
        use_weights = self.configs.get("use_weights", False)
        weights_cfg = self.configs.get("weights", {})

        for model_name, df in df_to_aggregate.items():
            agg.add_model(data=df, weight=weights_cfg.get(model_name), name=model_name)

        pred_type = agg.prediction_type
        if pred_type is None:
            raise RuntimeError("AggregationModule.prediction_type is None.")

        aggregated_pl = agg.aggregate(method=aggregation, use_weights=use_weights)
        aggregated_pdf = aggregated_pl.to_pandas()
        aggregated_pdf = aggregated_pdf.set_index(index_cols).sort_index()
        return aggregated_pdf

    def _apply_reconciliation(self, df_prediction: pd.DataFrame) -> pd.DataFrame:
        """Apply reconciliation to predictions if configured."""
        import wandb
        from views_pipeline_core.modules.reconciliation import RECONCILER_NOT_INJECTED_MSG

        reconciliation_type = self.configs.get("reconciliation", None)

        if reconciliation_type == "pgm_cm_point":
            reconciled_pg = self._EnsembleManager__reconcile_pg_with_c(pg_dataframe=df_prediction)
            if reconciled_pg is not None:
                logger.info(f"Reconciliation complete for {self._model_path.target}.")
                self._wandb_module.send_alert(
                    title=f"{self._model_path.target.title()} reconciliation complete",
                    level=wandb.AlertLevel.INFO,
                )
                return reconciled_pg
            else:
                logger.error("Reconciliation configured but failed.")
                raise PipelineException(
                    "Reconciliation configured but failed. C dataset could not be loaded.",
                    wandb_module=self._wandb_module,
                )
        else:
            logger.info("No valid reconciliation type specified. Skipping.")

        return df_prediction

    def _EnsembleManager__reconcile_pg_with_c(
        self, pg_dataframe: pd.DataFrame = None, c_dataframe: pd.DataFrame = None
    ) -> Optional[pd.DataFrame]:
        """Reconcile PG dataset with C dataset via injected Reconciler."""
        from views_pipeline_core.modules.reconciliation import RECONCILER_NOT_INJECTED_MSG

        cm_model = self.configs.get("reconcile_with", None)
        if cm_model is None:
            logger.info("No reconciliation model specified. Skipping.")
            return None

        if self._reconciler is None:
            raise PipelineException(
                RECONCILER_NOT_INJECTED_MSG,
                wandb_module=self._wandb_module,
            )

        latest_c_dataset = self._load_c_dataset(cm_model, c_dataframe)
        if latest_c_dataset is None:
            return None

        latest_pg_dataset = (
            _PGDataset(
                source=self._model_path.get_generated_predictions_data_file_paths(
                    run_type=self.configs["run_type"]
                )[0]
            )
            if pg_dataframe is None
            else _PGDataset(source=pg_dataframe)
        )

        if latest_pg_dataset is None:
            logger.error("Could not find latest PG dataset.")
            return None

        from views_pipeline_core.modules.reconciliation.adapter import reconcile_datasets
        return reconcile_datasets(self._reconciler, latest_c_dataset, latest_pg_dataset)
