"""
Ensemble aggregation and reconciliation mixin.

Contains the methods for aggregating predictions from multiple ensemble members
(both legacy and ``AggregationModule``-based) and for reconciling PG-level
predictions with country-level predictions.

All prediction data flows through ``SpatioTemporalDataset`` objects internally.
Pandas DataFrames are only produced at external-API boundaries where required.
"""

import logging
from typing import Optional, List, Dict, Union

import pandas as pd
import polars as pl
import wandb

from views_pipeline_core.modules.reconciliation.reconciliation import (
    ReconciliationModule,
)
from views_pipeline_core.modules.dataset.core import (
    SpatioTemporalDataset,
    PriogridMonthDataset,
    CountryMonthDataset,
)
from views_pipeline_core.modules.aggregation import AggregationModule

logger = logging.getLogger(__name__)


class EnsembleAggregationMixin:
    """Aggregation and reconciliation helpers for *EnsembleManager*."""

    # ============================================================
    # AGGREGATION
    # ============================================================

    @staticmethod
    def _get_aggregated_df_old(
        df_to_aggregate: List[pd.DataFrame], aggregation: str
    ) -> pd.DataFrame:
        """
        Aggregates DataFrames using mean or median aggregation.
        Handles single-element lists by converting to scalars.

        Args:
            df_to_aggregate (List[pd.DataFrame]): List of DataFrames to
                aggregate.
            aggregation (str): Aggregation method ('mean' or 'median').

        Returns:
            pd.DataFrame: Aggregated DataFrame.
        """
        processed_dfs = []

        for df in df_to_aggregate:
            df_processed = df.copy()

            for col in df_processed.columns:

                def process_element(elem):
                    if isinstance(elem, list):
                        if len(elem) == 1:
                            return elem[0]
                        elif len(elem) == 0:
                            return None
                        else:
                            raise ValueError(
                                f"Aggregating distributions is not supported. "
                                f"Found list with {len(elem)} values in column '{col}'."
                            )
                    return elem

                df_processed[col] = df_processed[col].apply(process_element)

            processed_dfs.append(df_processed)

        concatenated = pd.concat(processed_dfs)

        if aggregation == "mean":
            return concatenated.groupby(level=[0, 1]).mean()
        elif aggregation == "median":
            return concatenated.groupby(level=[0, 1]).median()
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation}")

    def _get_aggregated_df(
        self,
        df_to_aggregate: Dict[str, Union[SpatioTemporalDataset, pd.DataFrame]],
        aggregation: str,
    ) -> SpatioTemporalDataset:
        """
        Aggregate model predictions using the AggregationModule.

        Accepts dictionaries mapping model names to either
        ``SpatioTemporalDataset`` objects (recommended) or legacy
        ``pd.DataFrame`` objects.  Datasets are passed directly to the
        ``AggregationModule`` — no pandas round-trip.

        Args:
            df_to_aggregate: Dict mapping model names to prediction datasets
                or DataFrames (all covering the same index).
            aggregation:
                - for *point* predictions: ``'mean'``, ``'median'``,
                  ``'min'``, ``'max'``, or custom name.
                - for *distribution* predictions: ``'concat'`` or
                  ``'vincentization'``.

        Returns:
            Aggregated predictions as a ``SpatioTemporalDataset``.
        """

        if not df_to_aggregate:
            raise ValueError("df_to_aggregate must contain at least one DataFrame.")

        # ---- 1) Define index + target columns --------------------------------
        first_item = next(iter(df_to_aggregate.values()))
        if isinstance(first_item, SpatioTemporalDataset):
            index_cols = [first_item.time_col, first_item.entity_col]
        else:
            # Legacy pd.DataFrame case
            index_cols = list(first_item.index.names)
        target_cols = ["pred_" + col for col in self.configs.get("targets")]

        # ---- 2) Create AggregationModule -------------------------------------
        manager = AggregationModule(
            index_cols=index_cols,
            target_cols=target_cols,
        )

        # ---- 3) Add each model to the manager --------------------------------
        use_weights = self.configs.get("use_weights", False)
        weights_cfg = self.configs.get("weights", {})  # dict: {model_name: weight}

        for model_name, df in df_to_aggregate.items():
            weight = weights_cfg.get(model_name)  # may be None
            manager.add_model(
                data=df,
                weight=weight,
                name=model_name,
            )

        # ---- 4) Decide how to call aggregate() based on prediction type ------
        pred_type = manager.prediction_type
        if pred_type is None:
            raise RuntimeError(
                "AggregationModule.prediction_type is None. "
                "Make sure at least one model was added with `add_model`."
            )

        aggregated_pl = manager.aggregate(
            method=aggregation,
            use_weights=use_weights,
        )

        # ---- 5) Wrap result in a SpatioTemporalDataset -----------------------
        return self._coerce_to_dataset(aggregated_pl)

    # ============================================================
    # RECONCILIATION
    # ============================================================

    def _apply_reconciliation(
        self, prediction: SpatioTemporalDataset,
    ) -> SpatioTemporalDataset:
        """
        Apply reconciliation to predictions if configured.

        Args:
            prediction: The prediction dataset to reconcile.

        Returns:
            SpatioTemporalDataset: Reconciled or original prediction dataset.
        """

        reconciliation_type = self.configs.get("reconciliation", None)

        if reconciliation_type == "pgm_cm_point":
            reconciled_ds = self.__reconcile_pg_with_c(pg_data=prediction)

            if reconciled_ds is not None:
                logger.info(
                    f"Reconciliation complete for {self._model_path.target}. "
                    "Predictions reconciled with C dataset."
                )
                self._wandb_module.send_alert(
                    title=f"{self._model_path.target.title()} reconciliation complete",
                    level=wandb.AlertLevel.INFO,
                )
                return reconciled_ds
            else:
                self._wandb_module.send_alert(
                    title=f"{self._model_path.target.title()} Reconciliation Error",
                    text="Reconciliation returned None. Predictions not reconciled.",
                    level=wandb.AlertLevel.WARNING,
                )
                logger.warning(
                    "Reconciliation returned None. Predictions not reconciled."
                )
        else:
            logger.info(
                "No valid reconciliation type specified. Returning predictions without reconciliation."
            )

        return prediction

    def __reconcile_pg_with_c(
        self,
        pg_data: Union[SpatioTemporalDataset, pd.DataFrame, None] = None,
        c_dataframe: Optional[pd.DataFrame] = None,
    ) -> Optional[SpatioTemporalDataset]:
        """
        Reconciles the PG dataset with the C dataset using a specified
        reconciliation model.

        Accepts either a ``SpatioTemporalDataset`` or a legacy
        ``pd.DataFrame``.  The result is always wrapped in a
        ``PriogridMonthDataset`` for the caller.

        Args:
            pg_data: The PG prediction data to reconcile (dataset or
                DataFrame).
            c_dataframe: The C dataset to reconcile with (legacy path).

        Returns:
            Optional[SpatioTemporalDataset]: The reconciled PG dataset, or
            None if reconciliation fails.
        """
        cm_model = self.configs.get("reconcile_with", None)
        if cm_model is None:
            logger.info("No reconciliation model specified. Skipping reconciliation.")
            return None

        # Load C dataset
        latest_c_dataset = self._load_c_dataset(cm_model, c_dataframe)
        if latest_c_dataset is None:
            return None

        # Build PG dataset from input data
        if isinstance(pg_data, PriogridMonthDataset):
            latest_pg_dataset = pg_data
            # Ensure metadata is available for reconciliation
            if not pg_data._pg_meta.get_all_countries():
                pg_data._pg_meta.fetch()
        elif isinstance(pg_data, SpatioTemporalDataset):
            latest_pg_dataset = PriogridMonthDataset(
                data=pg_data.collect(), fetch_metadata=True,
            )
        elif pg_data is not None:
            latest_pg_dataset = PriogridMonthDataset(
                data=pg_data, fetch_metadata=True,
            )
        else:
            latest_pg_dataset = PriogridMonthDataset(
                data=self._model_path._get_generated_predictions_data_file_paths(
                    run_type=self.configs["run_type"]
                )[0],
                fetch_metadata=True,
            )

        if latest_pg_dataset is None:
            logger.error(
                "Could not find latest PG dataset. Reconciliation cannot proceed."
            )
            return None

        # Perform reconciliation
        reconciliation_manager = ReconciliationModule(
            c_dataset=latest_c_dataset, pg_dataset=latest_pg_dataset
        )
        reconciled_df = reconciliation_manager.reconcile(
            lr=0.01, max_iters=500, tol=1e-6
        )

        # Wrap reconciled Polars DataFrame back into a dataset
        if reconciled_df is not None:
            return PriogridMonthDataset(
                data=reconciled_df, fetch_metadata=True,
            )
        return None

    def _load_c_dataset(
        self, cm_model: str, c_dataframe: Optional[pd.DataFrame]
    ) -> Optional[CountryMonthDataset]:
        """
        Load C dataset from prediction store, local path, or provided
        DataFrame.

        Args:
            cm_model (str): The C model name.
            c_dataframe (Optional[pd.DataFrame]): Optional DataFrame to use.

        Returns:
            Optional[CountryMonthDataset]: The loaded C dataset or None.
        """
        if c_dataframe is not None:
            logger.info(f"Using provided C dataset for model {cm_model}")
            return CountryMonthDataset(data=c_dataframe)

        if self._use_prediction_store:
            try:
                from views_forecasts.extensions import ViewsMetadata

                logger.info(
                    f"Fetching latest C dataset for {cm_model} from prediction store"
                )
                run_id = ViewsMetadata().get_run_id_from_name(self._pred_store_name)
                all_runs = ViewsMetadata().with_name(cm_model).fetch()["name"].to_list()

                reconcile_with_forecasts = [
                    fc for fc in all_runs if cm_model in fc and "forecasting" in fc
                ]
                reconcile_with_forecasts.sort()
                reconcile_with_forecast = reconcile_with_forecasts[-1]

                return CountryMonthDataset(
                    data=pd.DataFrame.forecasts.read_store(
                        run=run_id, name=reconcile_with_forecast
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Could not find latest C dataset for {cm_model} in prediction store: {e}"
                )

        # Try local path — use local import to avoid circular dependency
        try:
            from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager

            logger.info(f"Fetching latest C dataset for {cm_model} from local path")
            return CountryMonthDataset(
                data=EnsemblePathManager(
                    cm_model
                )._get_generated_predictions_data_file_paths(
                    run_type=self.configs["run_type"]
                )[
                    0
                ]
            )
        except Exception as e:
            logger.warning(
                f"Could not find latest C dataset for {cm_model} locally: {e}"
            )
            return None
