import concurrent
from collections import defaultdict
from concurrent.futures import as_completed
import os
from tqdm import tqdm
import wandb
from views_pipeline_core.modules.dataset.core import CountryMonthDataset, PriogridMonthDataset
import torch
import numpy as np
import polars as pl
import logging
from views_pipeline_core.modules.statistics import ForecastReconciler
from views_pipeline_core.modules.wandb import WandBModule

logger = logging.getLogger(__name__)

class ReconciliationModule:
    """
    Hierarchical forecast reconciliation between country and grid levels.
    
    Reconciles predictions across geographic hierarchies using proportional
    scaling to ensure country-level totals match while preserving grid-level
    spatial patterns. Supports parallel processing for large-scale datasets.
    """
    def __init__(self, c_dataset: CountryMonthDataset, pg_dataset: PriogridMonthDataset, wandb_notifications: bool = True):
        """
        Initialize reconciliation module with country and grid datasets.

        Sets up reconciliation infrastructure including device detection,
        validation of dataset compatibility, and identification of valid
        reconciliation targets.

        Args:
            c_dataset: Country-level dataset with predictions to reconcile to
            pg_dataset: Grid-level dataset with predictions to reconcile from
            wandb_notifications: Whether to send WandB alerts during processing

        Raises:
            TypeError: If datasets are not correct types
            ValueError: If datasets have incompatible structures

        Example:
            >>> from views_pipeline_core.modules.dataset.core import CountryMonthDataset, PriogridMonthDataset
            >>> c_ds = CountryMonthDataset(country_predictions)
            >>> pg_ds = PriogridMonthDataset(grid_predictions, fetch_metadata=True)
            >>> reconciler = ReconciliationModule(c_ds, pg_ds)
            Using device: cuda
            All checks passed. Starting reconciliation with 180 valid countries...

        Note:
            - Automatically detects and uses GPU if available
            - Requires PGM dataset to have metadata loaded (via fetch_metadata=True or country_mapping)
            - Validates temporal and spatial alignment
            - Only reconciles targets present in both datasets
        """
        self._c_dataset = c_dataset
        self._pg_dataset = pg_dataset
        self._wandb_notifications = wandb_notifications
        if not isinstance(c_dataset, CountryMonthDataset):
            raise TypeError(f"Expected CountryMonthDataset, got {type(c_dataset)}")
        if not isinstance(pg_dataset, PriogridMonthDataset):
            raise TypeError(f"Expected PriogridMonthDataset, got {type(pg_dataset)}")

        self._device = self.__detect_torch_device()
        print(f"Using device: {self._device}")
        self._reconciler = ForecastReconciler(device=self._device)

        # Get country-to-grid mappings from PGM metadata
        # _pg_meta.get_all_countries() auto-fetches if not loaded
        mapped_country_ids = self._pg_dataset._pg_meta.get_all_countries()
        if not mapped_country_ids:
            raise ValueError(
                "PGM dataset has no country-to-grid mapping. "
                "Initialize with fetch_metadata=True or country_mapping=..."
            )

        if len(c_dataset._unique_times) != len(pg_dataset._unique_times):
            raise ValueError(
                "The number of time steps in the country dataset and the grid dataset must match."
            )
        
        if c_dataset.time_col != pg_dataset.time_col:
            raise ValueError(
                f"You are trying to reconcile datasets with different time units. "
                f"Country dataset time unit: {c_dataset.time_col}, "
                f"Grid dataset time unit: {pg_dataset.time_col}"
            )

        uncommon_time_steps = set(c_dataset._unique_times) ^ set(pg_dataset._unique_times)
        if uncommon_time_steps:
            raise ValueError(
                f"The datasets have different time steps: {uncommon_time_steps}. "
                "Ensure both datasets cover the same time periods."
            )

        self._valid_cids = list(
            set(mapped_country_ids)
            & set(self._c_dataset._unique_entities)
        )

        self._valid_targets = set(self._c_dataset.target_cols) & set(
            self._pg_dataset.target_cols
        )
        if not self._valid_targets:
            raise ValueError(
                "No valid targets to reconcile found in the datasets. "
                "Ensure that both datasets have at least one common target."
            )
        self._valid_time_ids = set(self._c_dataset._unique_times) & set(
            self._pg_dataset._unique_times
        )
        WandBModule.send_alert(
            title=self.__class__.__name__,
            text=f"All checks passed. Starting reconciliation with {len(self._valid_cids)} valid countries and {len(self._valid_time_ids)} valid time IDs for targets: {self._valid_targets}",
            notifications_enabled=self._wandb_notifications,
        )

    def __detect_torch_device(self):
        """
        Detect the best available PyTorch device.

        Internal Use:
            Called during initialization to select computation device.

        Returns:
            torch.device: The best available device:
                - 'cuda': NVIDIA GPU if available
                - 'mps': Apple Silicon GPU if available
                - 'cpu': CPU as fallback

        Note:
            - Prioritizes GPU acceleration when available
            - Automatically handles device compatibility
        """
        if torch.cuda.is_available():
            return torch.device("cuda")  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        else:
            return torch.device("cpu")  # Fallback to CPU

    # def _reconcile_single_timestep(
    #     self,
    #     country_id: int,
    #     time_id: int,
    #     feature: str,
    #     lr: float,
    #     max_iters: int,
    #     tol=float,
    # ):
    #     """
    #     Reconciles the forecast for a given country and time ID.
    #     """
    #     # Validate inputs
    #     if country_id not in self._valid_cids:
    #         raise ValueError(f"Invalid country ID: {country_id}")
    #     if time_id not in self._valid_time_ids:
    #         raise ValueError(f"Invalid time ID: {time_id}")
    #     if feature not in self._valid_targets:
    #         raise ValueError(f"Invalid feature: {feature}")

    #     pg_subset = self._pg_dataset.get_subset_by_country_id(country_ids=[country_id])
    #     c_subset = self._c_dataset.get_subset_dataframe(entity_ids=[country_id])

    #     c_subset_dataset = _CDataset(source=c_subset)
    #     pg_subset_dataset = _PGDataset(source=pg_subset)

    #     # Get the tensors for reconciliation
    #     pg_tensor = pg_subset_dataset.to_reconciler(feature=feature, time_id=time_id)
    #     c_tensor = c_subset_dataset.to_reconciler(
    #         feature=feature, time_id=time_id
    #     )

    #     # Perform reconciliation
    #     reconciled_tensor = self._reconciler.reconcile_forecast(
    #         grid_forecast=pg_tensor,
    #         country_forecast=c_tensor,
    #         lr=lr,
    #         max_iters=max_iters,
    #         tol=tol,
    #     )

    #     # Return the reconciled dataframe
    #     return reconciled_tensor

    # def reconcile(self, lr=0.01, max_iters=500, tol=1e-6):
    #     """
    #     Reconciles the forecast for all valid country and time IDs.
    #     """
    #     for country_idx, country_id in enumerate(self._valid_cids, start=1):
    #         for time_idx, time_id in enumerate(self._valid_time_ids, start=1):
    #             for feature_idx, feature in enumerate(self._valid_targets, start=1):
    #                 # Update log in place
    #                 sys.stdout.write(
    #                     f"\r{' ' * 80}\r"  # Clear the previous line
    #                     f"Reconciling country {country_idx}/{len(self._valid_cids)}, "
    #                     f"time {time_idx}/{len(self._valid_time_ids)}, "
    #                     f"feature {feature_idx}/{len(self._valid_targets)}..."
    #                 )
    #                 sys.stdout.flush()
                    
    #                 self._pg_dataset.reconcile(
    #                     country_id=country_id, 
    #                     time_id=time_id, 
    #                     reconciled_tensor=self._reconcile_single_timestep(
    #                         country_id, time_id, feature, lr, max_iters, tol
    #                     ), 
    #                     feature=feature
    #                 )

    #         if country_idx % 10 == 0 or country_idx == len(self._valid_cids):
    #             # logger.info(
    #             #     f"Reconciliation complete for country {country_id} ({country_idx}/{len(self._valid_cids)})"
    #             # )
    #             WandBModule.send_alert(
    #                 title=self.__class__.__name__,
    #                 text=f"Reconciliation complete for country {country_id} ({country_idx}/{len(self._valid_cids)})",
    #             )
        
    #     # Clear the line after completion
    #     sys.stdout.write("\rReconciliation complete.\n")
    #     sys.stdout.flush()
    #     WandBModule.send_alert(
    #         title=self.__class__.__name__,
    #         text="All reconciliations have been successfully completed."
    #     )
    #     return self._pg_dataset.reconciled_dataframe

    @staticmethod
    def _transform_to_natural(values: np.ndarray, feature: str) -> np.ndarray:
        """Transform from model scale to natural scale for reconciliation."""
        if "ln" in feature.split("_"):
            return np.exp(values) - 1
        elif "lx" in feature.split("_"):
            return np.exp(values) - np.exp(100)
        return values

    @staticmethod
    def _inverse_transform(values: np.ndarray, feature: str) -> np.ndarray:
        """Transform from natural scale back to model scale after reconciliation."""
        values = np.maximum(values, 1e-10)
        if "ln" in feature.split("_"):
            return np.log(values + 1)
        elif "lx" in feature.split("_"):
            return np.log(values + np.exp(-100))
        return values

    @staticmethod
    def _extract_tensor_from_pandas(df: "pd.DataFrame", feature: str, time_id: int) -> np.ndarray:
        """Extract values from a MultiIndex pandas DataFrame for reconciliation.
        
        Args:
            df: Pandas DataFrame with MultiIndex (time_col, entity_col).
            feature: Feature column to extract.
            time_id: Time step to extract.
            
        Returns:
            np.ndarray of shape (n_samples, n_entities) in natural scale.
        """
        # Get data for specific time_id (drops time level from index)
        time_data = df.xs(time_id, level=0)
        vals = time_data[feature].values
        
        # Handle array-valued columns (distributional predictions)
        if isinstance(vals[0], np.ndarray):
            data = np.stack(vals)  # (n_entities, n_samples)
            data = data.T  # (n_samples, n_entities)
        else:
            data = vals.reshape(1, -1)  # (1, n_entities) for point forecasts
        
        # Transform to natural scale
        return ReconciliationModule._transform_to_natural(data, feature)

    @staticmethod
    def _reconcile_country_worker(args):
        """
        Perform reconciliation for a single country-time-feature task.

        Internal Use:
            Worker function called by parallel executor in reconcile().

        Args:
            args: Tuple containing:
                - country_id (int): Country to reconcile
                - time_id (int): Time step to reconcile
                - feature (str): Target variable to reconcile
                - lr (float): Learning rate (currently unused)
                - max_iters (int): Max iterations (currently unused)
                - tol (float): Tolerance (currently unused)
                - c_subset (pd.DataFrame): Country data subset (MultiIndex pandas)
                - pg_subset (pd.DataFrame): Grid data subset (MultiIndex pandas)
                - device_str (str): Device string ('cuda', 'mps', 'cpu')

        Returns:
            Tuple of (country_id, time_id, feature, reconciled_values):
                - country_id: Input country ID
                - time_id: Input time ID
                - feature: Input feature name
                - reconciled_values: Reconciled grid predictions as numpy array
                  in natural scale, shape (n_samples, n_grids)

        Note:
            - Creates new ForecastReconciler instance per task
            - Extracts tensors directly from pandas DataFrames
            - Handles log transformations automatically
        """
        country_id, time_id, feature, lr, max_iters, tol, c_subset, pg_subset, device_str = args
        
        device = torch.device(device_str)
        reconciler = ForecastReconciler(device=device)

        # Extract values from pandas DataFrames
        pg_values = ReconciliationModule._extract_tensor_from_pandas(pg_subset, feature, time_id)
        c_values = ReconciliationModule._extract_tensor_from_pandas(c_subset, feature, time_id)
        c_values = c_values.squeeze(axis=-1)  # (n_samples,) since single entity

        # Convert to torch tensors for ForecastReconciler
        pg_tensor = torch.from_numpy(pg_values).float()
        c_tensor = torch.from_numpy(c_values).float()
        
        reconciled_tensor = reconciler.reconcile_forecast(
            grid_forecast=pg_tensor,
            country_forecast=c_tensor,
            lr=lr,
            max_iters=max_iters,
            tol=tol,
        )
        
        return country_id, time_id, feature, reconciled_tensor.cpu().numpy()
    
    def reconcile(self, lr=0.01, max_iters=500, tol=1e-6, max_workers=None):
        """
        Reconcile forecasts for all valid countries, time periods, and targets.

        Performs hierarchical reconciliation using parallel processing to ensure
        grid-level predictions sum to country-level totals while preserving
        spatial patterns and zero-inflation.

        Args:
            lr: Learning rate for optimization (currently unused). Default: 0.01
            max_iters: Maximum optimization iterations (currently unused). Default: 500
            tol: Convergence tolerance (currently unused). Default: 1e-6
            max_workers: Maximum parallel processes. If None, uses CPU count + 4.
                Recommended: Leave as None for automatic optimization.

        Returns:
            pd.DataFrame: Reconciled grid-level predictions with same structure
                as input pg_dataset, but with adjusted values that sum to
                country totals.

        Raises:
            RuntimeError: If too many tasks fail (currently logs but doesn't raise)

        Example:
            >>> reconciler = ReconciliationModule(country_ds, grid_ds)
            >>> reconciled = reconciler.reconcile(max_workers=16)
            Start multiprocessing reconciliation with 16 workers...
            All 54000 tasks have been submitted. Awaiting completion...
            Reconciling Tasks: 100%|██████████| 54000/54000
            Reconciliation complete for 10/180 countries
            ...
            All reconciliations have been successfully completed.

        Note:
            - Processes all combinations of (country, time, target)
            - Sends WandB alerts every 10 countries
            - Logs failed tasks but continues processing
            - Updates pg_dataset.reconciled_dataframe in-place
        """

        device_str = str(self._device)
        num_total_tasks = len(self._valid_cids) * len(self._valid_time_ids) * len(self._valid_targets)
        country_task_counts = {cid: len(self._valid_time_ids) * len(self._valid_targets) for cid in self._valid_cids}

        results = []
        failed_tasks = []
        country_completion_progress = defaultdict(int)
        completed_countries = set()

        num_of_workers = max_workers if max_workers is not None else min(32, os.cpu_count() + 4) # for version >=3.8 and <3.13
        logger.info(f"Start multiprocessing reconciliation with {num_of_workers} workers...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            future_to_task_info = {}

            for country_id in self._valid_cids:
                c_subset = self._c_dataset.get_subset_dataframe(
                    entity_ids=[country_id], return_pandas=True
                )
                pg_subset = self._pg_dataset.get_subset_by_country_id(
                    country_ids=[country_id], return_pandas=True
                )

                for time_id in self._valid_time_ids:
                    for feature in self._valid_targets:
                        task_args = (
                            country_id, time_id, feature, lr, max_iters, tol, 
                            c_subset, pg_subset, device_str
                        )
                        future = executor.submit(ReconciliationModule._reconcile_country_worker, task_args)
                        future_to_task_info[future] = (country_id, time_id, feature)

            logger.info(f"All {num_total_tasks} tasks have been submitted. Awaiting completion...")

            for future in tqdm(as_completed(future_to_task_info), desc="Reconciling Tasks", total=num_total_tasks):
                country_id, time_id, feature = future_to_task_info[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Task failed for country {country_id}, time {time_id}, feature {feature}: {e}")
                    failed_tasks.append((country_id, time_id, feature))
                    WandBModule.send_alert(
                        title=self.__class__.__name__,
                        text=f"Task failed for country {country_id}, time {time_id}, feature {feature}: {e}",
                        level=wandb.AlertLevel.ERROR,
                    )

                country_completion_progress[country_id] += 1

                if country_completion_progress[country_id] == country_task_counts[country_id]:
                    completed_countries.add(country_id)
                    num_done = len(completed_countries)
                    if num_done % 10 == 0 or num_done == len(self._valid_cids):
                        logger.info(f"Reconciliation complete for {num_done}/{len(self._valid_cids)} countries")

        if failed_tasks:
            logger.warning(f"{len(failed_tasks)} tasks failed during reconciliation. See logs for details.")
            # Depending on requirements, you might want to raise an error here.
            # raise RuntimeError(f"{len(failed_tasks)} reconciliation tasks failed.")
        
        logger.info(f"Updating dataset with {len(results)} successful results...")

        # Initialize reconciled dataframe if needed
        if self._pg_dataset.reconciled_dataframe is None:
            self._pg_dataset.reconciled_dataframe = self._pg_dataset.collect()

        for country_id, time_id, feature, reconciled_values in tqdm(results, desc="Updating dataset"):
            # reconciled_values is in natural scale (n_samples, n_grids)
            # Apply inverse transform to get back to model scale
            inv_values = self._inverse_transform(reconciled_values, feature)

            # Get entity IDs for this country
            entity_ids = self._pg_dataset._pg_meta.get_entities_for_country(country_id)

            # Update each grid cell in the reconciled dataframe
            for idx, entity_id in enumerate(entity_ids):
                new_samples = inv_values[:, idx].tolist()
                mask = (
                    (pl.col(self._pg_dataset.time_col) == time_id) &
                    (pl.col(self._pg_dataset.entity_col) == entity_id)
                )
                self._pg_dataset.reconciled_dataframe = self._pg_dataset.reconciled_dataframe.with_columns(
                    pl.when(mask)
                    .then(pl.lit(new_samples))
                    .otherwise(pl.col(feature))
                    .alias(feature)
                )
        
        logger.info("All reconciliations have been successfully completed.")
        WandBModule.send_alert(
            title=self.__class__.__name__,
            text="All reconciliations have been successfully completed."
        )
        return self._pg_dataset.reconciled_dataframe