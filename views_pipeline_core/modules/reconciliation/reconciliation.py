import concurrent
from collections import defaultdict
from concurrent.futures import as_completed
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
from tqdm import tqdm
import wandb
import torch
import logging

from views_pipeline_core.modules.dataset.core import (
    CountryDataset,
    CountryMonthDataset,
    PriogridDataset,
    PriogridMonthDataset,
    SpatioTemporalDataset,
)
from views_pipeline_core.modules.dataset.reconciliation import ReconciliationModule as _DatasetReconciler
from views_pipeline_core.modules.statistics import ForecastReconciler
from views_pipeline_core.modules.wandb import WandBModule

logger = logging.getLogger(__name__)


class ReconciliationModule:
    """
    Hierarchical forecast reconciliation between country and grid levels.

    Reconciles predictions across geographic hierarchies using proportional
    scaling to ensure country-level totals match while preserving grid-level
    spatial patterns. Uses Polars LazyFrames and PatchStore for memory-efficient
    updates without materialising the full dataset.
    """

    def __init__(
        self,
        c_dataset: Union[CountryDataset, CountryMonthDataset],
        pg_dataset: Union[PriogridDataset, PriogridMonthDataset],
        wandb_notifications: bool = True,
    ):
        """
        Initialize reconciliation module with country and grid datasets.

        Args:
            c_dataset: Country-level dataset (new Polars-native class).
            pg_dataset: Grid-level dataset with country mapping loaded.
            wandb_notifications: Whether to send WandB alerts during processing.

        Raises:
            TypeError: If datasets are not the expected types.
            ValueError: If datasets have incompatible structures.
        """
        self._c_dataset = c_dataset
        self._pg_dataset = pg_dataset
        self._wandb_notifications = wandb_notifications

        if not isinstance(c_dataset, CountryDataset):
            raise TypeError(f"Expected CountryDataset, got {type(c_dataset)}")
        if not isinstance(pg_dataset, PriogridDataset):
            raise TypeError(f"Expected PriogridDataset, got {type(pg_dataset)}")

        if pg_dataset._metadata is None:
            raise ValueError(
                "PriogridDataset must have country mapping loaded. "
                "Pass country_mapping= at construction time."
            )

        self._device = self._detect_torch_device()
        logger.info(f"Using device: {self._device}")
        self._reconciler = ForecastReconciler(device=self._device)

        # Validate temporal alignment
        c_times = set(c_dataset._unique_times)
        pg_times = set(pg_dataset._unique_times)

        if c_dataset.time_col != pg_dataset.time_col:
            raise ValueError(
                f"Datasets have different time columns: "
                f"'{c_dataset.time_col}' vs '{pg_dataset.time_col}'"
            )

        uncommon = c_times ^ pg_times
        if uncommon:
            logger.warning(
                f"Datasets have {len(uncommon)} non-overlapping time steps. "
                "Only common time steps will be reconciled."
            )

        self._valid_time_ids: List[int] = sorted(c_times & pg_times)
        if not self._valid_time_ids:
            raise ValueError("No overlapping time periods between datasets.")

        # Determine valid country IDs (countries with known grid mappings
        # that are also present in the country dataset)
        mapped_countries = set(pg_dataset._metadata._country_to_entities.keys())
        c_entities = set(c_dataset._unique_entities)
        self._valid_cids: List[int] = sorted(mapped_countries & c_entities)

        if not self._valid_cids:
            raise ValueError(
                "No valid countries for reconciliation. "
                "Check that country_mapping covers entities in the country dataset."
            )

        # Determine valid targets (prediction columns in both datasets)
        c_targets = set(c_dataset.get_pred_vars())
        pg_targets = set(pg_dataset.get_pred_vars())
        self._valid_targets: Set[str] = c_targets & pg_targets

        if not self._valid_targets:
            raise ValueError(
                "No common prediction targets between datasets. "
                "Ensure both have pred_* columns in common."
            )

        WandBModule.send_alert(
            title=self.__class__.__name__,
            text=(
                f"All checks passed. Starting reconciliation with "
                f"{len(self._valid_cids)} valid countries and "
                f"{len(self._valid_time_ids)} valid time IDs for targets: "
                f"{self._valid_targets}"
            ),
            notifications_enabled=self._wandb_notifications,
        )

    @staticmethod
    def _detect_torch_device() -> torch.device:
        """Detect best available PyTorch device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _reconcile_country_worker(args) -> Tuple[int, int, str, np.ndarray, List[int]]:
        """
        Worker function for parallel reconciliation.

        Args:
            args: Tuple of (country_id, time_id, feature, pg_values,
                  c_values, entity_ids, lr, max_iters, tol, device_str)

        Returns:
            Tuple of (country_id, time_id, feature, reconciled_values, entity_ids)
        """
        (
            country_id, time_id, feature,
            pg_values, c_values, entity_ids,
            lr, max_iters, tol, device_str
        ) = args

        device = torch.device(device_str)
        reconciler = ForecastReconciler(device=device)

        pg_tensor = torch.from_numpy(pg_values).float().to(device)
        c_tensor = torch.from_numpy(c_values).float().to(device)

        reconciled_tensor = reconciler.reconcile_forecast(
            grid_forecast=pg_tensor,
            country_forecast=c_tensor,
            lr=lr,
            max_iters=max_iters,
            tol=tol,
        )

        return (
            country_id, time_id, feature,
            reconciled_tensor.cpu().numpy(),
            entity_ids,
        )

    def reconcile(
        self,
        lr: float = 0.01,
        max_iters: int = 500,
        tol: float = 1e-6,
        max_workers: Optional[int] = None,
    ) -> pl.LazyFrame:
        """
        Reconcile forecasts for all valid countries, time periods, and targets.

        Extracts tensors directly from the datasets (no re-wrapping),
        reconciles via ForecastReconciler, and stores results as patches
        in the PriogridDataset's PatchStore.

        Args:
            lr: Learning rate for reconciliation optimizer.
            max_iters: Maximum optimization iterations.
            tol: Convergence tolerance.
            max_workers: Maximum parallel workers (None = auto).

        Returns:
            pl.LazyFrame: Reconciled grid-level predictions (lazy).
        """
        device_str = str(self._device)
        valid_targets = sorted(self._valid_targets)
        num_total_tasks = (
            len(self._valid_cids)
            * len(self._valid_time_ids)
            * len(valid_targets)
        )
        country_task_counts = {
            cid: len(self._valid_time_ids) * len(valid_targets)
            for cid in self._valid_cids
        }

        failed_tasks = []
        country_completion_progress = defaultdict(int)
        completed_countries = set()

        num_of_workers = (
            max_workers
            if max_workers is not None
            else min(32, (os.cpu_count() or 4) + 4)
        )
        logger.info(
            f"Start multiprocessing reconciliation with {num_of_workers} workers..."
        )

        # Pre-extract tensors per country to avoid redundant per-task collection.
        # Each country's grids are extracted once for all (time, feature) combos.
        task_args_list = []
        for country_id in tqdm(self._valid_cids, desc="Preparing tasks"):
            entity_ids = self._pg_dataset._metadata.get_entities_for_country(country_id)
            if not entity_ids:
                continue

            for time_id in self._valid_time_ids:
                for feature in valid_targets:
                    # Extract PG tensor: (n_samples, n_grids)
                    pg_values, _ = self._pg_dataset.to_reconciler(
                        feature=feature, time_id=time_id, country_id=country_id,
                    )
                    # Extract C tensor: (n_samples, 1)
                    c_values = self._c_dataset.to_reconciler(
                        feature=feature, time_id=time_id, country_id=country_id,
                    )

                    task_args_list.append((
                        country_id, time_id, feature,
                        pg_values, c_values, entity_ids,
                        lr, max_iters, tol, device_str,
                    ))

        logger.info(
            f"All {len(task_args_list)} tasks prepared. Submitting to executor..."
        )

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_workers) as executor:
            future_to_task_info = {}
            for task_args in task_args_list:
                country_id, time_id, feature = task_args[0], task_args[1], task_args[2]
                future = executor.submit(
                    ReconciliationModule._reconcile_country_worker, task_args
                )
                future_to_task_info[future] = (country_id, time_id, feature)

            for future in tqdm(
                as_completed(future_to_task_info),
                desc="Reconciling Tasks",
                total=len(future_to_task_info),
            ):
                country_id, time_id, feature = future_to_task_info[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(
                        f"Task failed for country {country_id}, "
                        f"time {time_id}, feature {feature}: {e}"
                    )
                    failed_tasks.append((country_id, time_id, feature))
                    WandBModule.send_alert(
                        title=self.__class__.__name__,
                        text=(
                            f"Task failed for country {country_id}, "
                            f"time {time_id}, feature {feature}: {e}"
                        ),
                        level=wandb.AlertLevel.ERROR,
                    )

                country_completion_progress[country_id] += 1
                if country_completion_progress[country_id] == country_task_counts.get(country_id, 0):
                    completed_countries.add(country_id)
                    num_done = len(completed_countries)
                    if num_done % 10 == 0 or num_done == len(self._valid_cids):
                        logger.info(
                            f"Reconciliation complete for "
                            f"{num_done}/{len(self._valid_cids)} countries"
                        )

        if failed_tasks:
            logger.warning(
                f"{len(failed_tasks)} tasks failed during reconciliation."
            )

        # Apply results via PatchStore (disk-backed, no full-frame mutation)
        logger.info(f"Updating dataset with {len(results)} successful results...")
        for country_id, time_id, feature, reconciled_values, entity_ids in tqdm(
            results, desc="Updating dataset"
        ):
            self._pg_dataset.reconcile(
                country_id=country_id,
                feature=feature,
                reconciled_values=reconciled_values,
                time_id=time_id,
            )

        logger.info("All reconciliations have been successfully completed.")
        WandBModule.send_alert(
            title=self.__class__.__name__,
            text="All reconciliations have been successfully completed.",
        )
        return self._pg_dataset.reconciled_lazy_frame