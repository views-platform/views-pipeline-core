import sys
from views_pipeline_core.data.handlers import _CDataset, _PGDataset
import torch
import logging
from views_pipeline_core.data.statistics import ForecastReconciler

logger = logging.getLogger(__name__)

class ReconciliationManager:
    def __init__(self, c_dataset: _CDataset, pg_dataset: _PGDataset):
        self._c_dataset = c_dataset
        self._pg_dataset = pg_dataset
        self._device = self.__detect_torch_device()
        print(f"Using device: {self._device}")
        self._reconciler = ForecastReconciler(device=self._device)
        self._pg_dataset._build_country_to_grids_cache()

        if c_dataset.num_time_steps != pg_dataset.num_time_steps:
            raise ValueError(
                "The number of time steps in the country dataset and the grid dataset must match."
            )
        
        if c_dataset._time_id != pg_dataset._time_id:
            raise ValueError(
                f"You are trying to reconcile datasets with different time units. "
                f"Country dataset time unit: {c_dataset._time_id}, "
                f"Grid dataset time unit: {pg_dataset._time_id}"
            )
        
        if c_dataset._time_values != pg_dataset._time_values:
            raise ValueError(
                f"The time values in the country dataset and the grid dataset must match. Uncommon time steps: "
                f"{set(c_dataset._time_values) ^ set(pg_dataset._time_values)}"
            )

        self._valid_cids = list(
            set(self._pg_dataset._country_to_grids_cache.keys())
            & set(self._c_dataset._entity_values.to_list())
        )

        self._valid_targets = set(self._c_dataset.targets) & set(
            self._pg_dataset.targets
        )
        if not self._valid_targets:
            raise ValueError(
                "No valid targets to reconcile found in the datasets. "
                "Ensure that both datasets have at least one common target."
            )
        self._valid_time_ids = set(self._c_dataset._time_values) & set(
            self._pg_dataset._time_values
        )

    def __detect_torch_device(self):
        """
        Detect the best available PyTorch device.

        Returns:
            torch.device: The best available device (GPU, MPS, or CPU).
        """
        if torch.cuda.is_available():
            return torch.device("cuda")  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        else:
            return torch.device("cpu")  # Fallback to CPU

    def _reconcile_single_timestep(
        self,
        country_id: int,
        time_id: int,
        feature: str,
        lr: float,
        max_iters: int,
        tol=float,
    ):
        """
        Reconciles the forecast for a given country and time ID.
        """
        # Validate inputs
        if country_id not in self._valid_cids:
            raise ValueError(f"Invalid country ID: {country_id}")
        if time_id not in self._valid_time_ids:
            raise ValueError(f"Invalid time ID: {time_id}")
        if feature not in self._valid_targets:
            raise ValueError(f"Invalid feature: {feature}")

        pg_subset = self._pg_dataset.get_subset_by_country_id(country_ids=[country_id])
        c_subset = self._c_dataset.get_subset_dataframe(entity_ids=[country_id])

        c_subset_dataset = _CDataset(source=c_subset)
        pg_subset_dataset = _PGDataset(source=pg_subset)

        # Get the tensors for reconciliation
        pg_tensor = pg_subset_dataset.to_reconciler(feature=feature, time_id=time_id)
        c_tensor = c_subset_dataset.to_reconciler(
            feature=feature, time_id=time_id
        )

        # Perform reconciliation
        reconciled_tensor = self._reconciler.reconcile_forecast(
            grid_forecast=pg_tensor,
            country_forecast=c_tensor,
            lr=lr,
            max_iters=max_iters,
            tol=tol,
        )

        # Return the reconciled dataframe
        return reconciled_tensor

    def reconcile(self, lr=0.01, max_iters=500, tol=1e-6):
        """
        Reconciles the forecast for all valid country and time IDs.
        """
        for country_idx, country_id in enumerate(self._valid_cids, start=1):
            for time_idx, time_id in enumerate(self._valid_time_ids, start=1):
                for feature_idx, feature in enumerate(self._valid_targets, start=1):
                    # Update log in place
                    sys.stdout.write(
                        f"\r{' ' * 80}\r"  # Clear the previous line
                        f"Reconciling country {country_idx}/{len(self._valid_cids)}, "
                        f"time {time_idx}/{len(self._valid_time_ids)}, "
                        f"feature {feature_idx}/{len(self._valid_targets)}..."
                    )
                    sys.stdout.flush()
                    
                    self._pg_dataset.reconcile(
                        country_id=country_id, 
                        time_id=time_id, 
                        reconciled_tensor=self._reconcile_single_timestep(
                            country_id, time_id, feature, lr, max_iters, tol
                        ), 
                        feature=feature
                    )
        
        # Clear the line after completion
        sys.stdout.write("\rReconciliation complete.\n")
        sys.stdout.flush()
        
        return self._pg_dataset.reconciled_dataframe