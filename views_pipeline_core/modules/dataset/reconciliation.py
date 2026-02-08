"""
Reconciliation Module
=====================

Wrapper around the existing ForecastReconciler for dataset operations.

This module provides a dataset-friendly interface to the reconciliation
functionality already defined in views_pipeline_core.modules.statistics.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .exceptions import ReconciliationError

# Import the existing reconciler
try:
    import torch
    from views_pipeline_core.modules.statistics.statistics import ForecastReconciler
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    ForecastReconciler = None
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from .metadata import MetadataModule


class ReconciliationModule:
    """Wrapper around ForecastReconciler for dataset operations.
    
    Provides hierarchical forecast reconciliation (Grid ↔ Country) using
    the existing ForecastReconciler implementation.
    
    Example:
        >>> reconciler = ReconciliationModule(metadata)
        >>> reconciled = reconciler.reconcile_proportional(
        ...     grid_values, country_total=1000.0
        ... )
    """
    
    def __init__(
        self,
        metadata: "MetadataModule",
        device: Optional[str] = None,
    ):
        """Initialize ReconciliationModule.
        
        Args:
            metadata: MetadataModule with country mappings loaded.
            device: Compute device ('cuda', 'cpu', or None for auto).
            
        Raises:
            ReconciliationError: If country mappings not available.
        """
        self.metadata = metadata
        self._logger = logging.getLogger(f"{__name__}.ReconciliationModule")
        
        if metadata._country_to_entities is None:
            raise ReconciliationError(
                "Country mappings required for reconciliation.",
                details={"hint": "Load metadata with country_id column"}
            )
        
        # Initialize the underlying reconciler
        if TORCH_AVAILABLE and ForecastReconciler is not None:
            self._reconciler = ForecastReconciler(device=device)
            self._device = device or self._reconciler.device
        else:
            self._reconciler = None
            self._device = "cpu"
            self._logger.warning(
                "PyTorch not available. Using numpy-based reconciliation."
            )
    
    @staticmethod
    def _transform_for_reconciliation(
        values: np.ndarray,
        feature_name: str,
    ) -> np.ndarray:
        """Apply transformation before reconciliation (exp for ln_ features).
        
        Reconciliation must be done in natural scale, so log-transformed
        features are exponentiated first.
        
        Args:
            values: Array of values.
            feature_name: Feature name (checked for ln_, lx_ prefix).
            
        Returns:
            Transformed values.
        """
        if "ln_" in feature_name:
            return np.exp(values)
        elif "lx_" in feature_name:
            return np.exp(values) - 1
        return values
    
    @staticmethod
    def _inverse_transform(
        values: np.ndarray,
        feature_name: str,
    ) -> np.ndarray:
        """Apply inverse transformation after reconciliation.
        
        Args:
            values: Reconciled values in natural scale.
            feature_name: Feature name.
            
        Returns:
            Values transformed back to original scale.
        """
        # Ensure non-negative for log
        values = np.maximum(values, 1e-10)
        
        if "ln_" in feature_name:
            return np.log(values)
        elif "lx_" in feature_name:
            return np.log(values + 1)
        return values
    
    def reconcile_proportional(
        self,
        grid_values: np.ndarray,
        country_total: float,
        preserve_zeros: bool = True,
    ) -> np.ndarray:
        """Reconcile grid values to match country total.
        
        Uses ForecastReconciler if available, otherwise falls back to
        numpy-based proportional scaling.
        
        Args:
            grid_values: Grid values with shape (n_samples, n_grids)
                        or (n_grids,).
            country_total: Target sum for the country.
            preserve_zeros: If True, zero values remain zero.
            
        Returns:
            Reconciled values with same shape as input.
        """
        is_1d = grid_values.ndim == 1
        if is_1d:
            grid_values = grid_values.reshape(1, -1)
        
        # Use PyTorch reconciler if available
        if self._reconciler is not None and TORCH_AVAILABLE:
            try:
                grid_tensor = torch.from_numpy(grid_values).float()
                country_tensor = torch.full(
                    (grid_values.shape[0],), 
                    country_total, 
                    dtype=torch.float32
                )
                
                reconciled = self._reconciler.reconcile_forecast(
                    grid_tensor, country_tensor
                )
                result = reconciled.cpu().numpy()
                
                return result.squeeze(0) if is_1d else result
            except Exception as e:
                self._logger.warning(
                    f"PyTorch reconciliation failed, using numpy: {e}"
                )
        
        # Fallback: numpy-based proportional scaling
        n_samples = grid_values.shape[0]
        reconciled = np.zeros_like(grid_values)
        
        for s in range(n_samples):
            vals = grid_values[s].copy()
            
            if preserve_zeros:
                mask = vals > 0
                total = vals[mask].sum()
                if total > 0:
                    vals[mask] *= country_total / total
            else:
                total = vals.sum()
                if total > 0:
                    vals *= country_total / total
            
            reconciled[s] = vals
        
        result = np.maximum(reconciled, 0.0)
        return result.squeeze(0) if is_1d else result
    
    def reconcile_mintrack(
        self,
        grid_values: np.ndarray,
        country_total: float,
        preserve_zeros: bool = True,
    ) -> np.ndarray:
        """Reconcile using minimum tracking (MinT) adjustment.
        
        Similar to proportional but minimizes variance of adjustments.
        
        Args:
            grid_values: Grid values with shape (n_samples, n_grids).
            country_total: Target sum for the country.
            preserve_zeros: If True, zero values remain zero.
            
        Returns:
            Reconciled values.
        """
        if grid_values.ndim == 1:
            grid_values = grid_values.reshape(1, -1)
        
        n_samples, n_grids = grid_values.shape
        reconciled = np.zeros_like(grid_values)
        
        for s in range(n_samples):
            vals = grid_values[s].copy()
            current_total = vals.sum()
            
            if current_total == 0:
                reconciled[s] = vals
                continue
            
            # MinT adjustment: distribute discrepancy proportionally
            discrepancy = country_total - current_total
            
            if preserve_zeros:
                mask = vals > 0
                n_active = mask.sum()
                if n_active > 0:
                    # Proportional share of discrepancy
                    weights = vals[mask] / vals[mask].sum()
                    vals[mask] += discrepancy * weights
            else:
                # Equal distribution
                vals += discrepancy / n_grids
            
            reconciled[s] = np.maximum(vals, 0.0)
        
        return reconciled
    
    def get_country_sum(
        self,
        grid_values: np.ndarray,
        entity_ids: List[int],
        country_id: int,
    ) -> np.ndarray:
        """Sum grid values for a specific country.
        
        Args:
            grid_values: Grid values with shape (..., n_grids).
            entity_ids: List of entity IDs corresponding to grid columns.
            country_id: Target country ID.
            
        Returns:
            Sum of values for grids belonging to the country.
        """
        country_entities = set(self.metadata.get_entities_for_country(country_id))
        
        # Find indices of grids belonging to this country
        mask = [eid in country_entities for eid in entity_ids]
        
        if not any(mask):
            return np.zeros(grid_values.shape[:-1])
        
        return grid_values[..., mask].sum(axis=-1)
    
    def reconcile_batch(
        self,
        grid_values: np.ndarray,
        country_totals: np.ndarray,
        entity_ids: List[int],
        country_ids: List[int],
        preserve_zeros: bool = True,
    ) -> np.ndarray:
        """Reconcile multiple countries in batch.
        
        Args:
            grid_values: Grid values with shape (n_samples, n_grids).
            country_totals: Target totals with shape (n_samples, n_countries).
            entity_ids: List of entity IDs for grid columns.
            country_ids: List of country IDs matching country_totals.
            preserve_zeros: If True, zero values remain zero.
            
        Returns:
            Reconciled grid values.
        """
        reconciled = grid_values.copy()
        
        for c_idx, country_id in enumerate(country_ids):
            country_entities = set(
                self.metadata.get_entities_for_country(country_id)
            )
            grid_mask = np.array([eid in country_entities for eid in entity_ids])
            
            if not grid_mask.any():
                continue
            
            # Extract grids for this country
            country_grids = reconciled[:, grid_mask]
            country_target = country_totals[:, c_idx]
            
            # Reconcile each sample
            for s in range(reconciled.shape[0]):
                vals = country_grids[s].copy()
                target = country_target[s]
                
                if preserve_zeros:
                    nonzero_mask = vals > 0
                    total = vals[nonzero_mask].sum()
                    if total > 0:
                        vals[nonzero_mask] *= target / total
                else:
                    total = vals.sum()
                    if total > 0:
                        vals *= target / total
                
                reconciled[s, grid_mask] = np.maximum(vals, 0.0)
        
        return reconciled


__all__ = ["ReconciliationModule"]
