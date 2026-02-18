"""
Reconciliation Module
=====================

Wrapper around the existing ForecastReconciler for dataset operations.

This module provides a dataset-friendly interface to the reconciliation
functionality already defined in views_pipeline_core.modules.statistics.
"""

from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch

from .exceptions import ReconciliationError
from views_pipeline_core.modules.statistics.statistics import ForecastReconciler

if TYPE_CHECKING:
    from .metadata import MetadataModule


class ReconciliationModule:
    """Wrapper around ForecastReconciler for dataset operations.
    
    Provides hierarchical forecast reconciliation (Grid ↔ Country) using
    the existing ForecastReconciler implementation.
    
    Example:
        >>> reconciler = ReconciliationModule(metadata)
        >>> reconciled = reconciler.reconcile(
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
        
        self._reconciler = ForecastReconciler(device=device)
        self._device = device or self._reconciler.device
    
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
        values = np.maximum(values, 1e-10)
        
        if "ln_" in feature_name:
            return np.log(values)
        elif "lx_" in feature_name:
            return np.log(values + 1)
        return values
    
    def reconcile(
        self,
        grid_values: np.ndarray,
        country_total: float,
    ) -> np.ndarray:
        """Reconcile grid values to match country total.
        
        Uses ForecastReconciler for proportional scaling that preserves
        zero values and relative patterns.
        
        Args:
            grid_values: Grid values with shape (n_samples, n_grids)
                        or (n_grids,).
            country_total: Target sum for the country.
            
        Returns:
            Reconciled values with same shape as input.
        """
        is_1d = grid_values.ndim == 1
        if is_1d:
            grid_values = grid_values.reshape(1, -1)
        
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
    ) -> np.ndarray:
        """Reconcile multiple countries in batch.
        
        Args:
            grid_values: Grid values with shape (n_samples, n_grids).
            country_totals: Target totals with shape (n_samples, n_countries).
            entity_ids: List of entity IDs for grid columns.
            country_ids: List of country IDs matching country_totals.
            
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
            
            # Convert to tensors
            grid_tensor = torch.from_numpy(country_grids).float()
            country_tensor = torch.from_numpy(country_target).float()
            
            # Reconcile using ForecastReconciler
            adjusted = self._reconciler.reconcile_forecast(
                grid_tensor, country_tensor
            )
            reconciled[:, grid_mask] = adjusted.cpu().numpy()
        
        return reconciled


__all__ = ["ReconciliationModule"]
