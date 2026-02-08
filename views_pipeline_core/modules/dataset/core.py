"""
Core Dataset Module
===================

Main SpatioTemporalDataset class and specialized variants.

This module provides:
    - SpatioTemporalDataset: Base class for spatiotemporal data
    - CountryMonthDataset: Country-month level data
    - PriogridMonthDataset: Priogrid-month level data with reconciliation
    - Factory functions for convenient loading

Supports three data layouts:
    - SCALAR: Single value per (time, entity) cell
    - ARRAY_COLUMN: Distributions as pl.Array/pl.List columns (recommended)
    - ROW_BASED: Distributions as repeated rows with sample_col
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import pandas as pd

from .exceptions import (
    DatasetError,
    ValidationError,
    IntegrityError,
    ReconciliationError,
    MetadataError,
    TensorConversionError,
)
from .loader import LoaderModule
from .index import IndexModule
from .grid import GridModule
from .subset import SubsetModule
from .tensor import TensorBundle, TensorConverter, TensorModule
from .mode import ModeModule
from .statistics import StatisticsModule
from .metadata import (
    MetadataModule,
    PriogridMetadata,
    CountryMetadata,
    VIEWSER_AVAILABLE,
)
from .reconciliation import ReconciliationModule
from .shape import DistributionLayout

# Optional PyTorch support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

BASE_YEAR = 1980
MONTHS_PER_YEAR = 12
DEFAULT_CACHE_SIZE = 100
MAX_WORKERS_RATIO = 0.75

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def month_id_to_date(month_id: int) -> Tuple[int, int]:
    """Convert VIEWS month_id to (year, month).
    
    Args:
        month_id: VIEWS month identifier (1-indexed from Jan 1980).
        
    Returns:
        Tuple of (year, month) where month is 1-12.
        
    Example:
        >>> month_id_to_date(1)
        (1980, 1)
        >>> month_id_to_date(529)
        (2024, 1)
    """
    year = BASE_YEAR + (month_id - 1) // MONTHS_PER_YEAR
    month = ((month_id - 1) % MONTHS_PER_YEAR) + 1
    return year, month


def date_to_month_id(year: int, month: int) -> int:
    """Convert (year, month) to VIEWS month_id.
    
    Args:
        year: Calendar year.
        month: Month (1-12).
        
    Returns:
        VIEWS month_id.
        
    Raises:
        ValueError: If month not in 1-12.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"Month must be 1-12, got {month}")
    return (year - BASE_YEAR) * MONTHS_PER_YEAR + month


def compute_cache_key(*args, **kwargs) -> str:
    """Compute a stable hash key for caching."""
    key_data = str((args, sorted(kwargs.items())))
    return hashlib.md5(key_data.encode()).hexdigest()[:16]


def polars_to_pandas_multiindex(
    df: pl.DataFrame,
    index_cols: List[str],
) -> pd.DataFrame:
    """Convert Polars DataFrame to Pandas with MultiIndex.
    
    Args:
        df: Polars DataFrame.
        index_cols: Columns to use as MultiIndex.
        
    Returns:
        Pandas DataFrame with MultiIndex.
    """
    pdf = df.to_pandas()
    pdf.set_index(index_cols, inplace=True)
    pdf.sort_index(inplace=True)
    return pdf


def get_optimal_workers(max_workers: Optional[int] = None) -> int:
    """Determine optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    auto_workers = max(1, int(cpu_count * MAX_WORKERS_RATIO))
    return min(max_workers, auto_workers) if max_workers else auto_workers


def detect_device() -> str:
    """Detect best available compute device."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Base Dataset Class
# =============================================================================

class SpatioTemporalDataset:
    """Base class for spatiotemporal datasets with tensor operations.
    
    Supports three data layouts:
        - SCALAR: Single value per (time, entity) cell
        - ARRAY_COLUMN: Distributions as pl.Array/pl.List columns (recommended)
        - ROW_BASED: Distributions as repeated rows with sample_col
        
    The layout is auto-detected from the data structure.
    
    Example:
        >>> # Scalar layout
        >>> dataset = SpatioTemporalDataset(
        ...     data="data.parquet",
        ...     time_col="month_id",
        ...     entity_col="priogrid_gid",
        ...     target_cols=["fatalities"],
        ... )
        
        >>> # Array-column layout (distributions)
        >>> dataset = SpatioTemporalDataset(
        ...     data=df_with_arrays,
        ...     time_col="month_id",
        ...     entity_col="priogrid_gid",
        ... )
        
        >>> # Access as tensor
        >>> tensor = dataset.to_tensor()  # (T, E, S, F)
    """
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, str, Path],
        time_col: str,
        entity_col: str,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = True,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        use_lazy_loading: bool = True,
    ):
        """Initialize SpatioTemporalDataset.
        
        Args:
            data: Data source (DataFrame or file path).
            time_col: Name of time index column.
            entity_col: Name of entity index column.
            sample_col: Name of sample column for row-based distributions.
                       Leave None for array-column or scalar layouts.
            target_cols: Target columns for historical mode.
            fix_structure: If True, fills missing grid points.
            auto_broadcast: If True, broadcasts scalars to match array dims.
            cache_tensors: If True, enables tensor caching.
            use_lazy_loading: If True, uses scan_parquet for large files.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize modules
        self._loader = LoaderModule(use_lazy=use_lazy_loading)
        self._grid = GridModule()
        self._subset = SubsetModule()
        self._tensor_mod = TensorModule()
        self._stats = StatisticsModule()
        
        # Load data
        self._logger.info(f"Loading data from: {type(data).__name__}")
        self.df = self._loader.load(data)
        self._logger.info(f"Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        
        # Detect distribution layout BEFORE creating index
        potential_data_cols = [
            c for c in self.df.columns 
            if c not in {time_col, entity_col, sample_col}
        ]
        self._dist_layout = DistributionLayout.detect(
            self.df, 
            sample_col=sample_col,
            data_cols=potential_data_cols,
        )
        self._logger.info(f"Distribution layout: {self._dist_layout}")
        
        # Initialize indices
        self._index, self.df = IndexModule.create(
            self.df, time_col, entity_col, 
            sample_col=sample_col,
            dist_layout=self._dist_layout,
        )
        
        # Detect mode
        has_predictions = any(c.startswith("pred_") for c in self.df.columns)
        detected_mode = "forecast" if has_predictions else "historical"
        
        if detected_mode == "historical" and target_cols is None:
            raise ValidationError(
                "Historical mode detected (no 'pred_' columns). Provide 'target_cols'."
            )
        
        self._mode = ModeModule(
            mode=detected_mode,
            target_cols=target_cols if detected_mode == "historical" else None,
        )
        
        if fix_structure:
            self.fix_space_time_consistency()
        
        self._mode.validate(self.df)
        
        # Log summary
        n_times, n_entities, n_samples = self._index.get_dimensions(
            self.df, self._dist_layout
        )
        self._logger.info(
            f"Dataset ready: mode={self.mode}, layout={self._dist_layout.layout}, "
            f"{n_times} times × {n_entities} entities × {n_samples} samples"
        )
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def time_col(self) -> str:
        """Time column name."""
        return self._index.time_col
    
    @property
    def entity_col(self) -> str:
        """Entity column name."""
        return self._index.entity_col
    
    @property
    def sample_col(self) -> Optional[str]:
        """Sample column name (None for array-column or scalar layouts)."""
        return self._index.sample_col
    
    @property
    def target_cols(self) -> List[str]:
        """Target column names."""
        return self._mode.get_targets(self.df)
    
    @property
    def mode(self) -> str:
        """Dataset mode ('historical' or 'forecast')."""
        return self._mode.mode
    
    @property
    def distribution_layout(self) -> DistributionLayout:
        """Distribution layout configuration."""
        return self._dist_layout
    
    @property
    def sample_size(self) -> int:
        """Number of samples in distribution."""
        return self._dist_layout.sample_size
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Dataset shape as (n_times, n_entities, n_samples, n_features)."""
        n_times, n_entities, n_samples = self._index.get_dimensions(
            self.df, self._dist_layout
        )
        n_features = len(self.get_all_data_cols())
        return (n_times, n_entities, n_samples, n_features)
    
    # -------------------------------------------------------------------------
    # Column Accessors
    # -------------------------------------------------------------------------
    
    def get_features(self) -> List[str]:
        """Get feature column names (excluding targets)."""
        return self._mode.get_features(self.df, self._index)
    
    def get_all_data_cols(self) -> List[str]:
        """Get all data columns (excluding indices)."""
        return self._mode.get_all_data_cols(self.df, self._index)
    
    def get_pred_vars(self) -> List[str]:
        """Get prediction column names (prefixed with 'pred_')."""
        return [c for c in self.df.columns if c.startswith("pred_")]
    
    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------
    
    def get_subset_dataframe(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get filtered subset as DataFrame.
        
        Args:
            time_ids: Time IDs to include (None = all).
            entity_ids: Entity IDs to include (None = all).
            sample_idx: Sample indices to include (None = all).
            features: Features to include (None = all).
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            Filtered DataFrame (Polars by default, or Pandas with MultiIndex).
        """
        result = self._subset.filter(
            self.df, self._index,
            time_ids=time_ids, entity_ids=entity_ids,
            sample_idx=sample_idx, features=features,
        )
        
        if return_pandas:
            index_cols = [self.time_col, self.entity_col]
            if self.sample_col and self.sample_col in result.columns:
                index_cols.append(self.sample_col)
            return polars_to_pandas_multiindex(result, index_cols)
        return result
    
    def get_subset_tensor(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Get filtered subset as 4D tensor (Time, Entity, Sample, Feature).
        
        Args:
            time_ids: Time IDs to include.
            entity_ids: Entity IDs to include.
            sample_idx: Sample indices to include.
            features: Features to include.
            
        Returns:
            4D numpy array.
        """
        feat_list = self._resolve_features(features)
        if not feat_list:
            return np.array([])
        
        sub_df = self.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids,
            sample_idx=sample_idx, features=feat_list,
        )
        
        return self._tensor_mod.convert(
            sub_df, feat_list, self._index, self._dist_layout
        )
    
    def to_tensor(self, include_targets: bool = True) -> np.ndarray:
        """Convert entire dataset to 4D tensor.
        
        Args:
            include_targets: If True, includes target columns.
            
        Returns:
            4D numpy array with shape (T, E, S, F).
        """
        features = self.get_all_data_cols() if include_targets else self.get_features()
        return self.get_subset_tensor(features=features)
    
    def split_data(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split into features (X) and targets (y).
        
        Args:
            time_ids: Time IDs to include.
            entity_ids: Entity IDs to include.
            sample_idx: Sample indices to include.
            features: Feature columns (targets auto-excluded).
            
        Returns:
            Tuple of (X, y) numpy arrays.
        """
        target_cols = self._mode.get_targets(self.df)
        if not target_cols:
            raise ValidationError("No target columns found")
        
        feat_list = [
            f for f in self._resolve_features(features) 
            if f not in target_cols
        ]
        
        X = self.get_subset_tensor(time_ids, entity_ids, sample_idx, feat_list)
        y = self.get_subset_tensor(time_ids, entity_ids, sample_idx, target_cols)
        
        return X, y
    
    def _resolve_features(
        self,
        features: Optional[Union[str, List[str]]],
    ) -> List[str]:
        """Resolve feature specification to list."""
        if features is None:
            return self.get_all_data_cols()
        if isinstance(features, str):
            return [features]
        return list(features)
    
    # -------------------------------------------------------------------------
    # Grid Operations
    # -------------------------------------------------------------------------
    
    def fix_space_time_consistency(self, fill_value: Optional[Any] = None) -> None:
        """Fill missing grid points.
        
        Ensures every (time, entity, [sample]) combination exists.
        
        Args:
            fill_value: Value to fill missing cells (None = keep as null).
        """
        self.df = self._grid.fix_consistency(
            self.df, self._index, self._dist_layout, fill_value
        )
    
    def check_grid_completeness(self) -> Tuple[bool, int]:
        """Check if grid has no missing combinations.
        
        Returns:
            Tuple of (is_complete, missing_count).
        """
        return self._grid.check_completeness(
            self.df, self._index, self._dist_layout
        )
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def calculate_hdi(
        self,
        alpha: float = 0.9,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Calculate Highest Density Interval for distributional data.
        
        Args:
            alpha: Credible mass (e.g., 0.9 for 90% HDI).
            features: Features to compute HDI for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with HDI bounds.
        """
        features = features or self.get_pred_vars() or self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]
        
        results = []
        times, entities, _ = self._index.get_unique_values(
            self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
        )
        
        for feature in features:
            tensor = self.get_subset_tensor(time_ids, entity_ids, features=[feature])
            if tensor.size == 0:
                continue
            
            data = tensor.squeeze(axis=-1)
            lower, upper = self._stats.calculate_hdi(data, alpha)
            
            for t_idx, t in enumerate(times):
                for e_idx, e in enumerate(entities):
                    results.append({
                        self.time_col: t,
                        self.entity_col: e,
                        "feature": feature,
                        "hdi_lower": lower[t_idx, e_idx],
                        "hdi_upper": upper[t_idx, e_idx],
                    })
        
        result = pl.DataFrame(results)
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def calculate_map(
        self,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        enforce_non_negative: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Calculate Maximum A Posteriori for distributional data.
        
        Args:
            features: Features to compute MAP for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            enforce_non_negative: Clip negative values to 0.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with MAP values.
        """
        features = features or self.get_pred_vars() or self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]
        
        results = []
        times, entities, _ = self._index.get_unique_values(
            self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
        )
        
        for feature in features:
            tensor = self.get_subset_tensor(time_ids, entity_ids, features=[feature])
            if tensor.size == 0:
                continue
            
            data = tensor.squeeze(axis=-1)
            map_vals = self._stats.calculate_map(data, enforce_non_negative)
            
            for t_idx, t in enumerate(times):
                for e_idx, e in enumerate(entities):
                    results.append({
                        self.time_col: t,
                        self.entity_col: e,
                        "feature": feature,
                        "map_value": map_vals[t_idx, e_idx],
                    })
        
        result = pl.DataFrame(results)
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def compute_statistics(
        self,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Compute comprehensive summary statistics.
        
        Args:
            features: Features to compute stats for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with summary statistics.
        """
        features = features or self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]
        
        results = []
        times, entities, _ = self._index.get_unique_values(
            self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
        )
        
        for feature in features:
            tensor = self.get_subset_tensor(time_ids, entity_ids, features=[feature])
            if tensor.size == 0:
                continue
            
            data = tensor.squeeze(axis=-1)
            stats = self._stats.compute_summary_statistics(data)
            
            for t_idx, t in enumerate(times):
                for e_idx, e in enumerate(entities):
                    record = {
                        self.time_col: t,
                        self.entity_col: e,
                        "feature": feature,
                    }
                    for name, arr in stats.items():
                        record[name] = arr[t_idx, e_idx]
                    results.append(record)
        
        result = pl.DataFrame(results)
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------
    
    def check_integrity(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        max_features: int = 5,
    ) -> bool:
        """Verify tensor conversion works without errors for a subset.
        
        Args:
            time_ids: Time indices to verify.
            entity_ids: Entity indices to verify.
            sample_idx: Sample indices to verify.
            features: Features to verify.
            max_features: Max features to test when features=None.
        
        Returns:
            True if tensor conversion succeeds.
        """
        try:
            feat_list = self._resolve_features(features)
            if features is None:
                feat_list = feat_list[:max_features]
            
            sub_df = self.get_subset_dataframe(
                time_ids=time_ids,
                entity_ids=entity_ids,
                sample_idx=sample_idx,
                features=feat_list,
            )
            
            if sub_df.is_empty():
                return True
            
            tensor = self._tensor_mod.convert(
                sub_df, feat_list, self._index, self._dist_layout
            )
            
            self._logger.debug(f"Integrity check passed: shape {tensor.shape}")
            return True
            
        except Exception as e:
            self._logger.error(f"Integrity check failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Multi-Dimensional Tensor Bundle
    # -------------------------------------------------------------------------
    
    def to_tensor_bundle(
        self,
        columns: Optional[List[str]] = None,
        shape_hints: Optional[Dict[str, Tuple[int, ...]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        dtype: str = "float32",
    ) -> TensorBundle:
        """Convert dataset to TensorBundle for multi-dimensional array support.
        
        Args:
            columns: Columns to convert (None = all data columns).
            shape_hints: Explicit shape overrides {col_name: (dim1, dim2, ...)}.
            time_ids: Filter to specific time indices.
            entity_ids: Filter to specific entity indices.
            dtype: Output numpy dtype.
            
        Returns:
            TensorBundle with properly shaped tensors.
        """
        if time_ids is not None or entity_ids is not None:
            df = self.get_subset_dataframe(
                time_ids=time_ids,
                entity_ids=entity_ids,
                features=columns,
            )
        else:
            df = self.df
        
        converter = TensorConverter(
            time_col=self.time_col,
            entity_col=self.entity_col,
            shape_hints=shape_hints,
            dtype=dtype,
        )
        
        return converter.convert(df, columns)
    
    def get_ml_inputs(
        self,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        shape_hints: Optional[Dict[str, Tuple[int, ...]]] = None,
        batch_format: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Get ML-ready inputs with proper tensor shapes.
        
        Args:
            feature_cols: Feature column names.
            target_cols: Target column names (optional).
            time_ids: Filter to specific time indices.
            entity_ids: Filter to specific entity indices.
            shape_hints: Explicit shape overrides.
            batch_format: If True, flatten T×E to batch dimension.
            
        Returns:
            Tuple of (X_dict, y_dict).
        """
        all_cols = list(feature_cols)
        if target_cols:
            all_cols.extend(target_cols)
        
        bundle = self.to_tensor_bundle(
            columns=all_cols,
            shape_hints=shape_hints,
            time_ids=time_ids,
            entity_ids=entity_ids,
        )
        
        X = {}
        for col in feature_cols:
            if col in bundle.columns:
                X[col] = bundle.get_batch(col) if batch_format else bundle.get(col)
        
        y = None
        if target_cols:
            y = {}
            for col in target_cols:
                if col in bundle.columns:
                    y[col] = bundle.get_batch(col) if batch_format else bundle.get(col)
        
        return X, y
    
    def __repr__(self) -> str:
        """String representation."""
        n_times, n_entities, n_samples = self._index.get_dimensions(
            self.df, self._dist_layout
        )
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', layout='{self._dist_layout.layout}', "
            f"times={n_times}, entities={n_entities}, samples={n_samples})"
        )
    
    def __len__(self) -> int:
        """Number of rows in dataset."""
        return len(self.df)


# =============================================================================
# Country-Month Dataset
# =============================================================================

class CountryMonthDataset(SpatioTemporalDataset):
    """Dataset specialized for Country-Month (CM) level data.
    
    Provides additional date utilities and metadata accessors specific
    to country-level analysis. Uses CountryMetadata for metadata operations.
    """
    
    DEFAULT_TIME_COL = "month_id"
    DEFAULT_ENTITY_COL = "country_id"
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, str, Path],
        time_col: str = DEFAULT_TIME_COL,
        entity_col: str = DEFAULT_ENTITY_COL,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = False,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        metadata_path: Optional[Union[str, Path]] = None,
        fetch_metadata: bool = False,
    ):
        """Initialize CountryMonthDataset.
        
        Args:
            data: Data source.
            time_col: Time column name.
            entity_col: Entity column name.
            sample_col: Sample column for row-based distributions.
            target_cols: Target columns for historical mode.
            fix_structure: Fill missing grid points.
            auto_broadcast: Broadcast scalars to match arrays.
            cache_tensors: Enable tensor caching.
            metadata_path: Path to country metadata file.
            fetch_metadata: If True, fetch metadata via viewser Queryset.
        """
        super().__init__(
            data=data, time_col=time_col, entity_col=entity_col,
            sample_col=sample_col, target_cols=target_cols,
            fix_structure=fix_structure, auto_broadcast=auto_broadcast,
            cache_tensors=cache_tensors,
        )
        
        # Initialize metadata handler
        self._country_meta = CountryMetadata(time_col=time_col, entity_col=entity_col)
        
        if metadata_path:
            self._country_meta.load_from_file(metadata_path)
        elif fetch_metadata:
            self._country_meta.fetch()
    
    # -------------------------------------------------------------------------
    # Date Utilities
    # -------------------------------------------------------------------------
    
    def get_year(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get year for each time ID."""
        times = self.df[self.time_col].unique().sort()
        years = [month_id_to_date(int(t))[0] for t in times.to_list()]
        result = pl.DataFrame({self.time_col: times, "year": years})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_month(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get month-of-year for each time ID."""
        times = self.df[self.time_col].unique().sort()
        months = [month_id_to_date(int(t))[1] for t in times.to_list()]
        result = pl.DataFrame({self.time_col: times, "month": months})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_date(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get full date information (year, month, date string)."""
        times = self.df[self.time_col].unique().sort()
        data = []
        for t in times.to_list():
            year, month = month_id_to_date(int(t))
            data.append((t, year, month, f"{year}-{month:02d}-01"))
        result = pl.DataFrame(data, schema=[self.time_col, "year", "month", "date"])
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_quarter(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get quarter for each time ID."""
        times = self.df[self.time_col].unique().sort()
        quarters = [
            (month_id_to_date(int(t))[1] - 1) // 3 + 1 
            for t in times.to_list()
        ]
        result = pl.DataFrame({self.time_col: times, "quarter": quarters})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    # -------------------------------------------------------------------------
    # Metadata Accessors (delegated to CountryMetadata)
    # -------------------------------------------------------------------------
    
    def get_isoab(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO country code (2-letter)."""
        return self._country_meta.get_isoab(self.df, return_pandas=return_pandas)
    
    def get_name(
        self,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country names."""
        return self._country_meta.get_name(self.df, with_id=with_id, return_pandas=return_pandas)
    
    def get_gwcode(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code."""
        return self._country_meta.get_gwcode(self.df, return_pandas=return_pandas)
    
    def get_isonum(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO numeric country code."""
        return self._country_meta.get_isonum(self.df, return_pandas=return_pandas)
    
    def get_capname(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city name."""
        return self._country_meta.get_capname(self.df, return_pandas=return_pandas)
    
    def get_cap_lat_lon(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city coordinates."""
        return self._country_meta.get_cap_lat_lon(self.df, return_pandas=return_pandas)
    
    def get_region(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region information (in_africa, in_me flags)."""
        return self._country_meta.get_region(self.df, return_pandas=return_pandas)
    
    def get_region_name(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region classification based on GW codes."""
        return self._country_meta.get_region_name(self.df, return_pandas=return_pandas)


# =============================================================================
# Priogrid-Month Dataset
# =============================================================================

class PriogridMonthDataset(SpatioTemporalDataset):
    """Dataset specialized for Priogrid-Month (PGM) level data.
    
    Includes hierarchical reconciliation capabilities for grid-to-country
    consistency. Uses PriogridMetadata for metadata operations.
    """
    
    DEFAULT_TIME_COL = "month_id"
    DEFAULT_ENTITY_COL = "priogrid_gid"
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, str, Path],
        time_col: str = DEFAULT_TIME_COL,
        entity_col: str = DEFAULT_ENTITY_COL,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = False,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        country_mapping: Optional[Union[str, Path, pl.DataFrame]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        fetch_metadata: bool = False,
    ):
        """Initialize PriogridMonthDataset.
        
        Args:
            data: Data source.
            time_col: Time column name.
            entity_col: Entity column name.
            sample_col: Sample column for row-based distributions.
            target_cols: Target columns for historical mode.
            fix_structure: Fill missing grid points.
            auto_broadcast: Broadcast scalars to match arrays.
            cache_tensors: Enable tensor caching.
            country_mapping: Grid-to-country mapping (file or DataFrame).
            metadata_path: Path to grid metadata file.
            fetch_metadata: If True, fetch metadata via viewser Queryset.
        """
        super().__init__(
            data=data, time_col=time_col, entity_col=entity_col,
            sample_col=sample_col, target_cols=target_cols,
            fix_structure=fix_structure, auto_broadcast=auto_broadcast,
            cache_tensors=cache_tensors,
        )
        
        # Initialize metadata handler
        self._pg_meta = PriogridMetadata(time_col=time_col, entity_col=entity_col)
        
        self._metadata: Optional[MetadataModule] = None
        self._reconciler: Optional[ReconciliationModule] = None
        
        if country_mapping is not None:
            self._load_country_mapping(country_mapping)
        elif metadata_path:
            self._pg_meta.load_from_file(metadata_path)
        elif fetch_metadata:
            self._pg_meta.fetch()
    
    def _load_country_mapping(
        self,
        mapping: Union[str, Path, pl.DataFrame],
    ) -> None:
        """Load grid-to-country mapping."""
        if isinstance(mapping, (str, Path)):
            path = Path(mapping)
            if path.suffix == ".parquet":
                mapping_df = pl.read_parquet(path)
            else:
                mapping_df = pl.read_csv(path)
        else:
            mapping_df = mapping.clone()
        
        # Handle column name variations
        entity_col = self.entity_col
        if "priogrid_id" in mapping_df.columns and entity_col not in mapping_df.columns:
            mapping_df = mapping_df.rename({"priogrid_id": entity_col})
        
        required = {entity_col, "country_id"}
        missing = required - set(mapping_df.columns)
        if missing:
            raise ValidationError(f"Country mapping missing columns: {missing}")
        
        self._metadata = MetadataModule(entity_col)
        self._metadata.load_from_dataframe(mapping_df, entity_col)
        
        if self._metadata._country_to_entities:
            self._reconciler = ReconciliationModule(self._metadata)
        
        self._logger.info(
            f"Country mapping loaded: {len(self._metadata._entity_to_country)} grids"
        )
    
    # -------------------------------------------------------------------------
    # Date Utilities
    # -------------------------------------------------------------------------
    
    def get_year(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get year for each time ID."""
        times = self.df[self.time_col].unique().sort()
        years = [month_id_to_date(int(t))[0] for t in times.to_list()]
        result = pl.DataFrame({self.time_col: times, "year": years})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_month(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get month-of-year for each time ID."""
        times = self.df[self.time_col].unique().sort()
        months = [month_id_to_date(int(t))[1] for t in times.to_list()]
        result = pl.DataFrame({self.time_col: times, "month": months})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_date(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get full date information (year, month, date string)."""
        times = self.df[self.time_col].unique().sort()
        data = []
        for t in times.to_list():
            year, month = month_id_to_date(int(t))
            data.append((t, year, month, f"{year}-{month:02d}-01"))
        result = pl.DataFrame(data, schema=[self.time_col, "year", "month", "date"])
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    # -------------------------------------------------------------------------
    # Metadata Accessors (delegated to PriogridMetadata)
    # -------------------------------------------------------------------------
    
    def get_lat_lon(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get geographic coordinates for grids."""
        return self._pg_meta.get_lat_lon(self.df, return_pandas=return_pandas)
    
    def get_row_col(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get row and column indices for each priogrid."""
        return self._pg_meta.get_row_col(self.df, return_pandas=return_pandas)
    
    def get_country_id(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country ID for each grid."""
        return self._pg_meta.get_country_id(self.df, return_pandas=return_pandas)
    
    def get_isoab(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO code for the country of each priogrid."""
        return self._pg_meta.get_isoab(self.df, return_pandas=return_pandas)
    
    def get_name(
        self,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country names for each priogrid."""
        return self._pg_meta.get_name(self.df, with_id=with_id, return_pandas=return_pandas)
    
    def get_gwcode(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code for each priogrid."""
        return self._pg_meta.get_gwcode(self.df, return_pandas=return_pandas)
    
    def get_region(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region classification based on GW codes."""
        return self._pg_meta.get_region(self.df, return_pandas=return_pandas)
    
    # -------------------------------------------------------------------------
    # Country-based Operations
    # -------------------------------------------------------------------------
    
    def get_subset_by_country_id(
        self,
        country_ids: Union[int, List[int]],
        time_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Filter dataset to grids belonging to specified countries.
        
        Args:
            country_ids: Country IDs to include.
            time_ids: Time IDs to include.
            sample_idx: Sample indices to include.
            features: Features to include.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            Filtered DataFrame.
        """
        target_grids = self._pg_meta.get_grids_for_countries(country_ids)
        
        if not target_grids:
            self._logger.warning(f"No grids found for countries: {country_ids}")
            if return_pandas:
                return pd.DataFrame()
            return pl.DataFrame()
        
        return self.get_subset_dataframe(
            time_ids=time_ids, entity_ids=target_grids,
            sample_idx=sample_idx, features=features,
            return_pandas=return_pandas,
        )
    
    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------
    
    def get_reconciliation_tensor(
        self,
        feature: str,
        time_id: int,
        country_id: int,
        transform: bool = True,
    ) -> Tuple[np.ndarray, List[int]]:
        """Extract grid values for reconciliation."""
        if self._metadata is None:
            raise ReconciliationError("Country mapping not loaded.")
        
        entity_ids = self._metadata.get_entities_for_country(country_id)
        if not entity_ids:
            raise ReconciliationError(f"No grids for country {country_id}")
        
        tensor = self.get_subset_tensor(
            time_ids=[time_id], entity_ids=entity_ids, features=[feature]
        )
        
        # Shape: (1, n_grids, n_samples, 1) -> (n_samples, n_grids)
        values = tensor.squeeze(axis=(0, 3)).T
        
        if transform:
            values = ReconciliationModule._transform_for_reconciliation(values, feature)
        
        return values, entity_ids
    
    def reconcile(
        self,
        feature: str,
        time_id: int,
        country_id: int,
        country_total: float,
        preserve_zeros: bool = True,
        in_place: bool = True,
    ) -> Optional[np.ndarray]:
        """Reconcile grid predictions to match country total."""
        if self._reconciler is None:
            raise ReconciliationError("Reconciler not initialized.")
        
        values, entity_ids = self.get_reconciliation_tensor(
            feature, time_id, country_id, transform=True
        )
        
        reconciled = self._reconciler.reconcile_proportional(
            values, country_total, preserve_zeros
        )
        
        reconciled = ReconciliationModule._inverse_transform(reconciled, feature)
        
        if not in_place:
            return reconciled
        
        self._logger.info(
            f"Reconciled {feature} for country {country_id} at time {time_id}"
        )
        return None


# =============================================================================
# Factory Functions
# =============================================================================

def load_pgm_dataset(
    path: Union[str, Path],
    target_cols: Optional[List[str]] = None,
    country_mapping: Optional[Union[str, Path]] = None,
    **kwargs,
) -> PriogridMonthDataset:
    """Convenience factory for PriogridMonthDataset.
    
    Args:
        path: Path to data file.
        target_cols: Target column names.
        country_mapping: Grid-to-country mapping file.
        **kwargs: Additional arguments to PriogridMonthDataset.
        
    Returns:
        Configured PriogridMonthDataset.
    """
    return PriogridMonthDataset(
        data=path, target_cols=target_cols,
        country_mapping=country_mapping, **kwargs,
    )


def load_cm_dataset(
    path: Union[str, Path],
    target_cols: Optional[List[str]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> CountryMonthDataset:
    """Convenience factory for CountryMonthDataset.
    
    Args:
        path: Path to data file.
        target_cols: Target column names.
        metadata_path: Path to country metadata file.
        **kwargs: Additional arguments to CountryMonthDataset.
        
    Returns:
        Configured CountryMonthDataset.
    """
    return CountryMonthDataset(
        data=path, target_cols=target_cols,
        metadata_path=metadata_path, **kwargs,
    )


__all__ = [
    # Dataset Classes
    "SpatioTemporalDataset",
    "CountryMonthDataset", 
    "PriogridMonthDataset",
    # Factory Functions
    "load_pgm_dataset",
    "load_cm_dataset",
    # Utilities
    "month_id_to_date",
    "date_to_month_id",
    "compute_cache_key",
    "get_optimal_workers",
    "detect_device",
    "polars_to_pandas_multiindex",
    # Constants
    "BASE_YEAR",
    "MONTHS_PER_YEAR",
    "TORCH_AVAILABLE",
    "VIEWSER_AVAILABLE",
]
