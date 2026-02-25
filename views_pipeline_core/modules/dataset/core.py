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
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
        time_col: str,
        entity_col: str,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = True,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
    ):
        """Initialize SpatioTemporalDataset.

        Data is stored internally as a ``pl.LazyFrame`` for deferred
        execution.  Call ``get_subset_dataframe()`` or ``collect()`` to
        materialise data.

        Args:
            data: Data source (DataFrame, LazyFrame, file, dir, or glob).
            time_col: Name of time index column.
            entity_col: Name of entity index column.
            sample_col: Name of sample column for row-based distributions.
                       Leave None for array-column or scalar layouts.
            target_cols: Target columns for historical mode.
            fix_structure: If True, auto-completes grid at query time.
            auto_broadcast: If True, broadcasts scalars to match array dims.
            cache_tensors: If True, enables tensor caching.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.warning(
            "SpatioTemporalDataset and its subclasses are in early "
            "development. API may change."
        )

        # Initialize modules
        self._loader = LoaderModule()
        self._grid = GridModule()
        self._subset = SubsetModule()
        self._tensor_mod = TensorModule(
            auto_broadcast=auto_broadcast,
            cache_enabled=cache_tensors,
        )
        self._stats = StatisticsModule()
        self._fix_structure = fix_structure

        # Load data as LazyFrame (no materialisation)
        self._logger.info(f"Loading data from: {type(data).__name__}")
        self._lf: pl.LazyFrame = self._loader.load(data)

        # Detect distribution layout from schema (no collect)
        cols = self._lf.columns
        potential_data_cols = [
            c for c in cols
            if c not in {time_col, entity_col, sample_col}
        ]
        self._dist_layout = DistributionLayout.detect(
            self._lf,
            sample_col=sample_col,
            data_cols=potential_data_cols,
        )
        self._logger.info(f"Distribution layout: {self._dist_layout}")

        # Initialize indices (validates columns, adds sort to lazy plan)
        self._index, self._lf = IndexModule.create(
            self._lf, time_col, entity_col,
            sample_col=sample_col,
            dist_layout=self._dist_layout,
        )

        # Cache lightweight index metadata (small targeted collect)
        self._cache_index_metadata()

        # Detect mode
        has_predictions = any(c.startswith("pred_") for c in cols)
        detected_mode = "forecast" if has_predictions else "historical"

        if detected_mode == "historical" and target_cols is None:
            raise ValidationError(
                "Historical mode detected (no 'pred_' columns). "
                "Provide 'target_cols'."
            )

        self._mode = ModeModule(
            mode=detected_mode,
            target_cols=target_cols if detected_mode == "historical" else None,
        )
        self._mode.validate(self._lf)

        # Log summary
        self._logger.info(
            f"Dataset ready: mode={self.mode}, "
            f"layout={self._dist_layout.layout}, "
            f"{len(self._unique_times)} times \u00d7 "
            f"{len(self._unique_entities)} entities \u00d7 "
            f"{self._dist_layout.sample_size} samples"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_index_metadata(self) -> None:
        """Cache lightweight index metadata from the LazyFrame."""
        index_cols = [self._index.time_col, self._index.entity_col]
        if self._index.sample_col:
            index_cols.append(self._index.sample_col)
        meta = self._lf.select(index_cols).unique().collect()

        self._unique_times: List[int] = (
            meta[self._index.time_col].unique().sort().to_list()
        )
        self._unique_entities: List[int] = (
            meta[self._index.entity_col].unique().sort().to_list()
        )
        self._unique_samples: Optional[List[int]] = None
        if self._index.sample_col and self._index.sample_col in meta.columns:
            self._unique_samples = (
                meta[self._index.sample_col].unique().sort().to_list()
            )
        self._n_rows: int = self._lf.select(pl.len()).collect().item()

    def _ensure_grid_complete(
        self,
        lf: pl.LazyFrame,
        time_ids: Optional[Union[int, List[int]]],
        entity_ids: Optional[Union[int, List[int]]],
        sample_idx: Optional[Union[int, List[int]]],
    ) -> pl.LazyFrame:
        """Auto-complete grid for a subset query."""
        t = list(time_ids) if time_ids is not None else self._unique_times
        if isinstance(t, int):
            t = [t]
        e = list(entity_ids) if entity_ids is not None else self._unique_entities
        if isinstance(e, int):
            e = [e]
        s = (list(sample_idx) if sample_idx is not None
             else self._unique_samples)
        if isinstance(s, int):
            s = [s]

        skeleton_rows = len(t) * len(e) * (len(s) if s else 1)
        if skeleton_rows > 500_000_000:
            self._logger.warning(
                f"Grid skeleton too large ({skeleton_rows:,} rows), "
                "skipping auto-fix"
            )
            return lf

        grid = pl.DataFrame({self._index.time_col: t}).join(
            pl.DataFrame({self._index.entity_col: e}), how="cross",
        )
        if s is not None and self._index.sample_col:
            grid = grid.join(
                pl.DataFrame({self._index.sample_col: s}), how="cross",
            )

        result = grid.lazy().join(lf, on=self._index.index_cols, how="left")
        data_cols = [
            c for c in lf.columns if c not in self._index.index_cols_set
        ]
        if data_cols:
            result = result.with_columns(
                [pl.col(c).fill_null(0.0) for c in data_cols]
            )
        return result

    def _entity_lookup_df(self) -> pl.DataFrame:
        """Minimal DataFrame with entity IDs for metadata lookups."""
        return pl.DataFrame({self.entity_col: self._unique_entities})

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
        return self._mode.get_targets(self._lf)
    
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
        n_features = len(self.get_all_data_cols())
        return (
            len(self._unique_times),
            len(self._unique_entities),
            self._dist_layout.sample_size,
            n_features,
        )

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Access the underlying LazyFrame."""
        return self._lf

    @property
    def columns(self) -> List[str]:
        """Column names."""
        return self._lf.columns

    def collect(self) -> pl.DataFrame:
        """Materialise the full dataset as a DataFrame.

        .. warning::
            For large datasets (>10M rows) this may exceed available
            RAM.  Prefer ``get_subset_dataframe()`` for targeted access.
        """
        if self._n_rows > 10_000_000:
            self._logger.warning(
                f"Collecting {self._n_rows:,} rows — consider using "
                "get_subset_dataframe() for large datasets."
            )
        return self._lf.collect()

    # -------------------------------------------------------------------------
    # Column Accessors
    # -------------------------------------------------------------------------

    def get_features(self) -> List[str]:
        """Get feature column names (excluding targets)."""
        return self._mode.get_features(self._lf, self._index)

    def get_all_data_cols(self) -> List[str]:
        """Get all data columns (excluding indices)."""
        return self._mode.get_all_data_cols(self._lf, self._index)

    def get_pred_vars(self) -> List[str]:
        """Get prediction column names (prefixed with 'pred_')."""
        return [c for c in self._lf.columns if c.startswith("pred_")]
    
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
            self._lf, self._index,
            time_ids=time_ids, entity_ids=entity_ids,
            sample_idx=sample_idx, features=features,
        )

        # Auto-fix grid for subset queries
        is_subset = time_ids is not None or entity_ids is not None
        if self._fix_structure and is_subset:
            result = self._ensure_grid_complete(
                result, time_ids, entity_ids, sample_idx,
            )

        result = result.sort(self._index.index_cols).collect()

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
        target_cols = self._mode.get_targets(self._lf)
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

        Adds the grid-fix join to the lazy plan (does not collect).

        Args:
            fill_value: Value to fill missing cells (None = keep as null).
        """
        self._lf = self._grid.fix_consistency(
            self._lf, self._index, self._dist_layout, fill_value,
            known_times=self._unique_times,
            known_entities=self._unique_entities,
            known_samples=self._unique_samples,
        )

    def check_grid_completeness(self) -> Tuple[bool, int]:
        """Check if grid has no missing combinations.

        Returns:
            Tuple of (is_complete, missing_count).
        """
        return self._grid.check_completeness(
            self._lf, self._index, self._dist_layout
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
        
        Matches the old dataset format with columns named {feature}_hdi_lower
        and {feature}_hdi_upper, with a MultiIndex of (time_col, entity_col).
        
        Args:
            alpha: Credible mass (e.g., 0.9 for 90% HDI).
            features: Features to compute HDI for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with HDI bounds. Columns are {feature}_hdi_lower and
            {feature}_hdi_upper for each feature.
        """
        features = features or self.get_pred_vars() or self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]
        
        times, entities, _ = self._index.get_unique_values(
            self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
        )
        
        # Build base index DataFrame
        index_df = pl.DataFrame({
            self.time_col: [t for t in times for _ in entities],
            self.entity_col: [e for _ in times for e in entities],
        })
        
        hdi_columns = {}
        for feature in features:
            tensor = self.get_subset_tensor(time_ids, entity_ids, features=[feature])
            if tensor.size == 0:
                continue
            
            data = tensor.squeeze(axis=-1)
            lower, upper = self._stats.calculate_hdi(data, alpha)
            
            # Flatten in row-major order (time, entity)
            hdi_columns[f"{feature}_hdi_lower"] = lower.flatten()
            hdi_columns[f"{feature}_hdi_upper"] = upper.flatten()
        
        result = index_df.with_columns([
            pl.Series(name=col_name, values=values)
            for col_name, values in hdi_columns.items()
        ])
        
        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
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
        
        Matches the old dataset format with columns named {feature}_map,
        with a MultiIndex of (time_col, entity_col).
        
        Args:
            features: Features to compute MAP for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            enforce_non_negative: Clip negative values to 0.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with MAP values. Columns are {feature}_map for each feature.
        """
        features = features or self.get_pred_vars() or self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]
        
        times, entities, _ = self._index.get_unique_values(
            self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
        )
        
        # Build base index DataFrame
        index_df = pl.DataFrame({
            self.time_col: [t for t in times for _ in entities],
            self.entity_col: [e for _ in times for e in entities],
        })
        
        map_columns = {}
        for feature in features:
            tensor = self.get_subset_tensor(time_ids, entity_ids, features=[feature])
            if tensor.size == 0:
                continue
            
            data = tensor.squeeze(axis=-1)
            map_vals = self._stats.calculate_map(data, enforce_non_negative)
            
            # Flatten in row-major order (time, entity)
            map_columns[f"{feature}_map"] = map_vals.flatten()
        
        result = index_df.with_columns([
            pl.Series(name=col_name, values=values)
            for col_name, values in map_columns.items()
        ])
        
        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
        return result
    
    def calculate_hdi_map(
        self,
        alpha: float = 0.9,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        enforce_non_negative: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Calculate HDI and MAP in a single efficient pass.
        
        This is more efficient than calling calculate_hdi and calculate_map
        separately as it processes each cell once. Uses multithreading for
        large datasets.
        
        Matches the old dataset format with columns named {feature}_hdi_lower,
        {feature}_hdi_upper, and {feature}_map.
        
        Args:
            alpha: Credible mass (e.g., 0.9 for 90% HDI).
            features: Features to compute for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            enforce_non_negative: Clip negative MAP values to 0.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with HDI bounds and MAP values.
        """
        features = features or self.get_pred_vars() or self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]
        
        times, entities, _ = self._index.get_unique_values(
            self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
        )
        
        # Build base index DataFrame
        index_df = pl.DataFrame({
            self.time_col: [t for t in times for _ in entities],
            self.entity_col: [e for _ in times for e in entities],
        })
        
        all_columns = {}
        for feature in features:
            tensor = self.get_subset_tensor(time_ids, entity_ids, features=[feature])
            if tensor.size == 0:
                continue
            
            data = tensor.squeeze(axis=-1)
            lower, upper, map_vals = self._stats.calculate_hdi_map(
                data, alpha, enforce_non_negative
            )
            
            # Flatten in row-major order (time, entity)
            all_columns[f"{feature}_hdi_lower"] = lower.flatten()
            all_columns[f"{feature}_hdi_upper"] = upper.flatten()
            all_columns[f"{feature}_map"] = map_vals.flatten()
        
        result = index_df.with_columns([
            pl.Series(name=col_name, values=values)
            for col_name, values in all_columns.items()
        ])
        
        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
        return result
    
    def compute_statistics(
        self,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Compute comprehensive summary statistics.
        
        Matches the old dataset format with columns named {feature}_{metric}
        and a MultiIndex of (time_col, entity_col) when returning pandas.
        
        Args:
            features: Features to compute stats for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with summary statistics. Columns are named {feature}_{metric}
            where metric is one of: mean, std, q05, q25, q50, q75, q95, q98, q100.
        """
        features = features or self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]
        
        times, entities, _ = self._index.get_unique_values(
            self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
        )
        
        # Build base index DataFrame
        index_df = pl.DataFrame({
            self.time_col: [t for t in times for _ in entities],
            self.entity_col: [e for _ in times for e in entities],
        })
        
        stat_columns = {}
        for feature in features:
            tensor = self.get_subset_tensor(time_ids, entity_ids, features=[feature])
            if tensor.size == 0:
                continue
            
            data = tensor.squeeze(axis=-1)
            stats = self._stats.compute_summary_statistics(data)
            
            # Flatten each metric in row-major order (time, entity)
            for metric_name, metric_arr in stats.items():
                stat_columns[f"{feature}_{metric_name}"] = metric_arr.flatten()
        
        if not stat_columns:
            # Return empty DataFrame with proper structure
            if return_pandas:
                empty_df = pd.DataFrame()
                empty_df.index = pd.MultiIndex.from_tuples(
                    [], names=[self.time_col, self.entity_col]
                )
                return empty_df
            return index_df.head(0)
        
        result = index_df.with_columns([
            pl.Series(name=col_name, values=values)
            for col_name, values in stat_columns.items()
        ])
        
        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
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
            df = self._lf.collect()
        
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
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', layout='{self._dist_layout.layout}', "
            f"times={len(self._unique_times)}, "
            f"entities={len(self._unique_entities)}, "
            f"samples={self._dist_layout.sample_size})"
        )

    def __len__(self) -> int:
        """Number of rows in dataset."""
        return self._n_rows


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
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
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
            fix_structure: Auto-complete grid at query time.
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
        times = pl.Series(self.time_col, self._unique_times)
        years = [month_id_to_date(int(t))[0] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "year": years})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_month(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get month-of-year for each time ID."""
        times = pl.Series(self.time_col, self._unique_times)
        months = [month_id_to_date(int(t))[1] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "month": months})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_date(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get full date information (year, month, date string)."""
        data = []
        for t in self._unique_times:
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
        times = pl.Series(self.time_col, self._unique_times)
        quarters = [
            (month_id_to_date(int(t))[1] - 1) // 3 + 1 
            for t in self._unique_times
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
        return self._country_meta.get_isoab(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_name(
        self,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country names."""
        return self._country_meta.get_name(self._entity_lookup_df(), with_id=with_id, return_pandas=return_pandas)
    
    def get_gwcode(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code."""
        return self._country_meta.get_gwcode(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_isonum(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO numeric country code."""
        return self._country_meta.get_isonum(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_capname(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city name."""
        return self._country_meta.get_capname(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_cap_lat_lon(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city coordinates."""
        return self._country_meta.get_cap_lat_lon(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_region(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region information (in_africa, in_me flags)."""
        return self._country_meta.get_region(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_region_name(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region classification based on GW codes."""
        return self._country_meta.get_region_name(self._entity_lookup_df(), return_pandas=return_pandas)


# =============================================================================
# Priogrid-Month Dataset
# =============================================================================

class PriogridMonthDataset(SpatioTemporalDataset):
    """Dataset specialized for Priogrid-Month (PGM) level data.
    
    Includes hierarchical reconciliation capabilities for grid-to-country
    consistency. Uses PriogridMetadata for metadata operations.
    """
    
    DEFAULT_TIME_COL = "month_id"
    DEFAULT_ENTITY_COL = "priogrid_gd"
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
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
            fix_structure: Auto-complete grid at query time.
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
        
        # Reconciled dataframe - populated by reconcile()
        self.reconciled_dataframe: Optional[pl.DataFrame] = None
        
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
        times = pl.Series(self.time_col, self._unique_times)
        years = [month_id_to_date(int(t))[0] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "year": years})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_month(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get month-of-year for each time ID."""
        times = pl.Series(self.time_col, self._unique_times)
        months = [month_id_to_date(int(t))[1] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "month": months})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_date(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get full date information (year, month, date string)."""
        data = []
        for t in self._unique_times:
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
        return self._pg_meta.get_lat_lon(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_row_col(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get row and column indices for each priogrid."""
        return self._pg_meta.get_row_col(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_country_id(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country ID for each grid."""
        return self._pg_meta.get_country_id(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_isoab(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO code for the country of each priogrid."""
        return self._pg_meta.get_isoab(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_name(
        self,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country names for each priogrid."""
        return self._pg_meta.get_name(self._entity_lookup_df(), with_id=with_id, return_pandas=return_pandas)
    
    def get_gwcode(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code for each priogrid."""
        return self._pg_meta.get_gwcode(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_region(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region classification based on GW codes."""
        return self._pg_meta.get_region(self._entity_lookup_df(), return_pandas=return_pandas)
    
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
    
    def to_reconciler(
        self,
        feature: str,
        time_id: int,
        country_id: int,
    ) -> Tuple[np.ndarray, List[int]]:
        """Extract grid values for ForecastReconciler.
        
        Extracts values in natural scale (untransformed) for reconciliation.
        
        Args:
            feature: Prediction feature to extract.
            time_id: Time ID to extract.
            country_id: Country ID whose grids to extract.
            
        Returns:
            Tuple of (values, entity_ids) where values has shape (n_samples, n_grids)
            in natural scale (exp applied to ln_ features).
        """
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
        
        # Transform to natural scale for reconciliation
        values = ReconciliationModule._transform_for_reconciliation(values, feature)
        
        return values, entity_ids
    
    def reconcile(
        self,
        country_id: int,
        feature: str,
        reconciled_values: np.ndarray,
        time_id: int,
    ) -> None:
        """Update reconciled_dataframe with reconciled values.
        
        Writes reconciled values back to the dataframe for a specific
        country's grid cells at a specified time_id.
        
        Args:
            country_id: Country ID whose grid cells to update.
            feature: Prediction feature to update.
            reconciled_values: Array of reconciled values (n_samples, n_grids)
                              in natural scale (will be log-transformed if needed).
            time_id: Time ID to update.
            
        Example:
            >>> # Get tensor for reconciliation
            >>> values, entity_ids = dataset.to_reconciler('pred_sb', 529, 475)
            >>> # Reconcile using ForecastReconciler
            >>> reconciled = reconciler.reconcile_forecast(values, country_total)
            >>> # Write back to dataframe
            >>> dataset.reconcile(475, 'pred_sb', reconciled.numpy(), 529)
            >>> # Access result
            >>> df = dataset.reconciled_dataframe
        """
        if self._metadata is None:
            raise ReconciliationError("Country mapping not loaded.")
        
        # Initialize reconciled dataframe on first call
        if self.reconciled_dataframe is None:
            self.reconciled_dataframe = self._lf.collect()
        
        entity_ids = self._metadata.get_entities_for_country(country_id)
        if not entity_ids:
            raise ReconciliationError(f"No grids for country {country_id}")
        
        if reconciled_values.shape[1] != len(entity_ids):
            raise ReconciliationError(
                f"Values shape {reconciled_values.shape} doesn't match "
                f"{len(entity_ids)} grid cells in country {country_id}"
            )
        
        # Transform back to original scale (log for ln_ features)
        reconciled_values = ReconciliationModule._inverse_transform(
            reconciled_values, feature
        )
        
        # Update each grid cell in the reconciled dataframe
        for idx, entity_id in enumerate(entity_ids):
            new_samples = reconciled_values[:, idx].tolist()
            
            # Find the row to update
            mask = (
                (pl.col(self.time_col) == time_id) & 
                (pl.col(self.entity_col) == entity_id)
            )
            
            # Update the feature column with new samples
            self.reconciled_dataframe = self.reconciled_dataframe.with_columns(
                pl.when(mask)
                .then(pl.lit(new_samples))
                .otherwise(pl.col(feature))
                .alias(feature)
            )
        
        self._logger.info(
            f"Reconciled {feature} for country {country_id} at time {time_id}"
        )


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
