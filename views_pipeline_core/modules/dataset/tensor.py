"""
Tensor Module
=============

High-performance tensor conversion and multi-dimensional array handling.

This module provides:
    - TensorBundle: Container for heterogeneous tensor data
    - TensorConverter: Conversion from DataFrames to tensors with shape inference
    - TensorModule: Integration layer for SpatioTemporalDataset

Optimizations:
    - Zero-copy from Polars Arrow buffers where possible
    - Single contiguous buffer per column
    - Lazy reshape - stores flat, reshapes on access
    - Broadcasting tracked but not materialized until needed
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import polars as pl

from .shape import ArrayShape, infer_shape_from_column

# Optional PyTorch support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from .index import IndexModule
    from .shape import DistributionLayout


def detect_device() -> str:
    """Detect best available compute device.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TensorBundle:
    """High-performance container for heterogeneous tensor data.
    
    Provides optimized access to multiple tensors with different shapes,
    supporting automatic broadcasting and ML-ready batch conversion.
    
    Optimizations:
        - Single contiguous buffer per column
        - Zero-copy slicing via numpy views
        - Lazy reshape - stores flat, reshapes on access
        - Broadcasting tracked but not materialized until needed
    
    Attributes:
        n_times: Number of time steps.
        n_entities: Number of spatial entities.
    
    Example:
        >>> bundle = TensorBundle(n_times=100, n_entities=500)
        >>> bundle.add_column("rainfall", data, shape)
        >>> batch = bundle.get_batch("rainfall")  # (50000, *inner_dims)
    """
    
    n_times: int
    n_entities: int
    _buffers: Dict[str, np.ndarray] = field(default_factory=dict)
    _shapes: Dict[str, ArrayShape] = field(default_factory=dict)
    
    def add_column(self, name: str, data: np.ndarray, shape: ArrayShape) -> None:
        """Add a column to the bundle.
        
        Args:
            name: Column identifier.
            data: Flat or shaped numpy array.
            shape: ArrayShape describing inner dimensions.
        """
        self._buffers[name] = data
        self._shapes[name] = shape
    
    def get(
        self,
        name: str,
        reshape: bool = True,
    ) -> np.ndarray:
        """Get tensor, optionally reshaped to full dimensions.
        
        Args:
            name: Column name.
            reshape: If True, reshape to (T, E, *inner_dims).
            
        Returns:
            NumPy array with requested shape.
        """
        data = self._buffers[name]
        shape = self._shapes[name]
        
        if not reshape:
            return data
        
        target_shape = (self.n_times, self.n_entities) + shape.dims
        return data.reshape(target_shape)
    
    def get_batch(
        self,
        name: str,
        time_slice: Optional[slice] = None,
        entity_slice: Optional[slice] = None,
    ) -> np.ndarray:
        """Get tensor slice in batch format (T*E, *dims).
        
        Args:
            name: Column name.
            time_slice: Optional time range slice.
            entity_slice: Optional entity range slice.
            
        Returns:
            Batch-formatted tensor with T×E flattened to first dimension.
        """
        full = self.get(name, reshape=True)
        
        if time_slice is not None:
            full = full[time_slice]
        if entity_slice is not None:
            full = full[:, entity_slice]
        
        batch_size = full.shape[0] * full.shape[1]
        return full.reshape((batch_size,) + full.shape[2:])
    
    def broadcast_scalar_to(self, scalar_col: str, target_col: str) -> np.ndarray:
        """Broadcast scalar column to match target's inner dimensions.
        
        Uses numpy broadcasting views (no memory copy).
        
        Args:
            scalar_col: Name of scalar column to broadcast.
            target_col: Name of target column (for shape reference).
            
        Returns:
            Broadcasted array matching target shape.
        """
        scalar_data = self.get(scalar_col, reshape=True)
        target_shape = self._shapes[target_col]
        
        broadcast_shape = (self.n_times, self.n_entities) + tuple(1 for _ in target_shape.dims)
        return np.broadcast_to(
            scalar_data.reshape(broadcast_shape),
            (self.n_times, self.n_entities) + target_shape.dims
        )
    
    @property
    def columns(self) -> List[str]:
        """List of column names in bundle."""
        return list(self._buffers.keys())
    
    @property  
    def shapes(self) -> Dict[str, ArrayShape]:
        """Dictionary of column shapes."""
        return self._shapes.copy()
    
    def memory_usage(self) -> Dict[str, int]:
        """Memory usage in bytes per column.
        
        Returns:
            Dictionary mapping column names to byte counts.
        """
        return {name: arr.nbytes for name, arr in self._buffers.items()}
    
    def total_memory(self) -> int:
        """Total memory usage in bytes.
        
        Returns:
            Sum of all buffer sizes.
        """
        return sum(arr.nbytes for arr in self._buffers.values())
    
    def to_torch(
        self,
        name: str,
        device: Optional[str] = None,
        batch_format: bool = True,
    ) -> "torch.Tensor":
        """Convert column to PyTorch tensor.
        
        Args:
            name: Column name.
            device: Target device ('cuda', 'mps', 'cpu'). Auto-detected if None.
            batch_format: If True, flatten T×E to batch dimension.
            
        Returns:
            PyTorch tensor on specified device.
            
        Raises:
            ImportError: If PyTorch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for to_torch()")
        
        data = self.get_batch(name) if batch_format else self.get(name)
        device = device or detect_device()
        return torch.from_numpy(data.copy()).to(device=device, dtype=torch.float32)


class TensorConverter:
    """High-performance tensor conversion with automatic shape inference.
    
    Converts DataFrame columns to properly-shaped tensors based on:
        - Explicit shape hints
        - Column naming conventions (e.g., *_16x16 for images)
        - Polars dtype analysis
    
    Optimizations:
        - Zero-copy from Polars Arrow buffers where possible
        - Pre-allocated output buffers
        - Vectorized operations only
    
    Example:
        >>> converter = TensorConverter(time_col="month_id", entity_col="priogrid_gid")
        >>> bundle = converter.convert(df, columns=["rainfall", "satellite_16x16"])
    """
    
    def __init__(
        self,
        time_col: str = "month_id",
        entity_col: str = "priogrid_gid",
        shape_hints: Optional[Dict[str, Tuple[int, ...]]] = None,
        dtype: str = "float32",
    ):
        """Initialize converter.
        
        Args:
            time_col: Name of time index column.
            entity_col: Name of entity index column.
            shape_hints: Explicit shape overrides {col_name: (dim1, dim2, ...)}.
            dtype: Output numpy dtype string.
        """
        self.time_col = time_col
        self.entity_col = entity_col
        self.shape_hints = shape_hints or {}
        self.dtype = np.dtype(dtype)
    
    def convert(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        exclude_index: bool = True,
    ) -> TensorBundle:
        """Convert DataFrame to TensorBundle with optimized memory layout.
        
        Args:
            df: Sorted DataFrame (by time, entity).
            columns: Columns to convert (None = all data columns).
            exclude_index: Whether to exclude time/entity columns.
            
        Returns:
            TensorBundle with properly shaped tensors.
        """
        times = df[self.time_col].unique().sort()
        entities = df[self.entity_col].unique().sort()
        n_times, n_entities = len(times), len(entities)
        
        if columns is None:
            columns = [c for c in df.columns 
                      if c not in {self.time_col, self.entity_col}]
        elif exclude_index:
            columns = [c for c in columns 
                      if c not in {self.time_col, self.entity_col}]
        
        bundle = TensorBundle(n_times=n_times, n_entities=n_entities)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            shape = infer_shape_from_column(df, col, self.shape_hints)
            data = self._extract_column(df, col, shape, n_times, n_entities)
            bundle.add_column(col, data, shape)
        
        return bundle
    
    def _extract_column(
        self,
        df: pl.DataFrame,
        col: str,
        shape: ArrayShape,
        n_times: int,
        n_entities: int,
    ) -> np.ndarray:
        """Extract column data with minimal copies.
        
        Args:
            df: Source DataFrame.
            col: Column name.
            shape: Inferred or explicit shape.
            n_times: Number of time steps.
            n_entities: Number of entities.
            
        Returns:
            NumPy array with data.
        """
        dtype = df.schema[col]
        n_rows = n_times * n_entities
        
        # Scalar types - direct conversion
        if shape.semantic == "scalar" or dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            arr = df[col].to_numpy()
            return arr.astype(self.dtype, copy=False).reshape(n_rows, 1)
        
        # Fixed-size arrays - try zero-copy
        if isinstance(dtype, pl.Array):
            try:
                raw = df[col].to_numpy()
                if raw.dtype != object:
                    return raw.astype(self.dtype, copy=False)
                else:
                    return np.stack(raw).astype(self.dtype)
            except Exception:
                return np.array([np.array(x) for x in df[col].to_list()], 
                               dtype=self.dtype)
        
        # Variable-length lists
        if isinstance(dtype, pl.List):
            raw = df[col].to_list()
            return np.array([np.array(x) if x is not None else np.array([]) 
                            for x in raw], dtype=object)
        
        # Object dtype (numpy arrays stored in cells)
        if dtype == pl.Object:
            raw = df[col].to_list()
            if raw and hasattr(raw[0], '__array__'):
                return np.stack([np.asarray(x) for x in raw]).astype(self.dtype)
        
        # Fallback - treat as scalar
        return df[col].to_numpy().astype(self.dtype, copy=False).reshape(n_rows, 1)


class TensorModule:
    """Tensor conversion module for SpatioTemporalDataset.
    
    Provides integration between the dataset and tensor conversion utilities.
    Handles conversion to 4D tensors with shape (Time, Entity, Sample, Feature).
    """
    
    def __init__(
        self,
        auto_broadcast: bool = True,
        cache_enabled: bool = True,
        dtype: str = "float32",
    ):
        """Initialize TensorModule.
        
        Args:
            auto_broadcast: If True, broadcasts scalars to match array dims.
            cache_enabled: If True, enables tensor caching.
            dtype: Default numpy dtype for conversions.
        """
        self.auto_broadcast = auto_broadcast
        self.cache_enabled = cache_enabled
        self.dtype = np.dtype(dtype)
        self._cache: Dict[str, np.ndarray] = {}
        self._logger = logging.getLogger(f"{__name__}.TensorModule")
    
    def clear_cache(self) -> None:
        """Clear the tensor cache."""
        self._cache.clear()
    
    def convert(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        index_mgr: "IndexModule",
        dist_layout: "DistributionLayout",
        use_cache: bool = True,
    ) -> np.ndarray:
        """Convert DataFrame to 4D tensor (Time, Entity, Sample, Feature).
        
        Args:
            df: Filtered DataFrame.
            feature_cols: Columns to include.
            index_mgr: Index configuration.
            dist_layout: Distribution layout configuration.
            use_cache: Use cached result if available.
            
        Returns:
            4D array with shape (n_times, n_entities, n_samples, n_features).
        """
        from .exceptions import ValidationError, TensorConversionError
        
        if df.is_empty() or not feature_cols:
            self._logger.warning("Empty DataFrame or no features")
            return np.array([])
        
        # Validate features
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValidationError(f"Features not found: {missing}")
        
        # Route to appropriate conversion method based on layout
        if dist_layout.is_array_based:
            return self._convert_array_layout(df, feature_cols, index_mgr, dist_layout)
        elif dist_layout.is_row_based:
            return self._convert_row_layout(df, feature_cols, index_mgr, dist_layout)
        else:
            return self._convert_scalar_layout(df, feature_cols, index_mgr)
    
    def _convert_scalar_layout(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        index_mgr: "IndexModule",
    ) -> np.ndarray:
        """Convert scalar layout to tensor with n_samples=1."""
        from .exceptions import TensorConversionError
        
        times, entities, _ = index_mgr.get_unique_values(df)
        n_times, n_entities = len(times), len(entities)
        n_features = len(feature_cols)
        
        expected_rows = n_times * n_entities
        if len(df) != expected_rows:
            raise TensorConversionError(
                f"Row count {len(df)} != expected {expected_rows}. Grid may have gaps."
            )
        
        values = df.select(feature_cols).to_numpy()
        tensor = values.reshape(n_times, n_entities, n_features)
        tensor = np.expand_dims(tensor, axis=2)  # Add sample dimension
        
        return tensor
    
    def _convert_array_layout(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        index_mgr: "IndexModule",
        dist_layout: "DistributionLayout",
    ) -> np.ndarray:
        """Convert array-column layout to tensor."""
        from .exceptions import TensorConversionError
        
        times, entities, _ = index_mgr.get_unique_values(df)
        n_times, n_entities = len(times), len(entities)
        n_samples = dist_layout.sample_size
        n_features = len(feature_cols)
        
        expected_rows = n_times * n_entities
        if len(df) != expected_rows:
            raise TensorConversionError(
                f"Row count {len(df)} != expected {expected_rows}. Grid may have gaps."
            )
        
        tensor = np.empty((n_times, n_entities, n_samples, n_features), dtype=np.float64)
        
        for f_idx, col in enumerate(feature_cols):
            dtype = df.schema[col]
            
            if isinstance(dtype, (pl.Array, pl.List)):
                arr = df[col].to_numpy()
                
                if arr.dtype == object:
                    stacked = np.stack([
                        np.array(x) if x is not None else np.full(n_samples, np.nan)
                        for x in arr
                    ])
                else:
                    stacked = arr
                
                reshaped = stacked.reshape(n_times, n_entities, -1)
                actual_samples = reshaped.shape[2]
                
                if actual_samples < n_samples:
                    if self.auto_broadcast and actual_samples == 1:
                        reshaped = np.repeat(reshaped, n_samples, axis=2)
                    else:
                        padded = np.full((n_times, n_entities, n_samples), np.nan)
                        padded[:, :, :actual_samples] = reshaped
                        reshaped = padded
                elif actual_samples > n_samples:
                    reshaped = reshaped[:, :, :n_samples]
                
                tensor[:, :, :, f_idx] = reshaped
            else:
                values = df[col].to_numpy().reshape(n_times, n_entities)
                tensor[:, :, :, f_idx] = np.expand_dims(values, axis=2)
        
        return tensor
    
    def _convert_row_layout(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        index_mgr: "IndexModule",
        dist_layout: "DistributionLayout",
    ) -> np.ndarray:
        """Convert row-based layout to tensor."""
        from .exceptions import TensorConversionError
        
        times, entities, samples = index_mgr.get_unique_values(df)
        n_times, n_entities, n_samples = len(times), len(entities), len(samples)
        n_features = len(feature_cols)
        
        expected_rows = n_times * n_entities * n_samples
        if len(df) != expected_rows:
            raise TensorConversionError(
                f"Row count {len(df)} != expected {expected_rows}. Grid may have gaps."
            )
        
        values = df.select(feature_cols).to_numpy()
        tensor = values.reshape(n_samples, n_times, n_entities, n_features)
        tensor = tensor.transpose(1, 2, 0, 3)
        
        return tensor
    
    def to_tensor_bundle(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        columns: Optional[List[str]] = None,
        shape_hints: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> TensorBundle:
        """Convert DataFrame to TensorBundle.
        
        Args:
            df: DataFrame to convert.
            index_mgr: Index configuration.
            columns: Columns to include (None = all data columns).
            shape_hints: Explicit shape overrides.
            
        Returns:
            TensorBundle with converted data.
        """
        converter = TensorConverter(
            time_col=index_mgr.time_col,
            entity_col=index_mgr.entity_col,
            shape_hints=shape_hints,
            dtype=str(self.dtype),
        )
        return converter.convert(df, columns)
    
    def to_numpy(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Convert DataFrame to numpy array.
        
        Simple 2D conversion without shape inference.
        
        Args:
            df: DataFrame to convert.
            index_mgr: Index configuration.
            columns: Columns to include.
            
        Returns:
            2D numpy array of shape (n_rows, n_cols).
        """
        if columns is None:
            columns = [c for c in df.columns if c not in index_mgr.index_cols_set]
        
        return df.select(columns).to_numpy().astype(self.dtype)
    
    def to_torch(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        columns: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> "torch.Tensor":
        """Convert DataFrame to PyTorch tensor.
        
        Args:
            df: DataFrame to convert.
            index_mgr: Index configuration.
            columns: Columns to include.
            device: Target device ('cuda', 'mps', 'cpu').
            
        Returns:
            PyTorch tensor.
            
        Raises:
            ImportError: If PyTorch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for to_torch()")
        
        arr = self.to_numpy(df, index_mgr, columns)
        device = device or detect_device()
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


__all__ = [
    "TensorBundle",
    "TensorConverter",
    "TensorModule",
    "detect_device",
    "TORCH_AVAILABLE",
]
