"""
Distribution Layout and Array Shape Detection
==============================================

Manages how distributional data is stored and provides shape inference
for multi-dimensional array columns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import polars as pl


logger = logging.getLogger(__name__)


# =============================================================================
# Array Shape for Multi-Dimensional Data
# =============================================================================

class ArrayShape(NamedTuple):
    """Immutable shape descriptor for efficient hashing/comparison.
    
    Used to describe the inner dimensions of array columns (excluding
    the time and entity axes which are handled separately).
    
    Attributes:
        dims: Tuple of dimension sizes (e.g., (16, 16) for a 16x16 image).
        semantic: Semantic meaning describing the data type:
            - 'scalar': Single value
            - 'distribution': Probability samples
            - 'image_2d': 2D image (H, W)
            - 'image_3d': 3D image (C, H, W)
            - 'embedding': Dense vector
            - 'array_1d': Generic 1D array
            - 'generic': Unknown/unspecified
    
    Example:
        >>> shape = ArrayShape(dims=(16, 16), semantic='image_2d')
        >>> shape.size
        256
        >>> shape.ndim
        2
    """
    dims: Tuple[int, ...]
    semantic: str = "generic"
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dims)
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        result = 1
        for d in self.dims:
            result *= d
        return result
    
    def is_broadcastable_to(self, other: "ArrayShape") -> bool:
        """Check if self can broadcast to other (numpy broadcasting rules).
        
        Args:
            other: Target shape to broadcast to.
            
        Returns:
            True if broadcasting is possible.
        """
        if self.dims == other.dims:
            return True
        if self.dims == (1,) or self.dims == ():
            return True
        for s, o in zip(reversed(self.dims), reversed(other.dims)):
            if s != o and s != 1:
                return False
        return True


def infer_shape_from_column(
    df: pl.DataFrame,
    col: str,
    shape_hints: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> ArrayShape:
    """Infer ArrayShape from column dtype and naming conventions.
    
    Naming conventions for auto-detection:
        - *_HxW or *_HxWxC → Image (e.g., satellite_16x16, rgb_32x32x3)
        - *_samples or *_dist → Distribution
        - *_emb or *_embed → Embedding
        - *_seq_L → Sequence of length L
    
    Args:
        df: DataFrame containing the column.
        col: Column name to analyze.
        shape_hints: Optional explicit shape overrides {col_name: (dim1, dim2, ...)}.
        
    Returns:
        ArrayShape with inferred dimensions and semantic type.
        
    Example:
        >>> df = pl.DataFrame({"satellite_16x16": [np.zeros(256)] * 10})
        >>> shape = infer_shape_from_column(df, "satellite_16x16")
        >>> shape
        ArrayShape(dims=(16, 16), semantic='image_2d')
    """
    dtype = df.schema[col]
    
    # Check explicit hints first
    if shape_hints and col in shape_hints:
        return ArrayShape(dims=shape_hints[col], semantic="explicit")
    
    # Scalar types
    if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
        return ArrayShape(dims=(1,), semantic="scalar")
    
    # Array columns (fixed-size)
    if isinstance(dtype, pl.Array):
        arr_size = dtype.size
        
        # Check for image naming convention: *_HxW or *_HxWxC
        match = re.search(r'_(\d+)x(\d+)(?:x(\d+))?$', col)
        if match:
            h, w = int(match.group(1)), int(match.group(2))
            c = int(match.group(3)) if match.group(3) else None
            if c and h * w * c == arr_size:
                return ArrayShape(dims=(c, h, w), semantic="image_3d")
            elif h * w == arr_size:
                return ArrayShape(dims=(h, w), semantic="image_2d")
        
        # Check for embedding naming
        if "_emb" in col.lower():
            return ArrayShape(dims=(arr_size,), semantic="embedding")
        
        # Check for distribution naming
        if "_samples" in col.lower() or "_dist" in col.lower():
            return ArrayShape(dims=(arr_size,), semantic="distribution")
        
        return ArrayShape(dims=(arr_size,), semantic="array_1d")
    
    # List columns (variable-size)
    if isinstance(dtype, pl.List):
        first = df[col].drop_nulls().head(1).to_list()
        if first and first[0]:
            return ArrayShape(dims=(len(first[0]),), semantic="list_1d")
        return ArrayShape(dims=(1,), semantic="unknown")
    
    # Object columns (numpy arrays)
    if dtype == pl.Object:
        first = df[col].drop_nulls().head(1).to_list()
        if first and hasattr(first[0], 'shape'):
            return ArrayShape(dims=first[0].shape, semantic="object_nd")
        return ArrayShape(dims=(1,), semantic="object")
    
    return ArrayShape(dims=(1,), semantic="scalar")


# =============================================================================
# Distribution Layout Detection
# =============================================================================

class DistributionLayout:
    """Detects and manages how distributional data is stored.
    
    Three layouts supported:
        - SCALAR: No distributions, single value per cell (n_samples=1)
        - ARRAY_COLUMN: Distributions stored as pl.Array/pl.List columns
        - ROW_BASED: Distributions as repeated rows with sample_col
        
    The layout determines how tensor conversion handles the sample dimension.
    
    Attributes:
        layout: One of 'scalar', 'array_column', 'row_based'.
        sample_size: Number of samples in distribution.
        array_columns: List of columns containing arrays (ARRAY_COLUMN layout).
        sample_col: Sample column name (ROW_BASED layout).
    """
    
    SCALAR = "scalar"
    ARRAY_COLUMN = "array_column"
    ROW_BASED = "row_based"
    
    def __init__(
        self,
        layout: str,
        sample_size: int = 1,
        array_columns: Optional[List[str]] = None,
        sample_col: Optional[str] = None,
    ):
        """Initialize DistributionLayout.
        
        Args:
            layout: One of SCALAR, ARRAY_COLUMN, ROW_BASED.
            sample_size: Number of samples in distribution.
            array_columns: Columns containing arrays (for ARRAY_COLUMN layout).
            sample_col: Sample column name (for ROW_BASED layout).
        """
        self.layout = layout
        self.sample_size = sample_size
        self.array_columns = array_columns or []
        self.sample_col = sample_col
    
    @classmethod
    def detect(
        cls,
        df: pl.DataFrame,
        sample_col: Optional[str] = None,
        data_cols: Optional[List[str]] = None,
    ) -> "DistributionLayout":
        """Automatically detect distribution layout from DataFrame.
        
        Detection priority:
            1. If sample_col is provided and exists → ROW_BASED
            2. If any data column is pl.Array/pl.List → ARRAY_COLUMN
            3. Otherwise → SCALAR
            
        Args:
            df: DataFrame to analyze.
            sample_col: Explicit sample column name (optional).
            data_cols: Data columns to check for arrays (optional).
            
        Returns:
            DistributionLayout instance with detected configuration.
        """
        # Check for explicit row-based layout
        if sample_col is not None and sample_col in df.columns:
            n_samples = df[sample_col].n_unique()
            logger.debug(f"Detected ROW_BASED layout with {n_samples} samples")
            return cls(
                layout=cls.ROW_BASED,
                sample_size=n_samples,
                sample_col=sample_col,
            )
        
        # Check for array columns
        cols_to_check = data_cols if data_cols else df.columns
        array_cols = []
        max_array_size = 1
        
        for col in cols_to_check:
            dtype = df.schema.get(col)
            if dtype is None:
                continue
            
            if isinstance(dtype, pl.Array):
                array_cols.append(col)
                max_array_size = max(max_array_size, dtype.size)
            elif isinstance(dtype, pl.List):
                array_cols.append(col)
                # Sample first non-null to get size
                first_val = df[col].drop_nulls().head(1).to_list()
                if first_val and first_val[0]:
                    max_array_size = max(max_array_size, len(first_val[0]))
        
        if array_cols:
            logger.debug(
                f"Detected ARRAY_COLUMN layout with {len(array_cols)} array columns, "
                f"sample_size={max_array_size}"
            )
            return cls(
                layout=cls.ARRAY_COLUMN,
                sample_size=max_array_size,
                array_columns=array_cols,
            )
        
        # Default to scalar
        logger.debug("Detected SCALAR layout (no distributions)")
        return cls(layout=cls.SCALAR, sample_size=1)
    
    @property
    def is_distributional(self) -> bool:
        """Check if layout contains distributional data."""
        return self.sample_size > 1
    
    @property
    def is_array_based(self) -> bool:
        """Check if distributions are stored in array columns."""
        return self.layout == self.ARRAY_COLUMN
    
    @property
    def is_row_based(self) -> bool:
        """Check if distributions are stored as repeated rows."""
        return self.layout == self.ROW_BASED
    
    def __repr__(self) -> str:
        return f"DistributionLayout(layout='{self.layout}', sample_size={self.sample_size})"


__all__ = [
    "ArrayShape",
    "DistributionLayout",
    "infer_shape_from_column",
]
