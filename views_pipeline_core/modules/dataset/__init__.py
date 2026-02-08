"""
VIEWS Dataset Module
====================

High-performance spatiotemporal dataset handling for conflict forecasting.

This module provides:
    - SpatioTemporalDataset: Base class for spatiotemporal data
    - CountryMonthDataset: Country-month level specialization
    - PriogridMonthDataset: Priogrid-month level with reconciliation
    - TensorBundle: Multi-dimensional array container
    - Distribution layout detection and conversion

Key Features:
    - Automatic distribution layout detection (scalar, array, row-based)
    - Efficient tensor conversion with shape inference
    - Hierarchical forecast reconciliation
    - HDI and MAP statistical computations
    - PyTorch integration with device detection

Example:
    >>> from views_pipeline_core.modules.dataset import (
    ...     PriogridMonthDataset,
    ...     load_pgm_dataset,
    ... )
    >>> 
    >>> # Load a dataset
    >>> dataset = load_pgm_dataset(
    ...     "predictions.parquet",
    ...     target_cols=["fatalities"],
    ... )
    >>> 
    >>> # Convert to tensor
    >>> tensor = dataset.to_tensor()  # (T, E, S, F)
    >>> 
    >>> # Get ML-ready inputs
    >>> X, y = dataset.split_data()
"""

# Exceptions
from .exceptions import (
    DatasetError,
    ValidationError,
    IntegrityError,
    ReconciliationError,
    MetadataError,
    TensorConversionError,
)

# Shape and layout
from .shape import (
    ArrayShape,
    DistributionLayout,
    infer_shape_from_column,
)

# Modules
from .loader import LoaderModule
from .index import IndexModule
from .grid import GridModule
from .subset import SubsetModule
from .tensor import (
    TensorBundle,
    TensorConverter,
    TensorModule,
    detect_device,
    TORCH_AVAILABLE,
)
from .mode import ModeModule
from .statistics import StatisticsModule, ZERO_MASS_THRESHOLD
from .metadata import MetadataModule, PriogridMetadata, CountryMetadata, VIEWSER_AVAILABLE
from .reconciliation import ReconciliationModule

# Core dataset classes
from .core import (
    SpatioTemporalDataset,
    CountryMonthDataset,
    PriogridMonthDataset,
    load_pgm_dataset,
    load_cm_dataset,
    month_id_to_date,
    date_to_month_id,
    compute_cache_key,
    get_optimal_workers,
    BASE_YEAR,
    MONTHS_PER_YEAR,
)


__all__ = [
    # Exceptions
    "DatasetError",
    "ValidationError",
    "IntegrityError",
    "ReconciliationError",
    "MetadataError",
    "TensorConversionError",
    # Shape and layout
    "ArrayShape",
    "DistributionLayout",
    "infer_shape_from_column",
    # Tensor support
    "TensorBundle",
    "TensorConverter",
    "TensorModule",
    "detect_device",
    "TORCH_AVAILABLE",
    # Modules
    "LoaderModule",
    "IndexModule",
    "GridModule",
    "SubsetModule",
    "ModeModule",
    "StatisticsModule",
    "ZERO_MASS_THRESHOLD",
    "MetadataModule",
    "PriogridMetadata",
    "CountryMetadata",
    "VIEWSER_AVAILABLE",
    "ReconciliationModule",
    # Dataset classes
    "SpatioTemporalDataset",
    "CountryMonthDataset",
    "PriogridMonthDataset",
    # Factory functions
    "load_pgm_dataset",
    "load_cm_dataset",
    # Utilities
    "month_id_to_date",
    "date_to_month_id",
    "compute_cache_key",
    "get_optimal_workers",
    # Constants
    "BASE_YEAR",
    "MONTHS_PER_YEAR",
]
